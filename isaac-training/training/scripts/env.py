import torch
import einops
import numpy as np
from omegaconf import OmegaConf
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec, DiscreteTensorSpec
from omni_drones.envs.isaac_env import IsaacEnv, AgentSpec
import omni.isaac.orbit.sim as sim_utils
# Topology-based graph navigation modules (pure PyTorch, no Isaac dependency)
# Imported lazily inside __init__ so the file is importable without Isaac when use_topo=False
from omni_drones.robots.drone import MultirotorBase
from omni.isaac.orbit.assets import AssetBaseCfg
from omni.isaac.orbit.terrains import TerrainImporterCfg, TerrainImporter, TerrainGeneratorCfg, HfDiscreteObstaclesTerrainCfg, FlatPatchSamplingCfg
from omni_drones.utils.torch import euler_to_quaternion, quat_axis
from omni.isaac.orbit.sensors import RayCaster, RayCasterCfg, patterns
from omni.isaac.core.utils.viewports import set_camera_view
from utils import vec_to_new_frame, vec_to_world, construct_input
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.orbit.sim as sim_utils
import omni.isaac.orbit.utils.math as math_utils
from omni.isaac.orbit.assets import RigidObject, RigidObjectCfg
import time

class NavigationEnv(IsaacEnv):

    # In one step:
    # 1. _pre_sim_step (apply action) -> step isaac sim
    # 2. _post_sim_step (update lidar)
    # 3. increment progress_buf
    # 4. _compute_state_and_obs (get observation and states, update stats)
    # 5. _compute_reward_and_done (update reward and calculate returns)

    def __init__(self, cfg):
        print("[Navigation Environment]: Initializing Env...")
        # LiDAR params:
        self.lidar_range = cfg.sensor.lidar_range
        self.lidar_vfov = (max(-89., cfg.sensor.lidar_vfov[0]), min(89., cfg.sensor.lidar_vfov[1]))
        self.lidar_vbeams = cfg.sensor.lidar_vbeams
        self.lidar_hres = cfg.sensor.lidar_hres
        self.lidar_hbeams = int(360/self.lidar_hres)
        # Spawn mode:
        #   edge          → original curriculum: spawn on 4 edges of the map
        #   interior_safe → sample from valid flat interior patches (free space between obstacles)
        self.spawn_mode = str(OmegaConf.select(cfg, 'env.spawn_mode', default='edge'))
        self.interior_spawn_num_patches = int(OmegaConf.select(cfg, 'env.interior_spawn_num_patches', default=512))
        self.interior_spawn_patch_radius = float(OmegaConf.select(cfg, 'env.interior_spawn_patch_radius', default=0.30))
        self.interior_spawn_max_height_diff = float(OmegaConf.select(cfg, 'env.interior_spawn_max_height_diff', default=0.10))

        # Graph-based topology navigation modules.
        # Require BOTH use_topo=true AND mode='graph_ppo' to be active.
        # This prevents accidental activation when mode=ppo (which would try to read
        # the velocity from tensordict["agents","action"] after VelController has already
        # replaced it with motor thrusts).
        self.use_topo = (
            OmegaConf.select(cfg, 'topo', default=None) is not None
            and OmegaConf.select(cfg, 'topo.use_topo', default=False)
            and OmegaConf.select(cfg, 'mode', default='ppo') == 'graph_ppo'
        )

        super().__init__(cfg, cfg.headless)
        
        # Drone Initialization
        self.drone.initialize()

        # lee_controller will be injected by train.py after construction so that
        # _pre_sim_step_graph can convert ctrl_vel → motor thrust without going
        # through VelController (which runs before _pre_sim_step).
        self.lee_controller = None
        if self.use_topo:
            import sys, os
            sys.path.insert(0, os.path.dirname(__file__))
            from topo_extractor import TopoExtractor
            from safety_shield import SafetyShieldQP
            from hierarchical_control import HierarchicalController
            self.topo_extractor = TopoExtractor(cfg.topo)
            self.safety_shield  = SafetyShieldQP(
                relaxation_weight=float(cfg.topo.qp_relaxation_weight),
                max_relaxation=float(cfg.topo.qp_max_relaxation),
                v_max=float(cfg.topo.qp_v_max),
                safe_distance=float(cfg.topo.qp_safe_distance),
                cbf_alpha=float(cfg.topo.qp_cbf_alpha),
            )
            self.hierarchical_controller = HierarchicalController(
                high_freq=float(cfg.topo.high_freq),
                low_freq=float(cfg.topo.low_freq),
                pid_kp=float(cfg.topo.pid_kp),
                pid_kd=float(cfg.topo.pid_kd),
                pid_ki=float(cfg.topo.pid_ki),
                goal_horizon=float(cfg.topo.goal_horizon),
                integral_limit=float(cfg.topo.pid_integral_limit),
            )
            self.hierarchical_controller.reset(self.num_envs, self.device)
            # Cache for last topo output (used in _pre_sim_step)
            self._last_topo_node_positions = None
            self._last_intervention        = None   # (B,) set by _pre_sim_step_graph
            self._last_tracking_error      = None   # (B,) set by _pre_sim_step_graph
        self.init_vels = torch.zeros_like(self.drone.get_velocities())

        # ── Privilege-free RewardShaper — always active for ALL training modes.
        #    Curriculum plugin: if enabled (mode=ppo only), overrides preset per stage.
        #    graph_ppo uses the default full_navigation preset.
        from reward_shaping import RewardShaper
        _cur_cfg = OmegaConf.select(cfg, 'curriculum', default=None)
        self.curriculum_enabled = (
            _cur_cfg is not None
            and OmegaConf.select(_cur_cfg, 'enabled', default=False)
            and OmegaConf.select(cfg, 'mode', default='ppo') == 'ppo'
        )
        self.curriculum_manager = None
        if self.curriculum_enabled:
            from curriculum_manager import CurriculumManager
            self.curriculum_manager = CurriculumManager(_cur_cfg)
            sc = self.curriculum_manager.get_stage_config()
            self.reward_shaper = RewardShaper(
                preset=sc["reward_preset"],
                speed_target=sc["speed_target"],
                speed_sigma=sc["speed_sigma"],
            )
            # Override spawn_mode and spawn_radius from curriculum stage
            self.spawn_mode = sc["spawn_mode"]
            self._curriculum_spawn_radius = sc.get("spawn_radius", 3.0)
            print(f"[Navigation Environment]: Curriculum enabled — Stage {self.curriculum_manager.stage} ({sc['reward_preset']})")
        else:
            _speed_target = float(OmegaConf.select(cfg, 'reward.speed_target', default=3.0))
            _speed_sigma  = float(OmegaConf.select(cfg, 'reward.speed_sigma',  default=1.5))
            self.reward_shaper = RewardShaper(
                preset="full_navigation",
                speed_target=_speed_target,
                speed_sigma=_speed_sigma,
            )
            print(f"[Navigation Environment]: Privilege-free reward — full_navigation "
                  f"(speed_target={_speed_target}, sigma={_speed_sigma})")
        # Keep a class reference for fast static-method calls in _compute_state_and_obs.
        self._RS = RewardShaper


        # LiDAR Intialization
        ray_caster_cfg = RayCasterCfg(
            prim_path="/World/envs/env_.*/Hummingbird_0/base_link",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
            attach_yaw_only=True,
            # attach_yaw_only=False,
            pattern_cfg=patterns.BpearlPatternCfg(
                horizontal_res=self.lidar_hres, # horizontal default is set to 10
                vertical_ray_angles=torch.linspace(*self.lidar_vfov, self.lidar_vbeams) 
            ),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
            # mesh_prim_paths=["/World"],
        )
        self.lidar = RayCaster(ray_caster_cfg)
        self.lidar._initialize_impl()
        self.lidar_resolution = (self.lidar_hbeams, self.lidar_vbeams) 
        
        # start and target 
        with torch.device(self.device):
            # self.start_pos = torch.zeros(self.num_envs, 1, 3)
            self.target_pos = torch.zeros(self.num_envs, 1, 3)
            
            # Coordinate change: add target direction variable
            self.target_dir = torch.zeros(self.num_envs, 1, 3)
            self.height_range = torch.zeros(self.num_envs, 1, 2)
            self.prev_drone_vel_w = torch.zeros(self.num_envs, 1 , 3)
            # self.target_pos[:, 0, 0] = torch.linspace(-0.5, 0.5, self.num_envs) * 32.
            # self.target_pos[:, 0, 1] = 24.
            # self.target_pos[:, 0, 2] = 2.     


    def _design_scene(self):
        # Initialize a drone in prim /World/envs/envs_0
        drone_model = MultirotorBase.REGISTRY[self.cfg.drone.model_name] # drone model class
        cfg = drone_model.cfg_cls(force_sensor=False)
        self.drone = drone_model(cfg=cfg)
        # drone_prim = self.drone.spawn(translations=[(0.0, 0.0, 1.0)])[0]
        drone_prim = self.drone.spawn(translations=[(0.0, 0.0, 2.0)])[0]

        # Lighting — warm directional sun + cool ambient dome
        light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DistantLightCfg(color=(1.0, 0.95, 0.85), intensity=4000.0),
        )
        sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(color=(0.4, 0.5, 0.7), intensity=1200.0),
        )
        light.spawn.func(light.prim_path, light.spawn, light.init_state.pos)
        sky_light.spawn.func(sky_light.prim_path, sky_light.spawn)

        # Ground Plane — dark charcoal for contrast
        cfg_ground = sim_utils.GroundPlaneCfg(color=(0.05, 0.05, 0.07), size=(300., 300.))
        cfg_ground.func("/World/defaultGroundPlane", cfg_ground, translation=(0, 0, 0.01))

        # Scene boundary definitions.
        # map_range: obstacle field half-extent.
        # terrain_half_extent_*: actual traversable terrain half-extent after adding border.
        # spawn_target_edge_*: start/goal reset line, kept 1 m inside the terrain edge.
        # OOB truncation happens exactly at the terrain edge, not at an arbitrary magic number.
        self.map_range = [20.0, 20.0, 4.5]
        self.terrain_border_width = 2.0
        self.terrain_half_extent_x = self.map_range[0] + self.terrain_border_width
        self.terrain_half_extent_y = self.map_range[1] + self.terrain_border_width
        self.spawn_target_margin_xy = 1.0
        self.spawn_target_edge_x = self.terrain_half_extent_x - self.spawn_target_margin_xy
        self.spawn_target_edge_y = self.terrain_half_extent_y - self.spawn_target_margin_xy
        self.z_terminate_min = 0.2
        self.z_terminate_max = 4.0

        obstacle_cfg_kwargs = dict(
            horizontal_scale=0.1,
            vertical_scale=0.1,
            border_width=0.0,
            num_obstacles=self.cfg.env.num_obstacles,
            obstacle_height_mode="range",
            obstacle_width_range=(0.4, 1.1),
            obstacle_height_range=[1.0, 1.5, 2.0, 4.0, 6.0],
            obstacle_height_probability=[0.1, 0.15, 0.20, 0.55],
            platform_width=0.0,
        )
        # Always bake flat patches when curriculum is enabled (worst-case stage init).
        # Stage 3 switches spawn_mode to "interior_safe" at runtime; if flat patches were
        # not baked at startup the terrain importer has no valid patch bank and
        # _sample_safe_interior_positions() raises RuntimeError.
        if self.spawn_mode == "interior_safe" or self.curriculum_enabled:
            obstacle_cfg_kwargs["flat_patch_sampling"] = {
                "init_pos": FlatPatchSamplingCfg(
                    num_patches=self.interior_spawn_num_patches,
                    patch_radius=self.interior_spawn_patch_radius,
                    x_range=(-self.map_range[0] + 1.0, self.map_range[0] - 1.0),
                    y_range=(-self.map_range[1] + 1.0, self.map_range[1] - 1.0),
                    z_range=(-0.05, 0.15),
                    max_height_diff=self.interior_spawn_max_height_diff,
                ),
            }

        terrain_cfg = TerrainImporterCfg(
            num_envs=self.num_envs,
            env_spacing=0.0,
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=TerrainGeneratorCfg(
                seed=0,
                size=(self.map_range[0]*2, self.map_range[1]*2), 
                border_width=self.terrain_border_width,
                num_rows=1, 
                num_cols=1, 
                horizontal_scale=0.1,
                vertical_scale=0.1,
                slope_threshold=0.75,
                use_cache=False,
                color_scheme="height",
                sub_terrains={
                    "obstacles": HfDiscreteObstaclesTerrainCfg(**obstacle_cfg_kwargs),
                },
            ),
            visual_material = None,
            max_init_terrain_level=None,
            collision_group=-1,
            debug_vis=True,
        )
        try:
            self.terrain_importer = TerrainImporter(terrain_cfg)
        except RuntimeError as exc:
            if self.spawn_mode == "interior_safe" and "Failed to find valid patches" in str(exc):
                raise RuntimeError(
                    "interior_safe 出生模式初始化失败：当前地形中过少可用空地，无法采样足够的 flat patches。\n"
                    "可尝试减小 env.interior_spawn_num_patches、减小 env.interior_spawn_patch_radius，"
                    "或增大 env.interior_spawn_max_height_diff。"
                ) from exc
            raise

        if (self.cfg.env_dyn.num_obstacles == 0):
            return
        # Dynamic Obstacles
        # NOTE: we use cuboid to represent 3D dynamic obstacles which can float in the air 
        # and the long cylinder to represent 2D dynamic obstacles for which the drone can only pass in 2D 
        # The width of the dynamic obstacles is divided into N_w=4 bins
        # [[0, 0.25], [0.25, 0.50], [0.50, 0.75], [0.75, 1.0]]
        # The height of the dynamic obstacles is divided into N_h=2 bins
        # [[0, 0.5], [0.5, inf]] we want to distinguish 3D obstacles and 2d obstacles
        N_w = 4 # number of width intervals between [0, 1]
        N_h = 2 # number of height: current only support binary
        max_obs_width = 1.0
        self.max_obs_3d_height = 1.0
        self.max_obs_2d_height = 5.0
        self.dyn_obs_width_res = max_obs_width/float(N_w)
        dyn_obs_category_num = N_w * N_h
        self.dyn_obs_num_of_each_category = int(self.cfg.env_dyn.num_obstacles / dyn_obs_category_num)
        self.cfg.env_dyn.num_obstacles = self.dyn_obs_num_of_each_category * dyn_obs_category_num # in case of the roundup error


        # Dynamic obstacle info
        self.dyn_obs_list = []
        self.dyn_obs_state = torch.zeros((self.cfg.env_dyn.num_obstacles, 13), dtype=torch.float, device=self.cfg.device) # 13 is based on the states from sim, we only care the first three which is position
        self.dyn_obs_state[:, 3] = 1. # Quaternion
        self.dyn_obs_goal = torch.zeros((self.cfg.env_dyn.num_obstacles, 3), dtype=torch.float, device=self.cfg.device)
        self.dyn_obs_origin = torch.zeros((self.cfg.env_dyn.num_obstacles, 3), dtype=torch.float, device=self.cfg.device)
        self.dyn_obs_vel = torch.zeros((self.cfg.env_dyn.num_obstacles, 3), dtype=torch.float, device=self.cfg.device)
        self.dyn_obs_step_count = 0 # dynamic obstacle motion step count
        self.dyn_obs_size = torch.zeros((self.cfg.env_dyn.num_obstacles, 3), dtype=torch.float, device=self.device) # size of dynamic obstacles


        # helper function to check pos validity for even distribution condition
        def check_pos_validity(prev_pos_list, curr_pos, adjusted_obs_dist):
            for prev_pos in prev_pos_list:
                if (np.linalg.norm(curr_pos - prev_pos) <= adjusted_obs_dist):
                    return False
            return True            
        
        obs_dist = 2 * np.sqrt(self.map_range[0] * self.map_range[1] / self.cfg.env_dyn.num_obstacles) # prefered distance between each dynamic obstacle
        curr_obs_dist = obs_dist
        prev_pos_list = [] # for distance check
        cuboid_category_num = cylinder_category_num = int(dyn_obs_category_num/N_h)
        for category_idx in range(cuboid_category_num + cylinder_category_num):
            # create all origins for 3D dynamic obstacles of this category (size)
            for origin_idx in range(self.dyn_obs_num_of_each_category):
                # random sample an origin until satisfy the evenly distributed condition
                start_time = time.time()
                while (True):
                    ox = np.random.uniform(low=-self.map_range[0], high=self.map_range[0])
                    oy = np.random.uniform(low=-self.map_range[1], high=self.map_range[1])
                    if (category_idx < cuboid_category_num):
                        oz = np.random.uniform(low=0.0, high=self.map_range[2]) 
                    else:
                        oz = self.max_obs_2d_height/2. # half of the height
                    curr_pos = np.array([ox, oy])
                    valid = check_pos_validity(prev_pos_list, curr_pos, curr_obs_dist)
                    curr_time = time.time()
                    if (curr_time - start_time > 0.1):
                        curr_obs_dist *= 0.8
                        start_time = time.time()
                    if (valid):
                        prev_pos_list.append(curr_pos)
                        break
                curr_obs_dist = obs_dist
                origin = [ox, oy, oz]
                self.dyn_obs_origin[origin_idx+category_idx*self.dyn_obs_num_of_each_category] = torch.tensor(origin, dtype=torch.float, device=self.cfg.device)     
                self.dyn_obs_state[origin_idx+category_idx*self.dyn_obs_num_of_each_category, :3] = torch.tensor(origin, dtype=torch.float, device=self.cfg.device)                        
                prim_utils.create_prim(f"/World/Origin{origin_idx+category_idx*self.dyn_obs_num_of_each_category}", "Xform", translation=origin)

            # Spawn various sizes of dynamic obstacles 
            if (category_idx < cuboid_category_num):
                # spawn for 3D dynamic obstacles
                obs_width = width = float(category_idx+1) * max_obs_width/float(N_w)
                obs_height = self.max_obs_3d_height
                cuboid_cfg = RigidObjectCfg(
                    prim_path=f"/World/Origin{construct_input(category_idx*self.dyn_obs_num_of_each_category, (category_idx+1)*self.dyn_obs_num_of_each_category)}/Cuboid",
                    spawn=sim_utils.CuboidCfg(
                        size=[width, width, self.max_obs_3d_height],
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                        # Orange-brown for 3D dynamic obstacles (warm, easy to spot)
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.40, 0.10), metallic=0.3),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(),
                )
                dynamic_obstacle = RigidObject(cfg=cuboid_cfg)
            else:
                radius = float(category_idx-cuboid_category_num+1) * max_obs_width/float(N_w) / 2.
                obs_width = radius * 2
                obs_height = self.max_obs_2d_height
                # spawn for 2D dynamic obstacles
                cylinder_cfg = RigidObjectCfg(
                    prim_path=f"/World/Origin{construct_input(category_idx*self.dyn_obs_num_of_each_category, (category_idx+1)*self.dyn_obs_num_of_each_category)}/Cylinder",
                    spawn=sim_utils.CylinderCfg(
                        radius = radius,
                        height = self.max_obs_2d_height,
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                        # Slate-blue for 2D dynamic obstacles (contrasts with 3D)
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.45, 0.75), metallic=0.3),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(),
                )
                dynamic_obstacle = RigidObject(cfg=cylinder_cfg)
            self.dyn_obs_list.append(dynamic_obstacle)
            self.dyn_obs_size[category_idx*self.dyn_obs_num_of_each_category:(category_idx+1)*self.dyn_obs_num_of_each_category] \
                = torch.tensor([obs_width, obs_width, obs_height], dtype=torch.float, device=self.cfg.device)



    def move_dynamic_obstacle(self):
        # Step 1: Random sample new goals for required update dynamic obstacles
        # Check whether the current dynamic obstacles need new goals
        dyn_obs_goal_dist = torch.sqrt(torch.sum((self.dyn_obs_state[:, :3] - self.dyn_obs_goal)**2, dim=1)) if self.dyn_obs_step_count !=0 \
            else torch.zeros(self.dyn_obs_state.size(0), device=self.cfg.device)
        dyn_obs_new_goal_mask = dyn_obs_goal_dist < 0.5 # change to a new goal if less than the threshold
        
        # sample new goals in local range
        num_new_goal = torch.sum(dyn_obs_new_goal_mask)
        sample_x_local = -self.cfg.env_dyn.local_range[0] + 2. * self.cfg.env_dyn.local_range[0] * torch.rand(num_new_goal, 1, dtype=torch.float, device=self.cfg.device)
        sample_y_local = -self.cfg.env_dyn.local_range[1] + 2. * self.cfg.env_dyn.local_range[1] * torch.rand(num_new_goal, 1, dtype=torch.float, device=self.cfg.device)
        sample_z_local = -self.cfg.env_dyn.local_range[2] + 2. * self.cfg.env_dyn.local_range[2] * torch.rand(num_new_goal, 1, dtype=torch.float, device=self.cfg.device)
        sample_goal_local = torch.cat([sample_x_local, sample_y_local, sample_z_local], dim=1)
    
        # apply local goal to the global range
        self.dyn_obs_goal[dyn_obs_new_goal_mask] = self.dyn_obs_origin[dyn_obs_new_goal_mask] + sample_goal_local
        # clamp the range if out of the static env range
        self.dyn_obs_goal[:, 0] = torch.clamp(self.dyn_obs_goal[:, 0], min=-self.map_range[0], max=self.map_range[0])
        self.dyn_obs_goal[:, 1] = torch.clamp(self.dyn_obs_goal[:, 1], min=-self.map_range[1], max=self.map_range[1])
        self.dyn_obs_goal[:, 2] = torch.clamp(self.dyn_obs_goal[:, 2], min=0., max=self.map_range[2])
        self.dyn_obs_goal[int(self.dyn_obs_goal.size(0)/2):, 2] = self.max_obs_2d_height/2. # for 2d obstacles


        # Step 2: Random sample velocity for roughly every 2 seconds
        if (self.dyn_obs_step_count % int(2.0/self.cfg.sim.dt) == 0):
            self.dyn_obs_vel_norm = self.cfg.env_dyn.vel_range[0] + (self.cfg.env_dyn.vel_range[1] \
              - self.cfg.env_dyn.vel_range[0]) * torch.rand(self.dyn_obs_vel.size(0), 1, dtype=torch.float, device=self.cfg.device)
            self.dyn_obs_vel = self.dyn_obs_vel_norm * \
                (self.dyn_obs_goal - self.dyn_obs_state[:, :3])/torch.norm((self.dyn_obs_goal - self.dyn_obs_state[:, :3]), dim=1, keepdim=True)

        # Step 3: Calculate new position update for current timestep
        self.dyn_obs_state[:, :3] += self.dyn_obs_vel * self.cfg.sim.dt


        # Step 4: Update Visualized Location in Simulation
        for category_idx, dynamic_obstacle in enumerate(self.dyn_obs_list):
            dynamic_obstacle.write_root_state_to_sim(self.dyn_obs_state[category_idx*self.dyn_obs_num_of_each_category:(category_idx+1)*self.dyn_obs_num_of_each_category]) 
            dynamic_obstacle.write_data_to_sim()
            dynamic_obstacle.update(self.cfg.sim.dt)

        self.dyn_obs_step_count += 1


    def _set_specs(self):
        observation_dim = 8
        num_dim_each_dyn_obs_state = 10

        # Build base observation sub-spec (before expand)
        obs_spec_dict = {
            "state": UnboundedContinuousTensorSpec((observation_dim,), device=self.device),
            "lidar": UnboundedContinuousTensorSpec((1, self.lidar_hbeams, self.lidar_vbeams), device=self.device),
            "direction": UnboundedContinuousTensorSpec((1, 3), device=self.device),
            "dynamic_obstacle": UnboundedContinuousTensorSpec((1, self.cfg.algo.feature_extractor.dyn_obs_num, num_dim_each_dyn_obs_state), device=self.device),
        }

        # Optional graph observation keys — must be added BEFORE expand() so that
        # the batch dimension is correctly prepended by expand(num_envs).
        if self.use_topo:
            _N  = self.cfg.topo.max_nodes       # candidate nodes (ego excluded)
            _nf = self.cfg.topo.node_feat_dim   # 18
            _ef = self.cfg.topo.edge_feat_dim   # 8
            obs_spec_dict["node_features"]  = UnboundedContinuousTensorSpec((_N + 1, _nf), device=self.device)
            obs_spec_dict["node_positions"] = UnboundedContinuousTensorSpec((_N + 1, 3), device=self.device)
            obs_spec_dict["edge_features"]  = UnboundedContinuousTensorSpec((_N + 1, _N + 1, _ef), device=self.device)
            # masks stored as float (0.0/1.0) – cast to bool before use in GraphTransformer
            obs_spec_dict["node_mask"] = UnboundedContinuousTensorSpec((_N + 1,), device=self.device)
            obs_spec_dict["edge_mask"] = UnboundedContinuousTensorSpec((_N + 1, _N + 1), device=self.device)

        # Observation Spec
        self.observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": CompositeSpec(obs_spec_dict),
            }).expand(self.num_envs)
        }, shape=[self.num_envs], device=self.device)

        # Action Spec
        self.action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "action": self.drone.action_spec, # number of motor
            })
        }).expand(self.num_envs).to(self.device)
        
        # Reward Spec
        self.reward_spec = CompositeSpec({
            "agents": CompositeSpec({
                "reward": UnboundedContinuousTensorSpec((1,))
            })
        }).expand(self.num_envs).to(self.device)

        # Done Spec
        self.done_spec = CompositeSpec({
            "done": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "terminated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "truncated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
        }).expand(self.num_envs).to(self.device) 


        stats_spec = CompositeSpec({
            "return": UnboundedContinuousTensorSpec(1),
            "episode_len": UnboundedContinuousTensorSpec(1),
            "reach_goal": UnboundedContinuousTensorSpec(1),
            "collision": UnboundedContinuousTensorSpec(1),
            "truncated": UnboundedContinuousTensorSpec(1),
            "distance_to_goal": UnboundedContinuousTensorSpec(1),
            "speed": UnboundedContinuousTensorSpec(1),
            # Privilege-free reward components (hardware-available signals)
            "reward_progress": UnboundedContinuousTensorSpec(1),   # vel · dir_hat × 10
            "reward_heading":  UnboundedContinuousTensorSpec(1),   # cos(vel, dir) × 5
            "reward_speed":    UnboundedContinuousTensorSpec(1),   # Gaussian speed-band × 3
            "reward_safety_static": UnboundedContinuousTensorSpec(1),
            "reward_safety_dynamic": UnboundedContinuousTensorSpec(1),
            "penalty_smooth": UnboundedContinuousTensorSpec(1),
            "penalty_height": UnboundedContinuousTensorSpec(1),
            "qp_intervention": UnboundedContinuousTensorSpec(1),
            "tracking_error": UnboundedContinuousTensorSpec(1),
        }).expand(self.num_envs).to(self.device)

        info_spec = CompositeSpec({
            "drone_state": UnboundedContinuousTensorSpec((self.drone.n, 13), device=self.device),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.observation_spec["info"] = info_spec
        self.stats = stats_spec.zero()
        self.info = info_spec.zero()

    
    def reset_target(self, env_ids: torch.Tensor):
        if (self.training):
            # ── Curriculum override: stage-aware target placement ──
            if self.curriculum_enabled and self.curriculum_manager is not None:
                self._reset_target_curriculum(env_ids)
                return

            # decide which side
            masks = torch.tensor([[1., 0., 1.], [1., 0., 1.], [0., 1., 1.], [0., 1., 1.]], dtype=torch.float, device=self.device)
            edge_x = self.spawn_target_edge_x
            edge_y = self.spawn_target_edge_y
            shifts = torch.tensor([[0., edge_y, 0.], [0., -edge_y, 0.], [edge_x, 0., 0.], [-edge_x, 0., 0.]], dtype=torch.float, device=self.device)
            mask_indices = np.random.randint(0, masks.size(0), size=env_ids.size(0))
            selected_masks = masks[mask_indices].unsqueeze(1)
            selected_shifts = shifts[mask_indices].unsqueeze(1)


            # generate random positions
            target_pos = torch.zeros(env_ids.size(0), 1, 3, dtype=torch.float, device=self.device)
            target_pos[:, 0, 0] = (2.0 * edge_x) * torch.rand(env_ids.size(0), dtype=torch.float, device=self.device) - edge_x
            target_pos[:, 0, 1] = (2.0 * edge_y) * torch.rand(env_ids.size(0), dtype=torch.float, device=self.device) - edge_y
            heights = 0.5 + torch.rand(env_ids.size(0), dtype=torch.float, device=self.device) * (2.5 - 0.5)
            target_pos[:, 0, 2] = heights# height
            target_pos = target_pos * selected_masks + selected_shifts
            
            # apply target pos
            self.target_pos[env_ids] = target_pos

            # self.target_pos[:, 0, 0] = torch.linspace(-0.5, 0.5, self.num_envs) * 32.
            # self.target_pos[:, 0, 1] = 24.
            # self.target_pos[:, 0, 2] = 2.    
        else:
            self.target_pos[:, 0, 0] = torch.linspace(-0.5, 0.5, self.num_envs) * 32.
            self.target_pos[:, 0, 1] = -self.spawn_target_edge_y
            self.target_pos[:, 0, 2] = 2.            

    def _reset_target_curriculum(self, env_ids: torch.Tensor):
        """Place targets according to curriculum stage config."""
        sc = self.curriculum_manager.get_stage_config()
        B = env_ids.size(0)
        target_pos = torch.zeros(B, 1, 3, dtype=torch.float, device=self.device)
        d_min, d_max = sc["target_distance"]

        if sc["target_mode"] == "direction":
            # Random bearing + distance from (0, 0) — used in Stage 1
            angles = torch.rand(B, device=self.device) * 2.0 * torch.pi
            dists = d_min + torch.rand(B, device=self.device) * (d_max - d_min)
            target_pos[:, 0, 0] = dists * torch.cos(angles)
            target_pos[:, 0, 1] = dists * torch.sin(angles)
        else:
            # "point" mode — targets on map edges (same as original reset_target)
            edge_x = self.spawn_target_edge_x
            edge_y = self.spawn_target_edge_y
            masks = torch.tensor([[1., 0., 1.], [1., 0., 1.], [0., 1., 1.], [0., 1., 1.]],
                                 dtype=torch.float, device=self.device)
            shifts = torch.tensor([[0., edge_y, 0.], [0., -edge_y, 0.],
                                   [edge_x, 0., 0.], [-edge_x, 0., 0.]],
                                  dtype=torch.float, device=self.device)
            idx = np.random.randint(0, 4, size=B)
            target_pos[:, 0, 0] = (2.0 * edge_x) * torch.rand(B, device=self.device) - edge_x
            target_pos[:, 0, 1] = (2.0 * edge_y) * torch.rand(B, device=self.device) - edge_y
            target_pos = target_pos * masks[idx].unsqueeze(1) + shifts[idx].unsqueeze(1)

        heights = 0.5 + torch.rand(B, device=self.device) * (2.5 - 0.5)
        target_pos[:, 0, 2] = heights
        self.target_pos[env_ids] = target_pos

    def apply_curriculum_stage(self) -> None:
        """Re-read current curriculum stage config and update env parameters.

        Called by the training loop after ``CurriculumManager.advance()``.
        """
        if not self.curriculum_enabled:
            return
        sc = self.curriculum_manager.get_stage_config()
        # Update spawn mode and spawn radius
        self.spawn_mode = sc.get("spawn_mode", self.spawn_mode)
        self._curriculum_spawn_radius = float(sc.get("spawn_radius", getattr(self, '_curriculum_spawn_radius', 3.0)))
        # Update max episode length
        new_max = sc.get("max_episode_length", None)
        if new_max is not None:
            self.max_episode_length = int(new_max)
        # Update reward shaper preset, speed target & speed sigma from stage config
        self.reward_shaper.update_config(
            preset=sc["reward_preset"],
            speed_target=sc.get("speed_target", self.reward_shaper.speed_target),
            speed_sigma=sc.get("speed_sigma", self.reward_shaper.speed_sigma),
        )

    def _sample_safe_interior_positions(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Sample interior spawn positions from precomputed valid flat patches.

        Orbit's flat-patch sampler already performs terrain-wide rejection sampling against
        the generated obstacle mesh. We reuse that valid patch bank here instead of doing
        expensive mesh collision queries inside every reset.
        """
        valid_patches = getattr(self.terrain_importer, 'flat_patches', {}).get("init_pos", None)
        if valid_patches is None:
            raise RuntimeError(
                "env.spawn_mode='interior_safe' requires terrain flat patches under 'init_pos'. "
                "Check TerrainGeneratorCfg.flat_patch_sampling in env.py."
            )

        flat_positions = valid_patches.reshape(-1, 3)
        sample_ids = torch.randint(0, flat_positions.shape[0], (env_ids.size(0),), device=self.device)
        pos = flat_positions[sample_ids].unsqueeze(1).clone()  # (B, 1, 3)
        heights = 0.5 + torch.rand(env_ids.size(0), dtype=torch.float, device=self.device) * (2.5 - 0.5)
        pos[:, 0, 2] = heights
        return pos


    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids, self.training)
        self.reset_target(env_ids)
        if (self.training):
            if self.spawn_mode == "edge":
                masks = torch.tensor([[1., 0., 1.], [1., 0., 1.], [0., 1., 1.], [0., 1., 1.]], dtype=torch.float, device=self.device)
                edge_x = self.spawn_target_edge_x
                edge_y = self.spawn_target_edge_y
                shifts = torch.tensor([[0., edge_y, 0.], [0., -edge_y, 0.], [edge_x, 0., 0.], [-edge_x, 0., 0.]], dtype=torch.float, device=self.device)
                mask_indices = np.random.randint(0, masks.size(0), size=env_ids.size(0))
                selected_masks = masks[mask_indices].unsqueeze(1)
                selected_shifts = shifts[mask_indices].unsqueeze(1)

                # generate random positions on the four map edges
                pos = torch.zeros(env_ids.size(0), 1, 3, dtype=torch.float, device=self.device)
                pos[:, 0, 0] = (2.0 * edge_x) * torch.rand(env_ids.size(0), dtype=torch.float, device=self.device) - edge_x
                pos[:, 0, 1] = (2.0 * edge_y) * torch.rand(env_ids.size(0), dtype=torch.float, device=self.device) - edge_y
                heights = 0.5 + torch.rand(env_ids.size(0), dtype=torch.float, device=self.device) * (2.5 - 0.5)
                pos[:, 0, 2] = heights
                pos = pos * selected_masks + selected_shifts
            elif self.spawn_mode == "interior_safe":
                # Sample from obstacle-free interior empty space.
                pos = self._sample_safe_interior_positions(env_ids)
            elif self.spawn_mode == "center":
                # Curriculum Stage 1: spawn near map centre within a small radius.
                radius = float(getattr(self, '_curriculum_spawn_radius', 3.0))
                pos = torch.zeros(env_ids.size(0), 1, 3, dtype=torch.float, device=self.device)
                angles = torch.rand(env_ids.size(0), device=self.device) * 2.0 * torch.pi
                dists = torch.rand(env_ids.size(0), device=self.device) * radius
                pos[:, 0, 0] = dists * torch.cos(angles)
                pos[:, 0, 1] = dists * torch.sin(angles)
                heights = 0.5 + torch.rand(env_ids.size(0), dtype=torch.float, device=self.device) * (2.5 - 0.5)
                pos[:, 0, 2] = heights
            else:
                raise ValueError(f"Unknown env.spawn_mode='{self.spawn_mode}'. Expected 'edge', 'interior_safe', or 'center'.")
            
            # pos = torch.zeros(len(env_ids), 1, 3, device=self.device)
            # pos[:, 0, 0] = (env_ids / self.num_envs - 0.5) * 32.
            # pos[:, 0, 1] = -24.
            # pos[:, 0, 2] = 2.
        else:
            pos = torch.zeros(len(env_ids), 1, 3, device=self.device)
            pos[:, 0, 0] = (env_ids / self.num_envs - 0.5) * 32.
            pos[:, 0, 1] = self.spawn_target_edge_y
            pos[:, 0, 2] = 2.
        
        # Coordinate change: after reset, the drone's target direction should be changed
        self.target_dir[env_ids] = self.target_pos[env_ids] - pos

        # Coordinate change: after reset, the drone's facing direction should face the current goal
        rpy = torch.zeros(len(env_ids), 1, 3, device=self.device)
        diff = self.target_pos[env_ids] - pos
        facing_yaw = torch.atan2(diff[..., 1], diff[..., 0])
        rpy[..., 2] = facing_yaw

        rot = euler_to_quaternion(rpy)
        self.drone.set_world_poses(pos, rot, env_ids)
        self.drone.set_velocities(self.init_vels[env_ids], env_ids)
        self.prev_drone_vel_w[env_ids] = 0.
        self.height_range[env_ids, 0, 0] = torch.min(pos[:, 0, 2], self.target_pos[env_ids, 0, 2])
        self.height_range[env_ids, 0, 1] = torch.max(pos[:, 0, 2], self.target_pos[env_ids, 0, 2])

        self.stats[env_ids] = 0.
        # pit #5: reset hierarchical controller PID state for terminated envs
        if self.use_topo:
            self.hierarchical_controller.reset_partial(env_ids)
        
    def _pre_sim_step(self, tensordict: TensorDictBase):
        if self.use_topo:
            self._pre_sim_step_graph(tensordict)
        else:
            actions = tensordict[("agents", "action")]
            self.drone.apply_action(actions)

    def _pre_sim_step_graph(self, tensordict: TensorDictBase):
        """Graph-PPO control chain: discrete action → velocity → QP → hierarchical PID → drone.

        Calling order (confirmed from IsaacEnv source):
          1. TransformedEnv calls VelController._inv_call(tensordict)
             → reads ("agents","action") as velocity (B,3), converts to motor thrusts (B,1,4),
               writes motor thrusts back into tensordict["agents","action"].
          2. IsaacEnv._step calls _pre_sim_step(tensordict)
             → at this point tensordict["agents","action"] is already MOTOR THRUSTS.

        Therefore we CANNOT read the policy velocity from tensordict["agents","action"] here.
        PPO._forward_graph saves the raw velocity in tensordict["_v_rl"] (B,3) for us.

        After QP + hierarchical PID we obtain ctrl_vel (B,3).  We must convert it to motor
        thrusts ourselves using the injected lee_controller, then call drone.apply_action.
        """
        B      = self.num_envs
        device = self.device

        # -- World velocity from policy (saved before VelController overwrites action) --
        v_rl = tensordict["_v_rl"]  # (B, 3)

        # -- Build obstacle tensor for QP: [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, radius] --
        if hasattr(self, 'dyn_obs_state') and self.cfg.env_dyn.num_obstacles > 0:
            obs_pos = self.dyn_obs_state[:, :3].unsqueeze(0).expand(B, -1, -1)  # (B, N_obs, 3)
            obs_vel = self.dyn_obs_vel.unsqueeze(0).expand(B, -1, -1)            # (B, N_obs, 3)
            obs_r   = (self.dyn_obs_size[:, 0] / 2.).unsqueeze(0).expand(B, -1).unsqueeze(-1)  # (B, N_obs, 1)
            obstacles = torch.cat([obs_pos, obs_vel, obs_r], dim=-1)             # (B, N_obs, 7)
        else:
            obstacles = torch.zeros(B, 0, 7, device=device)

        ego_pos_3d = self.drone.pos.squeeze(1)   # (B, 3)
        ego_vel_3d = self.root_state[..., 7:10].squeeze(1) if hasattr(self, 'root_state') \
                     else torch.zeros(B, 3, device=device)

        # -- SafetyShieldQP --
        v_safe, intervention = self.safety_shield.solve(v_rl, obstacles, ego_pos_3d)

        # Cache for reward shaping (read in _compute_state_and_obs)
        self._last_intervention  = intervention   # (B,)

        # -- HierarchicalController --
        is_hl = self.hierarchical_controller.is_high_level_step()
        ctrl_vel, ctrl_info = self.hierarchical_controller.step(
            ego_pos=ego_pos_3d,
            ego_vel=ego_vel_3d,
            high_level_velocity=v_safe if is_hl else None,
        )
        self._last_tracking_error = ctrl_info["tracking_error"]  # (B,)

        # -- Convert ctrl_vel → motor thrust via LeePositionController --
        # VelController already ran before _pre_sim_step; we must replicate what it did.
        # drone_state: (B, 1, 13); ctrl_vel: (B, 3) → unsqueeze to (B, 1, 3) for Lee.
        drone_state_13 = self.info["drone_state"]  # (B, 1, 13)
        motor_cmds = self.lee_controller(
            drone_state_13,
            target_vel=ctrl_vel.unsqueeze(1),  # (B, 1, 3)
            target_yaw=None,
        )  # → (B, 1, num_rotors)
        torch.nan_to_num_(motor_cmds, 0.)
        self.drone.apply_action(motor_cmds)


    def _post_sim_step(self, tensordict: TensorDictBase):
        if (self.cfg.env_dyn.num_obstacles != 0):
            self.move_dynamic_obstacle()
        self.lidar.update(self.dt)
    
    # get current states/observation
    def _compute_state_and_obs(self):
        self.root_state = self.drone.get_state(env_frame=False) # (world_pos, orientation (quat), world_vel_and_angular, heading, up, 4motorsthrust)
        self.info["drone_state"][:] = self.root_state[..., :13] # info is for controller

        # >>>>>>>>>>>>The relevant code starts from here<<<<<<<<<<<<
        # -----------Network Input I: LiDAR range data--------------
        self.lidar_scan = self.lidar_range - (
            (self.lidar.data.ray_hits_w - self.lidar.data.pos_w.unsqueeze(1))
            .norm(dim=-1)
            .clamp_max(self.lidar_range)
            .reshape(self.num_envs, 1, *self.lidar_resolution)
        ) # lidar scan store the data that is range - distance and it is in lidar's local frame

        # Optional render for LiDAR
        if self._should_render(0):
            self.debug_draw.clear()
            x = self.lidar.data.pos_w[0]
            # set_camera_view(
            #     eye=x.cpu() + torch.as_tensor(self.cfg.viewer.eye),
            #     target=x.cpu() + torch.as_tensor(self.cfg.viewer.lookat)                        
            # )
            v = (self.lidar.data.ray_hits_w[0] - x).reshape(*self.lidar_resolution, 3)
            # self.debug_draw.vector(x.expand_as(v[:, 0]), v[:, 0])
            # self.debug_draw.vector(x.expand_as(v[:, -1]), v[:, -1])
            self.debug_draw.vector(x.expand_as(v[:, 0])[0], v[0, 0])

        # ---------Network Input II: Drone's internal states---------
        # a. distance info in horizontal and vertical plane
        rpos = self.target_pos - self.root_state[..., :3]        
        distance = rpos.norm(dim=-1, keepdim=True) # start to goal distance
        distance_2d = rpos[..., :2].norm(dim=-1, keepdim=True)
        distance_z = rpos[..., 2].unsqueeze(-1)
        
        
        # b. unit direction vector to goal
        target_dir_2d = self.target_dir.clone()
        target_dir_2d[..., 2] = 0

        rpos_clipped = rpos / distance.clamp(1e-6) # unit vector: start to goal direction
        rpos_clipped_g = vec_to_new_frame(rpos_clipped, target_dir_2d) # express in the goal coodinate
        
        # c. velocity in the goal frame
        vel_w = self.root_state[..., 7:10] # world vel
        vel_g = vec_to_new_frame(vel_w, target_dir_2d)   # coordinate change for velocity

        # final drone's internal states
        drone_state = torch.cat([rpos_clipped_g, distance_2d, distance_z, vel_g], dim=-1).squeeze(1)

        if (self.cfg.env_dyn.num_obstacles != 0):
            # ---------Network Input III: Dynamic obstacle states--------
            # ------------------------------------------------------------
            # a. Closest N obstacles relative position in the goal frame 
            # Find the N closest and within range obstacles for each drone
            dyn_obs_pos_expanded = self.dyn_obs_state[..., :3].unsqueeze(0).repeat(self.num_envs, 1, 1)
            dyn_obs_rpos_expanded = dyn_obs_pos_expanded[..., :3] - self.root_state[..., :3] 
            dyn_obs_rpos_expanded[:, int(self.dyn_obs_state.size(0)/2):, 2] = 0.
            dyn_obs_distance_2d = torch.norm(dyn_obs_rpos_expanded[..., :2], dim=2)  # Shape: (1000, 40). calculate 2d distance to each obstacle for all drones
            _, closest_dyn_obs_idx = torch.topk(dyn_obs_distance_2d, self.cfg.algo.feature_extractor.dyn_obs_num, dim=1, largest=False) # pick top N closest obstacle index
            dyn_obs_range_mask = dyn_obs_distance_2d.gather(1, closest_dyn_obs_idx) > self.lidar_range

            # relative distance of obstacles in the goal frame
            closest_dyn_obs_rpos = torch.gather(dyn_obs_rpos_expanded, 1, closest_dyn_obs_idx.unsqueeze(-1).expand(-1, -1, 3))
            closest_dyn_obs_rpos_g = vec_to_new_frame(closest_dyn_obs_rpos, target_dir_2d) 
            closest_dyn_obs_rpos_g[dyn_obs_range_mask] = 0. # exclude out of range obstacles
            closest_dyn_obs_distance = closest_dyn_obs_rpos.norm(dim=-1, keepdim=True)
            closest_dyn_obs_distance_2d = closest_dyn_obs_rpos_g[..., :2].norm(dim=-1, keepdim=True)
            closest_dyn_obs_distance_z = closest_dyn_obs_rpos_g[..., 2].unsqueeze(-1)
            closest_dyn_obs_rpos_gn = closest_dyn_obs_rpos_g / closest_dyn_obs_distance.clamp(1e-6)

            # b. Velocity in the goal frame for the dynamic obstacles
            closest_dyn_obs_vel = self.dyn_obs_vel[closest_dyn_obs_idx]
            closest_dyn_obs_vel[dyn_obs_range_mask] = 0.
            closest_dyn_obs_vel_g = vec_to_new_frame(closest_dyn_obs_vel, target_dir_2d) 

            # c. Size of dynamic obstacles in category
            closest_dyn_obs_size = self.dyn_obs_size[closest_dyn_obs_idx] # the acutal size

            closest_dyn_obs_width = closest_dyn_obs_size[..., 0].unsqueeze(-1)
            closest_dyn_obs_width_category = closest_dyn_obs_width / self.dyn_obs_width_res - 1. # convert to category: [0, 1, 2, 3]
            closest_dyn_obs_width_category[dyn_obs_range_mask] = 0.

            closest_dyn_obs_height = closest_dyn_obs_size[..., 2].unsqueeze(-1)
            closest_dyn_obs_height_category = torch.where(closest_dyn_obs_height > self.max_obs_3d_height, torch.tensor(0.0), closest_dyn_obs_height)
            closest_dyn_obs_height_category[dyn_obs_range_mask] = 0.

            # concatenate all for dynamic obstacles
            # dyn_obs_states = torch.cat([closest_dyn_obs_rpos_g, closest_dyn_obs_vel_g, closest_dyn_obs_width_category, closest_dyn_obs_height_category], dim=-1).unsqueeze(1)
            dyn_obs_states = torch.cat([closest_dyn_obs_rpos_gn, closest_dyn_obs_distance_2d, closest_dyn_obs_distance_z, closest_dyn_obs_vel_g, closest_dyn_obs_width_category, closest_dyn_obs_height_category], dim=-1).unsqueeze(1)

            # check dynamic obstacle collision for later reward
            closest_dyn_obs_distance_2d_collsion = closest_dyn_obs_rpos[..., :2].norm(dim=-1, keepdim=True)
            closest_dyn_obs_distance_2d_collsion[dyn_obs_range_mask] = float('inf')
            closest_dyn_obs_distance_zn_collision = closest_dyn_obs_rpos[..., 2].unsqueeze(-1).norm(dim=-1, keepdim=True)
            closest_dyn_obs_distance_zn_collision[dyn_obs_range_mask] = float('inf')
            dynamic_collision_2d = closest_dyn_obs_distance_2d_collsion <= (closest_dyn_obs_width/2. + 0.3)
            dynamic_collision_z = closest_dyn_obs_distance_zn_collision <= (closest_dyn_obs_height/2. + 0.3)
            dynamic_collision_each = dynamic_collision_2d & dynamic_collision_z
            dynamic_collision = torch.any(dynamic_collision_each, dim=1)

            # distance to dynamic obstacle for reward calculation (not 100% correct in math but should be good enough for approximation)
            closest_dyn_obs_distance_reward = closest_dyn_obs_rpos.norm(dim=-1) - closest_dyn_obs_size[..., 0]/2. # for those 2D obstacle, z distance will not be considered
            closest_dyn_obs_distance_reward[dyn_obs_range_mask] = self.cfg.sensor.lidar_range
            
        else:
            dyn_obs_states = torch.zeros(self.num_envs, 1, self.cfg.algo.feature_extractor.dyn_obs_num, 10, device=self.cfg.device)
            dynamic_collision = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.cfg.device)
            
        # -----------------Network Input Final--------------
        obs = {
            "state": drone_state,
            "lidar": self.lidar_scan,
            "direction": target_dir_2d,
            "dynamic_obstacle": dyn_obs_states
        }

        # ----- Graph topology observation (when use_topo=True) -----
        if self.use_topo:
            # dyn_obs_states: (B, 1, N_dyn, 10) → squeeze to (B, N_dyn, 10)  [pit #9]
            dyn_obs_for_topo = dyn_obs_states.squeeze(1)  # (B, N_dyn, 10)
            # LiDAR world hit positions and sensor position
            ray_hits_w = self.lidar.data.ray_hits_w  # (B, N_rays, 3)
            ray_pos_w  = self.lidar.data.pos_w       # (B, 3)
            ego_vel_3d  = vel_w.squeeze(1)           # (B, 3)
            tgt_pos_3d  = self.target_pos.squeeze(1) # (B, 3)
            topo_out = self.topo_extractor.extract_topology(
                ray_hits_w, ray_pos_w, dyn_obs_for_topo, ego_vel_3d, tgt_pos_3d
            )
            # Cache node positions for _pre_sim_step
            self._last_topo_node_positions = topo_out["node_positions"]  # (B, N+1, 3)
            # Add to obs dict (masks as float for TorchRL spec compatibility)
            obs["node_features"]  = topo_out["node_features"]
            obs["node_positions"] = topo_out["node_positions"]
            obs["edge_features"]  = topo_out["edge_features"]
            obs["node_mask"]      = topo_out["node_mask"].float()
            obs["edge_mask"]      = topo_out["edge_mask"].float()


        # -----------------Reward Calculation-----------------
        # All signals are sourced from hardware-available sensors:
        #   velocity        → onboard IMU
        #   direction vec   → GPS position + known target position
        #   LiDAR scan      → onboard lidar
        #   height          → barometric altimeter
        #   collision       → lidar min-clearance
        # NO privileged simulation state (global distance delta, inverse potential, etc.)

        # a. LiDAR safety signals (for stats logging; also consumed by RewardShaper).
        reward_safety_static = torch.log(
            (self.lidar_range - self.lidar_scan).clamp(min=1e-6, max=self.lidar_range)
        ).mean(dim=(2, 3))  # (B, 1)

        if self.cfg.env_dyn.num_obstacles != 0:
            reward_safety_dynamic = torch.log(
                closest_dyn_obs_distance_reward.clamp(min=1e-6, max=self.lidar_range)
            ).mean(dim=-1, keepdim=True)
        else:
            reward_safety_dynamic = torch.zeros(self.num_envs, 1, device=self.cfg.device)

        # b. Smoothness penalty — jerk regularisation (stats + RewardShaper).
        penalty_smooth = (self.drone.vel_w[..., :3] - self.prev_drone_vel_w).norm(dim=-1)  # (B, 1)

        # c. Height penalty — quadratic when outside allowed band (stats + RewardShaper).
        penalty_height = torch.zeros(self.num_envs, 1, device=self.cfg.device)
        penalty_height[self.drone.pos[..., 2] > (self.height_range[..., 1] + 0.2)] = (
            (self.drone.pos[..., 2] - self.height_range[..., 1] - 0.2) ** 2
        )[self.drone.pos[..., 2] > (self.height_range[..., 1] + 0.2)]
        penalty_height[self.drone.pos[..., 2] < (self.height_range[..., 0] - 0.2)] = (
            (self.height_range[..., 0] - 0.2 - self.drone.pos[..., 2]) ** 2
        )[self.drone.pos[..., 2] < (self.height_range[..., 0] - 0.2)]

        # d. Collision detection (lidar-based, hardware-available).
        static_collision = einops.reduce(self.lidar_scan, "n 1 w h -> n 1", "max") > (self.lidar_range - 0.3)
        collision = static_collision | dynamic_collision

        # e. Reach-goal detection (GPS-based distance, available on real robot).
        distance_for_reward = distance.squeeze(1)          # (B, 1)
        reach_goal = (distance_for_reward < 1.0)           # (B, 1) bool

        # f. Horizontal out-of-bounds.
        out_of_bounds_xy = (
            (self.drone.pos[..., 0].abs() > self.terrain_half_extent_x)
            | (self.drone.pos[..., 1].abs() > self.terrain_half_extent_y)
        )  # (B, 1) bool

        # ==== Unified privilege-free reward (all modes: ppo, graph_ppo, curriculum) ====
        _ctx = {
            "vel_w":            self.drone.vel_w,
            "prev_vel_w":       self.prev_drone_vel_w,
            "target_dir_2d":    target_dir_2d,
            "lidar_scan":       self.lidar_scan,
            "lidar_range":      self.lidar_range,
            "drone_pos":        self.drone.pos,
            "height_range":     self.height_range,
            "collision":        collision,
            "reach_goal":       reach_goal,
            "out_of_bounds_xy": out_of_bounds_xy,
            "distance":         distance,
        }
        if self.cfg.env_dyn.num_obstacles != 0:
            _ctx["closest_dyn_obs_distance_reward"] = closest_dyn_obs_distance_reward
        self.reward = self.reward_shaper.compute(_ctx)

        # Graph-PPO: QP intervention penalty + tracking error penalty (additive on top).
        # Scale lifted to match new reward magnitudes.
        if self.use_topo and hasattr(self, '_last_intervention') and self._last_intervention is not None:
            I_t     = self._last_intervention                                      # (B,)
            # Smooth penalty: exp-Gaussian maps intervention magnitude to [−1, 0] per unit weight
            intv_w  = float(self.cfg.topo.reward_intervention_weight)               # config: -0.5
            r_intv  = intv_w * (1.0 - torch.exp(-I_t ** 2 / (2 * 0.25)))          # ∈ [intv_w, 0]
            track_w = float(self.cfg.topo.reward_tracking_weight)
            self.reward = self.reward \
                + r_intv.unsqueeze(-1) \
                + track_w * self._last_tracking_error.unsqueeze(-1)

        # Terminate Conditions
        # reach_goal → terminated so: (a) episode resets giving diverse new start points,
        #   (b) reach_goal_bonus is truly one-time, (c) value estimates stay bounded.
        below_bound = self.drone.pos[..., 2] < self.z_terminate_min
        above_bound = self.drone.pos[..., 2] > self.z_terminate_max
        # out_of_bounds_xy already computed above (before reward sum)
        self.terminated = below_bound | above_bound | collision | reach_goal | out_of_bounds_xy
        # progress_buf is incremented by isaac_env._step() BEFORE _compute_reward_and_done() is
        # called, so at step N the counter already reads N.  The condition >= max_episode_length
        # correctly fires on the last step (step 1000 when max_episode_length=1000).
        # truncated = time-limit only (not collision/goal); terminated = natural end-of-episode.
        self.truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(1) & ~self.terminated

        # update previous velocity for smoothness calculation in the next ieteration
        self.prev_drone_vel_w = self.drone.vel_w[..., :3].clone()

        # # -----------------Training Stats-----------------
        self.stats["return"] += self.reward
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)
        self.stats["reach_goal"] = reach_goal.float()
        self.stats["collision"] = collision.float()
        self.stats["truncated"] = self.truncated.float()
        self.stats["distance_to_goal"] = distance_for_reward
        self.stats["speed"] = self.drone.vel_w[..., :3].norm(dim=-1, keepdim=True)
        # Privilege-free reward component stats (static methods, no extra alloc overhead)
        self.stats["reward_progress"] = self._RS._progress_local(
            self.drone.vel_w, target_dir_2d) * 10.0          # vel · dir_hat × 10
        self.stats["reward_heading"]  = self._RS._heading_reward(
            self.drone.vel_w, target_dir_2d) * 5.0           # cos(vel, dir) × 5
        self.stats["reward_speed"]    = self._RS._speed_band(
            self.drone.vel_w,
            self.reward_shaper.speed_target,
            self.reward_shaper.speed_sigma) * 3.0            # speed-band Gaussian × 3
        self.stats["reward_safety_static"]  = reward_safety_static
        self.stats["reward_safety_dynamic"] = reward_safety_dynamic
        self.stats["penalty_smooth"] = penalty_smooth
        self.stats["penalty_height"] = penalty_height
        if self.use_topo and hasattr(self, '_last_intervention') and self._last_intervention is not None:
            self.stats["qp_intervention"] = self._last_intervention.unsqueeze(-1)
            self.stats["tracking_error"] = self._last_tracking_error.unsqueeze(-1)
        else:
            self.stats["qp_intervention"] = torch.zeros(self.num_envs, 1, device=self.cfg.device)
            self.stats["tracking_error"] = torch.zeros(self.num_envs, 1, device=self.cfg.device)

        return TensorDict({
            "agents": TensorDict(
                {
                    "observation": obs,
                }, 
                [self.num_envs]
            ),
            "stats": self.stats.clone(),
            "info": self.info
        }, self.batch_size)

    def _compute_reward_and_done(self):
        reward = self.reward
        terminated = self.terminated
        truncated = self.truncated
        return TensorDict(
            {
                "agents": {
                    "reward": reward
                },
                "done": terminated | truncated,
                "terminated": terminated,
                "truncated": truncated,
            },
            self.batch_size,
        )
