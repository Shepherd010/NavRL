import argparse
import os
import hydra
import datetime
import wandb
import torch
from omegaconf import DictConfig, OmegaConf
from omni.isaac.kit import SimulationApp
from ppo import PPO
from omni_drones.controllers import LeePositionController
from omni_drones.utils.torchrl.transforms import VelController, ravel_composite
from omni_drones.utils.torchrl import SyncDataCollector, EpisodeStats
from torchrl.envs.transforms import TransformedEnv, Compose
from utils import evaluate
from torchrl.envs.utils import ExplorationType
import viz as viz_lib  # visualization helpers (V1-V4)




FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cfg")
@hydra.main(config_path=FILE_PATH, config_name="train", version_base=None)
def main(cfg):
    # Simulation App
    sim_app = SimulationApp({"headless": cfg.headless, "anti_aliasing": 0})

    # Use Wandb to monitor training
    run_id = cfg.wandb.get("run_id")
    run = wandb.init(
        project=cfg.wandb.project,
        name=f"{cfg.wandb.name}/{datetime.datetime.now().strftime('%m-%d_%H-%M')}",
        entity=cfg.wandb.entity,
        config=OmegaConf.to_container(cfg, resolve=True),
        mode=cfg.wandb.mode,
        id=run_id if run_id is not None else wandb.util.generate_id(),
        resume="must" if run_id is not None else None,
    )

    # Navigation Training Environment
    from env import NavigationEnv
    env = NavigationEnv(cfg)

    # Transformed Environment
    transforms = []
    # transforms.append(ravel_composite(env.observation_spec, ("agents", "intrinsics"), start_dim=-1))
    controller = LeePositionController(9.81, env.drone.params).to(cfg.device)
    vel_transform = VelController(controller, yaw_control=False)
    transforms.append(vel_transform)
    transformed_env = TransformedEnv(env, Compose(*transforms)).train()
    transformed_env.set_seed(cfg.seed)

    # Inject the same LeePositionController into env so that _pre_sim_step_graph can
    # convert ctrl_vel → motor thrusts directly (VelController runs before _pre_sim_step,
    # so we cannot rely on it inside _pre_sim_step).
    if getattr(env, 'use_topo', False) or (hasattr(env, 'base_env') and getattr(env.base_env, 'use_topo', False)):
        # torchrl TransformedEnv wraps the real env, we need to access base_env
        base_env = env.base_env if hasattr(env, 'base_env') else env
        base_env.lee_controller = controller

    # PPO Policy (pass topo_cfg when graph_ppo mode is active)
    _topo_cfg = OmegaConf.select(cfg, 'topo', default=None) if OmegaConf.select(cfg, 'mode', default='ppo') == 'graph_ppo' else None
    policy = PPO(cfg.algo, transformed_env.observation_spec, transformed_env.action_spec, cfg.device,
                 topo_cfg=_topo_cfg)

    # checkpoint = "/home/zhefan/catkin_ws/src/navigation_runner/scripts/ckpts/checkpoint_2500.pt"
    # checkpoint = "/home/xinmingh/RLDrones/navigation/scripts/nav-ros/navigation_runner/ckpts/checkpoint_36000.pt"
    # policy.load_state_dict(torch.load(checkpoint))
    
    # Episode Stats Collector
    episode_stats_keys = [
        k for k in transformed_env.observation_spec.keys(True, True) 
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    episode_stats = EpisodeStats(episode_stats_keys)

    # RL Data Collector
    collector = SyncDataCollector(
        transformed_env,
        policy=policy, 
        frames_per_batch=cfg.env.num_envs * cfg.algo.training_frame_num, 
        total_frames=cfg.max_frame_num,
        device=cfg.device,
        return_same_td=True, # update the return tensordict inplace (should set to false if we need to use replace buffer)
        exploration_type=ExplorationType.RANDOM, # sample from normal distribution
    )

    # Training Loop
    log_interval = int(OmegaConf.select(cfg, 'log_interval', default=10))  # log every N steps
    viz_interval = int(OmegaConf.select(cfg, 'viz.interval', default=50))  # viz every N steps
    _traj_segs: list = []   # rolling buffer of (T,3) trajectory segments for V4
    # Rolling history buffers for learning curves (Panel C in nav_dashboard)
    _value_history:    list = []   # mean V(s) per viz step
    _distance_history: list = []   # mean dist-to-goal per viz step
    for i, data in enumerate(collector):
        # Log Info
        info = {"env_frames": collector._frames, "rollout_fps": collector._fps}

        # Train Policy
        train_loss_stats = policy.train(data)
        # NOTE: torch.cuda.empty_cache() removed — it forces all fragmented blocks back to
        # CUDA, destroying PyTorch's allocator pool locality, which causes re-allocation on
        # the next step and is measurably slower. Only call it before eval if needed.
        info.update(train_loss_stats) # log training loss info

        # Log real-time training status from the latest frame in this batch.
        # This complements EpisodeStats (which only updates on finished episodes)
        # and mirrors eval/stats.* so we can diagnose hovering early.
        if ("next", "stats") in data.keys(True, True):
            live_stats = {
                "train_live/" + (".".join(k) if isinstance(k, tuple) else k): v[:, -1].float().mean().item()
                for k, v in data[("next", "stats")].items(True, True)
            }
            info.update(live_stats)

        # Calculate and log training episode stats
        episode_stats.add(data)
        if len(episode_stats) >= transformed_env.num_envs: # evaluate once if all agents finished one episode
            stats = {
                "train/" + (".".join(k) if isinstance(k, tuple) else k): torch.mean(v.float()).item() 
                for k, v in episode_stats.pop().items(True, True)
            }
            info.update(stats)

        # Evaluate policy and log info
        if i > 0 and i % cfg.eval_interval == 0:
            print("[NavRL]: start evaluating policy at training step: ", i)
            torch.cuda.empty_cache()  # safe to call here: before eval, VRAM matters
            eval_info = evaluate(
                env=transformed_env, 
                policy=policy,
                seed=cfg.seed, 
                cfg=cfg,
                exploration_type=ExplorationType.MEAN,
                eval_num_envs=cfg.eval.num_envs,
                eval_max_steps=cfg.eval.max_episode_length,
            )
            env.train()
            env.reset()
            info.update(eval_info)
            print("\n[NavRL]: evaluation done.")
        
        # Update wandb — throttled to every log_interval steps to reduce serialisation overhead
        if i % log_interval == 0:
            run.log(info, step=i)

        # -----------------------------------------------------------------
        # Visualization block (V1 topo graph, V2 attention, V3 reward bars)
        # Runs every viz_interval steps; uses env 0 from the last rollout frame
        # -----------------------------------------------------------------
        if policy.graph_ppo and i > 0 and i % viz_interval == 0:
            try:
                viz_td   = data[:, -1]   # last frame of rollout, all envs
                viz_data = policy.get_viz_data(viz_td)

                # ── Build reward-bar dict (V3 / Panel B of nav_dashboard) ─────────
                rbar = None
                if ("next", "stats") in data.keys(True, True):
                    raw = data["next", "stats"]
                    _reward_key_map = {
                        "reward_vel":       ("vel",          +1.0),
                        "reward_progress":  ("progress",     +1.0),
                        "penalty_smooth":   ("smooth (−)",   -1.0),
                        "penalty_height":   ("height (−)",   -1.0),
                        "collision":        ("collision (−)", -10.0),
                        "qp_intervention":  ("QP interv.",   -1.0),
                    }
                    _tmp = {}
                    for rk, (display, sign) in _reward_key_map.items():
                        if rk in raw.keys():
                            _tmp[display] = sign * float(raw[rk][:, -1].float().mean().item())
                    if _tmp:
                        rbar = _tmp

                # ── Accumulate value & distance history (Panel C of nav_dashboard) ─
                _val_scalar  = float(viz_data["value"].mean().item()) \
                               if viz_data.get("value") is not None else None
                _dist_scalar = None
                if ("next", "stats") in data.keys(True, True):
                    raw = data["next", "stats"]
                    if "distance_to_goal" in raw.keys():
                        _dist_scalar = float(raw["distance_to_goal"][:, -1].float().mean().item())
                if _val_scalar is not None:
                    _value_history.append(_val_scalar)
                if _dist_scalar is not None:
                    _distance_history.append(_dist_scalar)
                # Keep last 100 data points to avoid ever-growing history
                if len(_value_history)    > 100: _value_history    = _value_history[-100:]
                if len(_distance_history) > 100: _distance_history = _distance_history[-100:]

                # ── V4 trajectory segments ───────────────────────────────────────
                np_key = ("agents", "observation", "node_positions")
                if np_key in data.keys(True, True):
                    ego_pos = data[np_key][0, :, 0, :].cpu()  # (T, 3)
                    _traj_segs.append(ego_pos)
                    if len(_traj_segs) > 16:
                        _traj_segs = _traj_segs[-16:]

                # ── Composite figure 1: Navigation Dashboard (V1 + V3 + curves) ──
                nav_fig = viz_lib.plot_nav_dashboard(
                    node_positions=viz_data["node_positions"],
                    node_mask=viz_data["node_mask"],
                    probs=viz_data["probs"],
                    selected_idx=viz_data["selected_idx"],
                    edge_mask=viz_data["edge_mask"],
                    reward_components=rbar,
                    value_history=_value_history    if len(_value_history)    > 1 else None,
                    distance_history=_distance_history if len(_distance_history) > 1 else None,
                    step=i,
                )

                # ── Composite figure 2: Training Internals (V2 + V4) ────────────
                train_fig = viz_lib.plot_training_status(
                    all_attn=viz_data["all_attn"],
                    node_mask=viz_data["node_mask"],
                    trajectories=_traj_segs if _traj_segs else None,
                    step=i,
                )

                run.log({
                    "viz/nav_dashboard":    wandb.Image(nav_fig),
                    "viz/training_status":  wandb.Image(train_fig),
                }, step=i)
            except Exception as _viz_err:
                # Never let viz crash training
                print(f"[NavRL viz] warning: {_viz_err}")

        # Save Model
        if i % cfg.save_interval == 0:
            ckpt_path = os.path.join(run.dir, f"checkpoint_{i}.pt")
            torch.save(policy.state_dict(), ckpt_path)
            print("[NavRL]: model saved at training step: ", i)

    ckpt_path = os.path.join(run.dir, "checkpoint_final.pt")
    torch.save(policy.state_dict(), ckpt_path)
    wandb.finish()
    sim_app.close()

if __name__ == "__main__":
    main()
    