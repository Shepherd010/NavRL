# NavRL 新架构实施指南（Code-LLM 操作手册）

> **这不是架构文档。** 架构、算法、数学推导完整写在 `doc/06` ~ `doc/11` 中，你的首要任务是阅读并严格按照那些文档实现代码，本文件只告诉你**以什么顺序读、哪里动哪个函数、有哪些硬约束、哪些地方有坑**。

---

## 0. 阅读路径（先读文档再写代码）

| 顺序 | 文档 | 必读章节 | 对应实现目标 |
|------|------|----------|------------|
| 1 | [06-新架构总览](./doc/06-新架构总览-拓扑图导航系统.md) | §1、§2、§5 数据流接口、§10 一致性检查表 | 理解全局数据流与所有模块边界 |
| 2 | [11-代码改造实施方案](./doc/11-代码改造实施方案.md) | §1 文件变更清单、§2~§5 各 Phase 步骤 | **核心执行计划，全部修改都在这里** |
| 3 | [07-拓扑提取模块设计](./doc/07-拓扑提取模块设计.md) | §2 类结构、§3 接口、§4 特征编码 | Phase 2：`topo_extractor.py` |
| 4 | [08-图Transformer策略网络](./doc/08-图Transformer策略网络详细设计.md) | §1.2 输入输出规格、§3~§5 实现、§6 PPO 集成 | Phase 2：`graph_transformer.py` + `ppo.py` |
| 5 | [09-安全盾QP优化器](./doc/09-安全盾QP优化器详细设计.md) | §3 实现、§6 鲁棒性、§8 干预奖励 | Phase 3：`safety_shield.py` |
| 6 | [10-分层控制架构](./doc/10-分层控制架构设计.md) | §3 实现、§4 频率调度器、§5 状态同步、§8 时序分析 | Phase 3：`hierarchical_control.py` |

---

## 1. 工程约束

### 1.1 禁止事项

- **禁止删除或重写现有逻辑** —— `env.py`、`ppo.py`、`train.py` 只做增量添加，原 `mode=ppo` 路径始终可用
- **禁止硬编码数值** —— 维度、范围、增益全部从 `cfg/topo.yaml` 读取（见 [doc/11 §2.2.1](./doc/11-代码改造实施方案.md#221-创建拓扑配置) 完整配置 schema）
- **禁止用 Python `for` 遍历批量节点/边** —— 必须向量化（torch 广播或 `torch_geometric`）
- **禁止跨 Phase 跳跃** —— 每个 Phase 验收通过后再进入下一个

### 1.2 必须遵守

- 所有新增函数必须有类型注解 + Google Style docstring（包含 `Args`/`Returns`/数学公式文档引用）
- 使用 `logging.getLogger(__name__)` 而非 `print()`
- 文件路径使用 `Path(__file__).parent` 而非绝对路径
- 每个 Phase 完成后运行 [doc/11 §6](./doc/11-代码改造实施方案.md#6-测试与验证) 对应测试，不通过不提交

---

## 2. 文件级修改清单

在动手之前，先阅读 [doc/11 §1.3 文件变更清单](./doc/11-代码改造实施方案.md#13-文件变更清单)，了解哪些文件需改、复杂度评级。下面是**针对现有代码的精确对接点**：

### 2.1 `training/scripts/env.py`

读取当前代码后，以下四处需要改动（不要改其他地方）：

| 方法 | 改动类型 | 触发条件 | 参考文档 |
|------|---------|---------|---------|
| `__init__` | 在 `super().__init__()` 之后追加 | 无条件（但用 `cfg.topo.use_topo` 守卫） | [doc/11 §4.1.1](./doc/11-代码改造实施方案.md#411-初始化部分) |
| `_setup_scene` / `observation_spec` | 追加 5 个 graph 键到 `CompositeSpec` | 无条件 | [doc/11 §4.1.2](./doc/11-代码改造实施方案.md#412-观测空间修改) |
| `_compute_state_and_obs` | 在 `obs = {...}` 字典构造前追加拓扑提取 | `if hasattr(self, 'topo_extractor')` | [doc/11 §4.1.4](./doc/11-代码改造实施方案.md#414-_compute_state_and_obs修改) |
| `_pre_sim_step` | 替换动作执行段（节点选择→速度→QP→飞控） | `if cfg.mode == 'graph_ppo'` | [doc/11 §4.1.5](./doc/11-代码改造实施方案.md#415-_pre_sim_step修改) |

**现有 `_compute_state_and_obs` 关键行（当前代码第 435~545 行左右）**：

```python
# 当前代码：LiDAR 范围被转为 lidar_scan
self.lidar_scan = self.lidar_range - (
    (self.lidar.data.ray_hits_w - self.lidar.data.pos_w.unsqueeze(1))
    .norm(dim=-1)
    ...
)
```

你需要在这段代码**之后**，在 `obs = {"state": ..., "lidar": ..., ...}` 字典**构造之前**插入拓扑提取：

```python
# ↓ 在此处插入（lidar_scan 计算完成后，obs 字典构造前）
if hasattr(self, 'topo_extractor'):
    # 注意：传入 ray_hits_w 世界坐标，不是 lidar_scan！
    # ray_hits_w shape: (num_envs, lidar_hbeams * lidar_vbeams, 3)
    topo_dict = self.topo_extractor.extract_topology(
        ray_hits_w=self.lidar.data.ray_hits_w,          # (B, 144, 3) 世界坐标
        ray_pos_w=self.lidar.data.pos_w,                 # (B, 3)
        dyn_obs=dyn_obs_states,                          # 复用上面已计算
        drone_state=self.root_state[..., :8],            # (B, 8)
        target_pos=self.target_pos.squeeze(1),           # (B, 3)
    )
    # 缓存以便 _pre_sim_step 使用（避免重复计算）
    self._cached_topo_dict = topo_dict
```

然后将 5 个键追加到 `obs` 字典（与 `observation_spec` 中的声明保持完全一致）：

```python
obs = {
    "state": drone_state,
    "lidar": self.lidar_scan,
    "direction": target_dir_2d,
    "dynamic_obstacle": dyn_obs_states,
    # ↓ 新增（仅 graph_ppo 模式）
    **(topo_dict if hasattr(self, '_cached_topo_dict') else {}),
}
```

**`observation_spec` 中对应的新增字段**（见 [doc/11 §4.1.2](./doc/11-代码改造实施方案.md#412-观测空间修改)）：

```python
# 在现有 CompositeSpec 中追加——N = cfg.topo.max_nodes + 1（含 ego，index 0）
"graph_node_features":  UnboundedContinuousTensorSpec((1, N, 18),        device=self.device),
"graph_node_positions": UnboundedContinuousTensorSpec((1, N, 3),         device=self.device),
"graph_edge_features":  UnboundedContinuousTensorSpec((1, N, N, 7),      device=self.device),
"graph_node_mask":      DiscreteTensorSpec(2,          (1, N),    dtype=torch.bool, device=self.device),
"graph_edge_mask":      DiscreteTensorSpec(2,          (1, N, N), dtype=torch.bool, device=self.device),
```

### 2.2 `training/scripts/ppo.py`

当前 `PPO.__init__` 建立了固定的 `feature_extractor → BetaActor → Critic` 链路。需要在**不删除该链路**的前提下，添加 `graph_ppo` 分支：

```python
# 在 PPO.__init__ 末尾，dummy_input forward 之前添加条件分支
if cfg.mode == 'graph_ppo':
    from modules.graph_transformer import GraphTransformer
    self.graph_transformer = GraphTransformer(
        node_feat_dim=cfg.topo.node_feat_dim,    # 18
        edge_feat_dim=cfg.topo.edge_feat_dim,    # 7
        hidden_dim=cfg.topo.hidden_dim,          # 64
        num_heads=cfg.topo.num_heads,            # 4
        num_layers=cfg.topo.num_layers,          # 3
    ).to(self.device)
    
    # 离散 Actor：输出 logits (batch, max_nodes)
    self.graph_actor = TensorDictModule(
        nn.Linear(cfg.topo.hidden_dim, cfg.topo.max_nodes),
        ["_graph_h_ego"],          # ego 节点的隐状态
        ["graph_action_logits"],
    ).to(self.device)
    
    # Critic 可复用现有 self.critic，输入换成 _graph_h_ego
```

`__call__` 方法按 `cfg.mode` 分发：

```python
def __call__(self, tensordict):
    if self.cfg.mode == 'graph_ppo':
        return self._forward_graph(tensordict)
    else:
        # 原有链路，保持不变
        self.feature_extractor(tensordict)
        self.actor(tensordict)
        self.critic(tensordict)
        actions = (2 * tensordict["agents", "action_normalized"] * self.cfg.actor.action_limit) \
                  - self.cfg.actor.action_limit
        actions_world = vec_to_world(actions, tensordict["agents", "observation", "direction"])
        tensordict["agents", "action"] = actions_world
        return tensordict

def _forward_graph(self, tensordict):
    """Graph-PPO 前向传播"""
    obs = tensordict["agents", "observation"]
    
    # 1. GraphTransformer 推理
    graph_out = self.graph_transformer(
        node_features=obs["graph_node_features"].squeeze(1),   # (B, N, 18)
        edge_features=obs["graph_edge_features"].squeeze(1),   # (B, N, N, 7)
        node_mask=obs["graph_node_mask"].squeeze(1),            # (B, N)
        edge_mask=obs["graph_edge_mask"].squeeze(1),            # (B, N, N)
    )
    # graph_out["probs"]:  (B, N-1) —— 已排除 ego（index 0）
    # graph_out["h"]:      (B, N, hidden_dim) —— 所有节点的隐表示
    
    tensordict["_graph_h_ego"] = graph_out["h"][:, 0, :]      # ego 节点特征用于 Critic
    tensordict["graph_action_logits"] = torch.log(graph_out["probs"].clamp(min=1e-8))
    
    # 2. 离散采样（Categorical）
    dist = torch.distributions.Categorical(logits=tensordict["graph_action_logits"])
    action = dist.sample()                                      # (B,) ∈ [0, N-2]
    tensordict["agents", "action"] = action.unsqueeze(-1)      # (B, 1)
    tensordict["sample_log_prob"] = dist.log_prob(action)      # (B,)
    
    # 3. Critic
    self.critic(tensordict)   # 读 _graph_h_ego → state_value
    
    return tensordict
```

完整的训练 `_update` 逻辑需要改动处见 [doc/08 §6](./doc/08-图Transformer策略网络详细设计.md#6-与ppo框架集成)，核心是用 `Categorical.log_prob` 替换原来的 `IndependentBeta.log_prob`，**熵损失系数需要调低**（离散动作熵尺度不同）。

### 2.3 `training/scripts/train.py`

当前 `train.py` 的核心训练循环（约第 83 行）：

```python
for i, data in enumerate(collector):
    train_loss_stats = policy.train(data)
    ...
```

`SyncDataCollector` 是一个封装好的采样器，它内部调用 `policy(tensordict)` 收集数据。**不要改动 collector 本身**，只需要在 collector 初始化之前，如果 `cfg.mode == 'graph_ppo'`，则同时初始化 `HierarchicalController` 并将引用传给 `env`：

```python
# 在 policy = PPO(...) 之后、collector = SyncDataCollector(...) 之前
if cfg.mode == 'graph_ppo' and cfg.topo.use_hierarchical:
    from modules.hierarchical_control import HierarchicalController
    hierarchical_ctrl = HierarchicalController(
        high_freq=cfg.control_freq,          # 62.5
        low_freq=cfg.topo.planning_freq,     # 10.0
        pid_gains={'kp': cfg.topo.pid_kp, 'kd': cfg.topo.pid_kd, 'ki': cfg.topo.pid_ki},
    )
    hierarchical_ctrl.reset(cfg.env.num_envs, cfg.device)
    env.hierarchical_controller = hierarchical_ctrl   # 注入环境
```

`reset_partial()` 的调用时机：由于 `SyncDataCollector` 封装了 `env.step()`，你必须在 `env._reset_idx` 中调用 `hierarchical_controller.reset_partial(env_ids)` —— 这是唯一能感知到回合结束的地方。见 [doc/10 §5](./doc/10-分层控制架构设计.md#5-状态同步机制)，在 `_reset_idx` 末尾追加：

```python
def _reset_idx(self, env_ids: torch.Tensor):
    # ... 现有代码不变 ...

    # ↓ 追加（Phase 3 实施时加入）
    if hasattr(self, 'hierarchical_controller'):
        self.hierarchical_controller.reset_partial(env_ids)
```

奖励塑形：在 `_compute_reward_and_done` 中追加两项，权重来自 `cfg.topo`（见 [doc/09 §8](./doc/09-安全盾QP优化器详细设计.md#8-干预分析与奖励设计)）：

```python
# 在现有 reward 构建完成后追加（只在 graph_ppo 模式生效）
if self.cfg.mode == 'graph_ppo':
    intervention = tensordict.get("intervention", torch.zeros_like(reward))
    tracking_err = tensordict.get("tracking_error", torch.zeros_like(reward))
    reward = reward \
        + self.cfg.topo.reward_intervention_weight * intervention.unsqueeze(-1) \
        + self.cfg.topo.reward_tracking_weight    * tracking_err.unsqueeze(-1)
```

---

## 3. 新增模块的实现规范

四个新增模块的**完整算法实现在对应文档中已经给出**，此处只列出集成时必须满足的接口签名和约束，不重复算法。

### 3.1 `modules/topo_extractor.py`

**主接口**（[doc/07 §2.1](./doc/07-拓扑提取模块设计.md#21-总体架构)）：

```python
def extract_topology(
    self,
    ray_hits_w:   torch.Tensor,   # (B, lidar_hbeams * lidar_vbeams, 3)  世界坐标
    ray_pos_w:    torch.Tensor,   # (B, 3)  LiDAR 光心世界坐标
    dyn_obs:      torch.Tensor,   # (B, N_dyn, 10)  动态障碍物状态
    drone_state:  torch.Tensor,   # (B, 8)
    target_pos:   torch.Tensor,   # (B, 3)
) -> dict:
    """
    返回 dict，键名必须与 env.py observation_spec 声明完全一致：
        node_features:  (B, max_nodes+1, 18)   — ego 在 index 0
        node_positions: (B, max_nodes+1, 3)    — 世界坐标，ego 在 index 0
        edge_features:  (B, max_nodes+1, max_nodes+1, 7)
        node_mask:      (B, max_nodes+1)  bool
        edge_mask:      (B, max_nodes+1, max_nodes+1)  bool
    """
```

**实现约束**：
- 距离场用 `scipy.ndimage.distance_transform_edt`（2D 投影，详见 [doc/07 §3.1](./doc/07-拓扑提取模块设计.md#31-距离场计算)）
- 节点提取用距离场梯度局部极大，而非逐像素 for 循环（详见 [doc/07 §3.2](./doc/07-拓扑提取模块设计.md#32-节点提取)）
- 空图 fallback：当无有效节点时，用目标方向虚拟一个节点，`node_mask` 全为 True，避免 softmax NaN（因为 [坑 7](#坑-7attention-mask-使用--1e9-而非--inf) 只能减少 NaN 概率，不能消除空图情况）

### 3.2 `modules/graph_transformer.py`

**主接口**（[doc/08 §1.2](./doc/08-图Transformer策略网络详细设计.md#12-输入输出规格)）：

```python
def forward(
    self,
    node_features: torch.Tensor,   # (B, N, 18)
    edge_features: torch.Tensor,   # (B, N, N, 7)
    node_mask:     torch.BoolTensor,  # (B, N)
    edge_mask:     torch.BoolTensor,  # (B, N, N)
) -> dict:
    """
    返回：
        probs:  (B, N-1)  float32，去掉 ego（index 0），softmax 归一化
        h:      (B, N, hidden_dim)  所有节点隐状态
    """
```

**实现约束**：
- attention bias 用 `-1e9` 而非 `-inf`（[坑 7](#坑-7attention-mask-使用--1e9-而非--inf)）
- ego 节点（index 0）在最终 logits 中被裁掉，只保留候选节点（[doc/08 §3](./doc/08-图Transformer策略网络详细设计.md#3-graphtransformer类详细实现)）
- 完整多层实现见 [doc/08 §4 GraphTransformerLayer](./doc/08-图Transformer策略网络详细设计.md#4-graphtransformerlayer详细实现)

### 3.3 `modules/safety_shield.py`

**主接口**（[doc/09 §3](./doc/09-安全盾QP优化器详细设计.md#3-safetyshieldqp类详细实现)）：

```python
def solve(
    self,
    v_rl:      torch.Tensor,   # (B, 3)  策略速度
    obstacles: torch.Tensor,   # (B, N_obs, 7)  [p_x,p_y,p_z,v_x,v_y,v_z,radius]
    ego_pos:   torch.Tensor,   # (B, 3)  可选，用于计算相对位置
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    返回 (v_safe, intervention)：
        v_safe:       (B, 3)
        intervention: (B,)  = ||v_safe - v_rl||，失败时返回 0（不惩罚）
    """
```

**实现约束**：
- OSQP 求解失败（status != 'solved'）时 fallback：`v_safe = v_rl, intervention = 0`（[doc/09 §6](./doc/09-安全盾QP优化器详细设计.md#6-鲁棒性设计)）
- 空障碍物时直接返回 `v_safe = v_rl`，不调用求解器
- OSQP 参数从 `cfg` 读取而非硬编码（见 [doc/11 §2.2.1](./doc/11-代码改造实施方案.md#221-创建拓扑配置)）

### 3.4 `modules/hierarchical_control.py`

**主接口**（[doc/10 §3](./doc/10-分层控制架构设计.md#3-hierarchicalcontroller类详细实现)）：

```python
def __init__(self, high_freq: float, low_freq: float, pid_gains: dict, goal_horizon: float):
    self.freq_ratio = int(round(high_freq / low_freq))   # 必须取整！= 6

def reset(self, batch_size: int, device: torch.device) -> None: ...

def reset_partial(self, env_ids: torch.Tensor) -> None:
    """清零 env_ids 对应环境的 PID 积分项"""

def step(
    self,
    ego_pos:              torch.Tensor,           # (B, 3)
    ego_vel:              torch.Tensor,           # (B, 3)
    high_level_velocity:  torch.Tensor | None,    # (B, 3) 或 None
) -> tuple[torch.Tensor, dict]:
    """
    返回 (ctrl_vel, info)：
        ctrl_vel:  (B, 3)  PID 输出，直接发给飞控
        info:      {"tracking_error": (B,)}
    
    当 high_level_velocity 非 None 时，更新中间目标；
    当 None 时，沿用上一次中间目标继续跟踪。
    """
```

**实现约束**：
- `freq_ratio = int(round(high_freq / low_freq))` —— 不用浮点取模（[坑 4](#坑-4频率比必须取整为-6)）
- `reset_partial` 只清零对应 env 的积分，不影响其他 env（[坑 5](#坑-5reset_partial-必须在每次-envstep-之后立即调用)）
- PID 目标速度跟踪详见 [doc/10 §6 级联优化策略](./doc/10-分层控制架构设计.md#6-级联优化策略)

---

## 4. 关键坑（审计发现，文档里不明显）

### 坑 1：`ray_hits_w` 形状错误
`self.lidar.data.ray_hits_w` 的实际 shape 是 `(num_envs, lidar_hbeams * lidar_vbeams, 3)` 世界坐标，hbeams=36, vbeams=4，即 `(B, 144, 3)`。  
当前 `env.py` 的 `lidar_scan` 计算中已经正确使用了这个数据，但有注释写的是 `lidar_resolution = (lidar_hbeams, lidar_vbeams)`，**不要把 2D 分辨率理解为点云形状**。  
传给 `extract_topology` 时直接用 `self.lidar.data.ray_hits_w`，不需要 reshape。参考 [doc/07 §1.1](./doc/07-拓扑提取模块设计.md#11-模块职责)。

### 坑 2：`node_positions` 在两个地方都要用
`extract_topology` 返回的 `node_positions` 需要：  
① 写入 `obs` dict 作为观测特征  
② 在 `_pre_sim_step` 中被 `_node_to_velocity()` 消费（把离散动作映射回速度）  

别只写入 obs 而忘记在 `_pre_sim_step` 里读它，否则 action 执行时找不到节点坐标。

### 坑 3：动作索引偏移（ego 占 index 0）
`GraphTransformer` 输出 `probs: (B, N-1)`，动作空间是候选节点 `[0, N-2]`，不包含 ego。  
`node_positions` 里 ego 在 index 0，所以把动作映射回坐标时必须 +1：

```python
# ppo.py _forward_graph 中：
action = dist.sample()   # (B,) ∈ [0, N-2]

# env.py _pre_sim_step 中：
real_idx = action + 1    # (B,) ∈ [1, N-1]，跳过 index 0 的 ego
selected_pos = node_positions[torch.arange(B, device=d), real_idx]
```

见 [doc/08 §1.2](./doc/08-图Transformer策略网络详细设计.md#12-输入输出规格)。

### 坑 4：频率比必须取整为 6
`62.5 / 10 = 6.25`，浮点取模（`step % 6.25`）无意义，正确写法：

```python
self.freq_ratio = int(round(high_freq / low_freq))  # = 6，不要用 6.25
```

累积误差 40ms / 1000 步，可接受，分析见 [doc/06 §4.2](./doc/06-新架构总览-拓扑图导航系统.md#42-频率同步精度分析)。

### 坑 5：`reset_partial()` 必须在 `_reset_idx` 中调用
`SyncDataCollector` 封装了 env.step，你无法从 train.py 的主循环中感知到哪些 env done。  
唯一正确位置是 `NavigationEnv._reset_idx(env_ids)`——它在回合结束时被 IsaacEnv 基类调用，参数正好是 done 的 env id。追加到 `_reset_idx` 末尾：

```python
# env.py _reset_idx 末尾
if hasattr(self, 'hierarchical_controller'):
    self.hierarchical_controller.reset_partial(env_ids)
```

**不要在 train.py 的 `for i, data in enumerate(collector)` 循环里写 `reset_partial`，那里拿不到 per-env done 信息。**  
见 [doc/10 §5](./doc/10-分层控制架构设计.md#5-状态同步机制)。

### 坑 6：QP 失败的 fallback（intervention=0，不是 NaN）
`osqp` 返回 `infeasible` 时，必须：

```python
if result.info.status != 'solved':
    return v_rl.clone(), torch.zeros(B, device=v_rl.device)   # 不报异常
```

如果直接 raise 或返回 `v_rl * float('nan')`，会导致飞控收到非法命令，仿真崩溃。见 [doc/09 §6](./doc/09-安全盾QP优化器详细设计.md#6-鲁棒性设计)。

### 坑 7：Attention mask 用 `-1e9`，不用 `-inf`
```python
# ❌ 会在全节点被 mask 时让 softmax 输出 NaN
logits = logits.masked_fill(~valid_mask, float('-inf'))

# ✅ 数值稳定
logits = logits.masked_fill(~valid_mask, -1e9)
```

见 [doc/08 §3](./doc/08-图Transformer策略网络详细设计.md#3-graphtransformer类详细实现)。

### 坑 8：`GraphTransformer` 不要放在 `env.py` 里实例化
当前 [doc/11 §3.1 代码模板](./doc/11-代码改造实施方案.md#31-拓扑提取模块) 的示例代码有个 `self.graph_policy = GraphTransformer(...)` 在 `env.__init__` 里，这是**错误的**。  
`GraphTransformer` 是策略网络，有需要被 PPO optimizer 优化的参数，必须在 `ppo.py` 中构建并由 `self.graph_transformer_optim` 管理。`env.py` 只负责拓扑提取（`TopoExtractor`）、控制（`HierarchicalController`）和安全（`SafetyShieldQP`），三者均**无需梯度**，不挂在 optimizer 上。

### 坑 9：`dyn_obs_states` 传给 `extract_topology` 的 shape
当前 `env.py _compute_state_and_obs` 里，`dyn_obs_states` 的最终 shape 是 `(B, 1, N_dyn, 10)`（有个 unsqueeze(1) 的维度），传入 `extract_topology` 前需要 squeeze：

```python
topo_dict = self.topo_extractor.extract_topology(
    ...
    dyn_obs=dyn_obs_states.squeeze(1),   # (B, N_dyn, 10)，不是 (B, 1, N_dyn, 10)
    ...
)
```

---

## 5. 数据接口速查

> 只用来核对 shape，完整字段定义见各模块文档。

| 变量 | Shape | 来源 | 去向 | 文档参考 |
|------|-------|------|------|---------|
| `self.lidar.data.ray_hits_w` | `(B, 144, 3)` | Isaac Sim LiDAR | `TopoExtractor.extract_topology` | [doc/07 §1.1](./doc/07-拓扑提取模块设计.md#11-模块职责) |
| `topo_dict['node_features']` | `(B, N+1, 18)` | `TopoExtractor` | obs dict & `GraphTransformer` | [doc/07 §3.2](./doc/07-拓扑提取模块设计.md#32-节点特征编码) |
| `topo_dict['node_positions']` | `(B, N+1, 3)` | `TopoExtractor` | obs dict & `_node_to_velocity` | [doc/07 §3.2](./doc/07-拓扑提取模块设计.md#32-节点特征编码) |
| `topo_dict['edge_features']` | `(B, N+1, N+1, 7)` | `TopoExtractor` | obs dict & `GraphTransformer` | [doc/07 §4](./doc/07-拓扑提取模块设计.md#4-边特征编码) |
| `graph_out['probs']` | `(B, N)` | `GraphTransformer` | `Categorical` 采样 | [doc/08 §1.2](./doc/08-图Transformer策略网络详细设计.md#12-输入输出规格) |
| `graph_out['h']` | `(B, N+1, 64)` | `GraphTransformer` | Critic（取 `[:, 0, :]`） | [doc/08 §1.2](./doc/08-图Transformer策略网络详细设计.md#12-输入输出规格) |
| `v_rl` | `(B, 3)` | `_node_to_velocity` | `SafetyShieldQP.solve` | [doc/06 §2.2.2](./doc/06-新架构总览-拓扑图导航系统.md#222-新架构数据流) |
| `v_safe` | `(B, 3)` | `SafetyShieldQP` | `HierarchicalController.step` | [doc/09 §1.3](./doc/09-安全盾QP优化器详细设计.md#13-模块依赖图) |
| `intervention` | `(B,)` | `SafetyShieldQP` | 奖励塑形 | [doc/09 §8](./doc/09-安全盾QP优化器详细设计.md#8-干预分析与奖励设计) |
| `ctrl_vel` | `(B, 3)` | `HierarchicalController` | `env._pre_sim_step → drone` | [doc/10 §3](./doc/10-分层控制架构设计.md#3-hierarchicalcontroller类详细实现) |

---

## 6. 验收标准

提交每个 Phase 前必须全部通过（见 [doc/11 §6](./doc/11-代码改造实施方案.md#6-测试与验证)）：

- [ ] `pytest training/scripts/test/` 全部通过
- [ ] `mypy --strict training/scripts/modules/*.py` 无 error
- [ ] `mode=ppo`：原始训练链路启动，运行 10 步无报错（向后兼容验证）
- [ ] `mode=graph_ppo`：完整链路 `lidar→topo→graph→action→vel→qp→step` 以 batch=64 运行 100 步，无 NaN/inf，无异常
- [ ] `intervention` 在随机策略下不全为零（QP 正在生效）
- [ ] `topo_dict['node_positions']` 的值确实是世界坐标（数量级与 `ray_pos_w` 一致，不是局部坐标）
- [ ] `freq_ratio == 6`（不是 6.25），高层更新恰好每 6 步触发一次
- [ ] `_reset_idx` 被调用后，被 reset 的 env 其 PID 积分为零
- [ ] 无 `print()`，无绝对路径
