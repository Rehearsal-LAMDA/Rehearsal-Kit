你可以直接用下面这个 prompt 作为 NeurIPS 2025 MUR worker agent 的任务说明：

你是 Rehearsal 项目的实现型 worker agent，负责把 NeurIPS 2025 工作
`Variance-Reduced Long-Term Rehearsal Learning with Quadratic Programming
Reformulation` 接入当前统一 Python package。

项目根目录：当前 Rehearsal 仓库根目录

背景：
- 当前项目已经有 `rehearsal.core.AUFTask`、`rehearsal.core.DecisionResult`、
  `rehearsal.methods.registry`、seeded batch experiment runner，以及多个已迁移
  method adapter。请适配这些现有接口，不要重新设计 framework。
- NeurIPS 2025 在当前 `ExecPlan.md` 里曾被标为 reference-only；本任务就是下一轮
  专门把它迁入 package。
- 原始材料位于 `previous_works/07-NeurIPS 2025/`：
  - 论文：`previous_works/07-NeurIPS 2025/paper/neurips_2025.tex`
  - GMuR 原始 solver：`previous_works/07-NeurIPS 2025/code/GMuR/Greedy_Slover.py`
  - GMuR 原始实验：`previous_works/07-NeurIPS 2025/code/GMuR/main_toy.py`、
    `main_syn.py`、`main_bermuda.py`
  - FarMuR 原始 solver：
    `previous_works/07-NeurIPS 2025/code/FarMuR/FarSight_Slover.py`
  - FarMuR 原始实验：`previous_works/07-NeurIPS 2025/code/FarMuR/main_toy.py`、
    `main_syn.py`、`main_bermuda.py`
- 旧代码只能作为只读参考，不要修改 `previous_works/` 下的任何文件。
- 不要直接复制 legacy 脚本式全局代码，也不要把 plotting、progress bar、dataframe、
  cvxopt、sklearn 等研究脚本依赖变成 package runtime 依赖。

特别注意模型层：
- NeurIPS 2025 的模型虽然也是 linear additive，但它不是 ICML 2025 CARE 那种
  单轮静态 `LinearGaussianSRM` 问题。这里的核心是时间序列 SRM：

```text
V_t = A V_t + B V_{t-1} + eps_t
```

- 不能直接沿用 `src/rehearsal/models/linear_gaussian.py` 中面向单轮 DAG/path-effect
  的 `LinearGaussianSRM.effect_matrices(...)` 风格来实现 MUR。那个模型不表示
  lagged matrix `B`、长期聚合目标、remaining horizon、FarMuR center update，也不适合
  legacy 代码中的 mutually influenced / time-series setting。
- 可以借鉴 ICML 2025 adapter 的 package 组织方式：method adapter 薄、模型学习在
  `models/`、决策优化在 `optimizers/`、输出统一为 `DecisionResult`。但 MUR 需要
  手动新增一个专用的 time-series linear additive model，并按 NeurIPS 2025 论文和
  legacy solver 计算 `M/N/H/F`。

GMuR 和 FarMuR 必须放在同一个 package method 下。注册名只能是一个稳定名称：

```python
"mur": MURRehearsal
```

不要额外注册 `"gmur"`、`"farmur"`、`"far-mur"`、`"greedy-mur"` 等 alias。
GMuR/FarMuR 的选择通过同一个 `mur` method 的可选参数完成，例如：

```bash
--method mur --method-params variant=gmur,horizon=0
--method mur --method-params variant=farmur,horizon=8
```

你的 ownership：
- `src/rehearsal/methods/mur.py`
- `src/rehearsal/optimizers/mur.py`
- `src/rehearsal/models/time_series.py` 或 `src/rehearsal/models/mur.py`
  （新增 MUR 专用 linear additive time-series SRM；不要把它硬塞进 ICML 2025 的
  `LinearGaussianSRM`）
- `src/rehearsal/metrics/mur.py`，仅在确实需要共享 desired-region center、
  variance、AUF probability、matrix diagnostics 等轻量函数时新增
- `examples/mur/mur_toy_experiment.py`
- `examples/mur/bermuda_example.py`，必须实现一个与当前 README Bermuda 示例体系
  统一的 method-runner config，用来生成 README 表格里的 MUR reference result
- `tests/test_mur.py`
- 如需让 CLI 可用，可以小范围修改：
  - `src/rehearsal/methods/registry.py`
  - `src/rehearsal/methods/__init__.py`
  - `src/rehearsal/models/__init__.py`
  - `README.md` 的 implemented method 表和结果表。CLI 示例可以暂时不加入 README
- 如有必要，可以对 `src/rehearsal/core/` 做很小的兼容性补充；不要重构现有
  framework。

目标：
实现 NeurIPS 2025 GMuR/FarMuR 的统一 package adapter，使其通过当前统一接口工作：

- `fit(data, task, config=None)`
- `suggest(observation, task)`
- `evaluate(task, n_samples)`

`task` 必须使用 `rehearsal.core.AUFTask`。`suggest` 必须返回当前
`rehearsal.core.DecisionResult` 实例；不要在实现或文档中重新定义
`DecisionResult` 字段，以当前 `rehearsal.core` 的定义为准。

建议 public class：

```python
class MURRehearsal:
    ...

NeurIPS2025MURRehearsal = MURRehearsal
```

构造参数建议至少包含：
- `variant: str = "gmur"`，规范化后只允许 `"gmur"` 或 `"farmur"`；
  可以接受 `"greedy"`、`"far_sighted"`、`"farsighted"` 作为输入别名，但 diagnostics
  里必须输出规范值 `"gmur"` / `"farmur"`，并且 registry 不注册别名。
- `horizon: int = 0`，表示论文里的 \(L=t_e-t_0\)，也就是当前轮之后还考虑多少个
  future steps；实际决策轮数是 `horizon + 1`。如需兼容 legacy `window_length`，
  只能作为 constructor/config alias，映射为 `horizon = window_length - 1`。
- `candidate_alteration_sets: Sequence[Sequence[str]] | None = None`
- `n_mc_samples: int = 256`，用于 `DecisionResult.estimated_success_probability`
  或 `evaluate(...)` 的轻量 Monte Carlo 估计
- `bounded_solver: str = "projected-gradient"` 或等价轻量 box-QP solver 选项
- `learning_rate` / `max_iters` / `tolerance` / `num_restarts` 等优化参数
- `previous_state_prefix: str = "prev__"`，用于从 `observation` 中读取
  \(v_{t-1}\)
- `seed: int | None = None`

请优先实现一个 CPU 可运行、依赖轻量、可测试的版本：
- 核心实现只依赖 `numpy`。
- 不要添加新的 production dependency。
- 不要把 legacy 脚本中的 `pandas`、`sklearn`、`tqdm`、`matplotlib`、`cvxopt`、
  `brokenaxes` 等变成 package runtime 依赖。
- 如果 `scipy` 在本地可用，可以只在现有 Bermuda `.mat` 数据加载路径中使用；
  MUR 的模型学习、矩阵计算、QP/box-QP、evaluation 不应依赖它。
- 不要引入 torch、cvxopt、rpy2、FrEIA 等新依赖。

方法要点：
1. 阅读 NeurIPS 2025 论文的这些部分，抽取核心概念，而不是复制脚本：
   - `Sec. 3.1` formulation：长期目标
     \(\bar{Y} = \frac{1}{L+1}\sum_{i=t}^{t+L}Y_i\)
   - `Sec. 3.2` QP reformulation：
     \(\min_{\tilde z_t^\xi}\|M x_t + N v_{t-1} + H\tilde z_t^\xi - s\|_2^2\)
   - `Alg. 1` GMuR：每轮设 `T=0`，只做当前轮 greedy 最优
   - `Alg. 2` FarMuR：每轮按 remaining horizon 重新计算 `M,N,H`，按
     `(T+1)/(L+1)` reweight，执行当前 action，并在 rollout 中用已收到的 `y_t`
     更新 region center
   - `Appx. Proof of Prop. 3.1` 中 `Eq. (computation)`：计算 `M,N,H,F`
   - `Thm. 3.5` variance reduction：GMuR/FarMuR 的 horizon 行为
2. 新代码必须围绕当前 `AUFTask`：
   - `task.observed_variables` 对应当前轮可观察的 `X_t`
   - `task.alterable_variables` 对应可干预的 `Z`
   - `task.outcome_variables` 对应 `Y`
   - `task.desired_region` 对应 desired region `S`
   - `task.alteration_domain` 提供 box bounds；论文无界解是基础，但 package 输出
     必须尊重 bounds
   - `task.candidate_alteration_sets` 仍按当前 package 规范支持；默认候选应是
     `task.alterable_variables` 整体
   - `task.variable_order` 是时序 SRM 矩阵 `A/B/covariance` 的列顺序，必须显式校验
3. MUR 需要 \(v_{t-1}\) 和 \(x_t\)。建议 `suggest(observation, task)` 采用：
   - 当前观测 `x_t`：从 `observation[name]` 读取 `name in task.observed`
   - 前一轮完整状态 `v_{t-1}`：默认从 `observation[f"prev__{name}"]` 读取
     `name in task.variable_order`
   - 如果缺少前一轮状态，可以按优先级 fallback：
     `task.metadata["mur_initial_state"]` -> fit data 的最后一行 -> 全零向量。
     但 diagnostics 必须写清楚 `previous_state_source`；对非 toy 配置，推荐显式传入
     `prev__*`。
4. desired region center `s` 必须可解释：
   - 对 `DesiredRegion.from_intervals(...)` 生成的 box/interval region，从上下界恢复
     midpoint。
   - 对 `circular_region_inner_care(...)` 或其它 region，如果 metadata 里有 `center`，
     使用 metadata center。
   - 其它 region 要求 `task.metadata["mur_region_center"]` 或 config
     `region_center` 明确提供；否则 raise `ValueError`。
   - diagnostics 里记录 `region_center_source`。理论要求 centrally symmetric convex
     region；不能静默假装任意 polytope 都满足。

模型层硬要求：
- 新增的 time-series model 应直接表达 legacy solver 里的 `A_true`、`B_true`、
  `C_true/noise covariance` 和估计得到的 `A_est`、`B_est`。
- 不要用 `total_path_effects(...)` 这类静态 DAG path-effect 逻辑替代
  NeurIPS 2025 的矩阵递推。MUR 要通过 `Gamma`、`U`、`U_tilde`、`Xi` 等矩阵显式
  处理同轮影响和 lagged influence。
- 允许模型有 bidirectional / mutually influenced 变量，只要 `I - P A` 可逆或可用
  pinv 稳定求解；这正是 legacy synthetic 中 `competitor_pricing` 与 `self_pricing`
  的 setting。
- `LinearTimeSeriesSRMLearner` 应支持三种来源：
  - 直接读取 `fit_config["mur_A"]`、`fit_config["mur_B"]`、
    `fit_config["mur_noise_covariance"]`
  - 从 parent-to-child dict 构造矩阵：
    `fit_config["mur_instantaneous_theta"]`、`fit_config["mur_lagged_theta"]`
  - 从 time-ordered data 按 mask 估计 `A` 和 `B`；mask 来自 `task.parents` 和
    `task.metadata["mur_lagged_parents"]` / `fit_config["mur_lagged_parents"]`

建议实现结构：
- `src/rehearsal/models/time_series.py`
  - `LinearTimeSeriesSRM`
    - `variable_order`
    - `instantaneous_matrix` / `A`
    - `lagged_matrix` / `B`
    - `noise_covariance`
    - helper: `selection_matrix(variables)`
    - helper: `sample_next(...)` 或 `simulate_rollout(...)`，用于 lightweight evaluation
  - `LinearTimeSeriesSRMLearner`
    - 支持直接从 `fit_config["mur_A"]`、`fit_config["mur_B"]`、
      `fit_config["mur_noise_covariance"]` 构造 model
    - 支持从 parent-to-child dict 构造矩阵：
      `fit_config["mur_instantaneous_theta"]`、`fit_config["mur_lagged_theta"]`
    - 支持从 time-ordered data 估计：
      `A` 的 mask 默认来自 `task.parents`，`B` 的 mask 默认来自
      `task.metadata["mur_lagged_parents"]` 或 `fit_config["mur_lagged_parents"]`
      （child -> lagged parents）
    - 估计时对每个 child 做 least squares：
      当前行 child 作为 `y`，当前行 instantaneous parents 和上一行 lagged parents
      作为 design；用 `np.linalg.pinv` 处理奇异矩阵
    - 残差协方差加 `min_variance * I`
- `src/rehearsal/optimizers/mur.py`
  - `MURMatrixBundle`
  - `compute_mur_matrices(model, task, candidate, remaining_horizon)`
  - `solve_mur_box_qp(M, N, H, x_t, v_prev, center, lower, upper, ...)`
  - `select_mur_action(...)`
  - `rollout_mur_policy(...)`，供 `evaluate(...)` 和 examples 的 true/surrogate
    evaluation 复用
- `src/rehearsal/methods/mur.py`
  - `MURRehearsal`
  - `fit`：学习或读取 time-series SRM，缓存模型、中心、候选集、diagnostics
  - `suggest`：对当前 observation 选择一个 bounded alteration，返回
    `DecisionResult`
  - `evaluate`：基于最近一次 `suggest` 的 context 做轻量 model-based rollout 或
    surrogate success estimate；如果 experiment config 提供 true simulator，则由
    config 的 `evaluate` 函数负责计算 true long-term AUF success rate
- `examples/mur/mur_toy_experiment.py`
  - 一个小型线性时序 toy，变量可以是 `("x", "z", "y")`
  - 必须能用 `rehearsal.experiments.run` 跑通 `variant=gmur` 和 `variant=farmur`
- `examples/mur/bermuda_example.py`
  - 必须实现，并与 README 中其它 Bermuda method examples 的 batch output contract
    保持统一：`runs` + `summary`，custom `evaluate(...)` 返回
    `true_auf_success_rate`、`no_action_true_auf_success_rate`、`eval_samples`
  - 该 README reference run 使用 legacy window length `1`。按本 package 参数约定，
    这等价于 `horizon=0`，也就是只考虑当前 round。此时 GMuR 与 FarMuR 必须退化为
    相同行为，README 表格可以只报告一个 `mur` 结果，或报告 `mur` with
    `variant=gmur/farmur` 且两者数值相同/近似相同。
  - 尽量复用当前 `rehearsal.datasets.bermuda` 的变量和 covariance
  - 需要补充 NeurIPS 2025 的 lagged `B` 矩阵或 metadata
  - 注意当前 `src/rehearsal/datasets/bermuda.py` 的 alterable set 不包含 `pHsw`、
    `CO2`，而 NeurIPS 2025 legacy Bermuda 代码把
    `DIC, TA, Omega, Nutrients_PC1, Chla, pHsw, CO2` 都放进 `Z`。如果要保持
    NeurIPS 2025 原实验设定，example 中应构造专用 MUR Bermuda task，而不是强行
    复用 CARE/Grad-Rh Bermuda task 的 alterable set。

矩阵计算要求：
实现时请把论文公式转为明确、可测的函数。设 `V` 的顺序为 `task.variable_order`，
`E_x`、`E_z`、`E_y` 分别为列选择矩阵。

论文在 `V=[X,Z,Y]` 且所有 `Z` 都被 alter 的情况下使用：

```text
U       = inv(I - (E_x E_x.T + E_y E_y.T) A) E_z
C       = inv(I - (E_x E_x.T + E_y E_y.T) A) (E_x E_x.T + E_y E_y.T)
Gamma   = C B
U_tilde = inv(I - E_y E_y.T A) E_z
C_tilde = inv(I - E_y E_y.T A) E_y E_y.T
Gamma_tilde = C_tilde B
Xi      = inv(I - E_y E_y.T A) E_x
```

为了兼容 `candidate_alteration_sets` 和可能存在的非 action 中间变量，建议推广为：

```text
P_future = I - E_action E_action.T
P_now    = I - E_x E_x.T - E_action E_action.T

U       = inv(I - P_future A) E_action
C       = inv(I - P_future A) P_future
Gamma   = C B
U_tilde = inv(I - P_now A) E_action
C_tilde = inv(I - P_now A) P_now
Gamma_tilde = C_tilde B
Xi      = inv(I - P_now A) E_x
```

当 action candidate 等于所有 `task.alterable_variables` 且变量只分为 `X/Z/Y`
时，这个推广会退化回论文公式。

对 remaining horizon `T`，按论文 `Eq. (computation)` 计算：

```text
S_T = sum_{i=0}^T Gamma^i
M = (1/(T+1)) E_y.T S_T Xi
N = (1/(T+1)) E_y.T S_T Gamma_tilde

H = (1/(T+1)) E_y.T [
  I U,
  (I + Gamma) U,
  ...,
  (sum_{i=0}^{T-1} Gamma^i) U,
  (sum_{i=0}^{T} Gamma^i) U_tilde
]
```

`H` 的 block 顺序要写清楚并加测试。论文定义
`\tilde z_t = [z_{t+T}, ..., z_t]`，legacy `FarSight_Slover.py` 也是解完整
sequence 后取最后一个 block 作为当前 `z_t`。论文 Alg. 2 文字里出现
`tilde_z[:|z|]`，这和 stack 定义不一致；实现时请跟 stack 定义和 legacy solver
保持一致，或者在代码中改用 chronological stack `[z_t, ..., z_{t+T}]` 并把公式和
测试一起调整。不要在没有测试的情况下混用两种 block 顺序。

QP / box-QP 要求：
- 基础目标：

```text
min_z ||H z - (center - M x_t - N v_prev)||_2^2
```

- 无界解可用 `H.T @ pinv(H @ H.T) @ b` 或 `np.linalg.lstsq(H, b)`。
- Package 必须尊重 `task.alteration_domain` 的 bounds。若无界解越界，不要直接返回
  越界值。可实现轻量 projected-gradient box-QP：
  - objective: `||H z - b||_2^2`
  - gradient: `2 * H.T @ (H @ z - b)`
  - 初始化包含 clipped unbounded solution、box midpoint、lower、upper、seeded random
    restarts
  - 每步投影到 `[lower, upper]`
  - deterministic under fixed seed
- 对 rank-deficient `H` 使用 pinv / projected gradient fallback，并在 diagnostics 写
  `rank`, `used_pinv`, `solver_status`。
- 多 candidate 时，对每个 candidate 独立解；选择 objective 最小者，tie-breaker
  依次用 estimated success probability 更高、cost 更低。

GMuR/FarMuR 变体要求：
- `variant="gmur"`：
  - 每次当前决策固定使用 `remaining_horizon_for_solver = 0`
  - `M,N,H` 可在 fit 后按 candidate 预计算，也可 lazy cache
  - 不更新 region center
  - 对长期 evaluation，可在 rollout 每轮重复调用当前 greedy action 规则
- `variant="farmur"`：
  - 对当前 round 使用 `remaining_horizon = horizon`，在 rollout 中每轮递减
  - 按论文 Alg. 2 在每轮使用 `((T + 1) / (L + 1)) * (M,N,H)` reweight
  - 每轮只执行当前 action block，丢弃未来 block
  - rollout 中收到 `y_t` 后更新 `center = center - y_t / (L + 1)`
  - 当 `horizon=0` 时，FarMuR 应与 GMuR 退化为同一行为；测试必须覆盖

`fit(...)` 应完成并缓存：
- time-ordered training matrix 和 column names
- `A`、`B`、noise covariance、父节点/lagged 父节点 metadata
- stationarity diagnostics：
  - `spectral_radius_natural = rho(inv(I-A) @ B)` 或等价 solve
  - 如果可计算 altered process，也记录 altered spectral radius
  - 默认 `allow_unstable=False` 时，半径 `>= 1` 应 raise `ValueError`
- selection matrices 或可复用 index metadata
- desired-region center 和来源
- candidate list

`suggest(...)` 应：
- 校验 `fit` 已完成
- 从 observation 中构造 `x_t` 和 `v_{t-1}`
- 根据 `variant` 和 `horizon` 计算当前 action
- 对所有 candidate 求解 bounded QP
- 返回当前 `DecisionResult`：
  - `alterations` 只包含所选 current candidate 的变量
  - `estimated_success_probability` 必须在 `[0, 1]`
  - `cost` 使用 `task.alteration_domain.cost(...)`
  - `runtime_seconds` 是 suggest 阶段耗时
- 结果 deterministic under fixed seed。所有随机 restart / MC sampling 必须走
  method-local `np.random.default_rng(seed)`，不要使用全局 `np.random`。

`evaluate(...)` 可以分两层实现：
- 最低要求：对最近一次 `suggest` 的 action 做 fitted-model surrogate evaluation，
  返回当前或 aggregate desired-region success estimate，字段至少包括
  `estimated_success_probability`、`n_samples`、`alterations`。
- 更好的实现：用 `rollout_mur_policy(...)` 从最近 context 开始模拟
  `horizon + 1` 个 rounds，报告：
  - `estimated_success_probability`
  - `aggregate_success_rate`
  - `current_round_success_rate`
  - `variant`
  - `horizon`
  - `n_samples`
  - `alterations`
- 如果 example config 有 true simulator，runner 中的 custom `evaluate(...)` 负责
  报告 `true_auf_success_rate` 和 `no_action_true_auf_success_rate`。

诊断信息至少包含：
- `method_family`: `"NeurIPS 2025 MUR"`
- `variant`: `"gmur"` 或 `"farmur"`
- `horizon`
- `selected_candidate`
- `objective_value`
- `solver_status`
- `n_candidates`
- `n_training_samples`
- `previous_state_source`
- `region_center`
- `region_center_source`
- `stationarity_spectral_radius`
- `candidate_diagnostics`
- matrix shape/rank diagnostics：`M_shape`、`N_shape`、`H_shape`、`H_rank`
- bounded solver metadata：`used_unbounded_solution`、`used_pinv`、`n_iters`

测试要求：
添加 `tests/test_mur.py`，至少覆盖：
- tiny linear time-series AUF task 可以 `fit`、`suggest`、`evaluate`
- registry 可以通过 `"mur"` 创建 method
- `variant="gmur"` 和 `variant="farmur"` 都能跑通
- `variant` 非法值会 raise `ValueError`
- `horizon=0` 时 FarMuR 与 GMuR 产生相同或数值上近似相同的 current action
- 输出 alterations 尊重 `task.alteration_domain` bounds
- fixed seed 下 deterministic
- desired-region center 可以从 interval region 推断
- metadata/config 提供 center 的路径可用
- 缺少 center 且无法推断时 raise
- `M,N,H` 计算在一个手算的一维/二维 toy case 上 shape 和数值正确
- block 顺序测试：FarMuR 解 sequence 后执行的是当前 action block，不是未来 block
- rank-deficient `H` 不崩溃，diagnostics 写出 fallback
- `previous_state_prefix` 读取 `prev__*` 成功；fallback source 也写入 diagnostics
- 多个 candidate 时能返回合法 selected candidate
- diagnostics 字段存在且 JSON-friendly
- `evaluate(...)` 返回 probability in `[0, 1]`

如果新增 example config，再添加 runner smoke test：
- `run_experiment_configs("examples/mur/mur_toy_experiment.py", seeds=(1,), method_name="mur", method_params={"variant": "gmur", ...})`
  返回 batch payload
- `variant="farmur"` 的 runner smoke 也至少跑一次
- `run_experiment_configs("examples/mur/bermuda_example.py", seeds=(1,), method_name="mur", method_params={"variant": "gmur", "horizon": 0, ...}, ...)`
  返回 batch payload，并包含 `true_auf_success_rate`
- 对 Bermuda `horizon=0` 再跑一次 `variant="farmur"`，断言与 GMuR 的 selected action
  和 true/surrogate result 数值相同或在浮点容差内一致

建议 README / output 更新：
- 在 README 的 implemented method 表中新增：
  - registry: `mur`
  - venue: `2025 NeurIPS`
  - paper: `Variance-Reduced Long-Term Rehearsal Learning with Quadratic Programming Reformulation`
  - example configs: `examples/mur/`
  - setting: `Long-term / multi-round AUF with time-series SRM`
- 在 README 的 Bermuda results 表中补充 MUR reference result。该 reference result
  使用 window length `1` / `horizon=0`，所以 GMuR 和 FarMuR 效果应一致；表格可写为
  `mur` 或 `mur (GMuR/FarMuR, window length 1)`，不要制造两个不同数值。
- README CLI 调用可以暂时不放。不要在 README 里新增未验证或会增加维护负担的长
  CLI 块。
- 需要生成至少一个 Bermuda reference output，并与 README 表格对应。建议路径：
  - `outputs/mur_bermuda_seed3.json`
  如确实同时保存两个 variant outputs，则必须确认 `horizon=0` 下两者 action 和
  result 一致：
  - `outputs/mur_gmur_bermuda_seed3.json`
  - `outputs/mur_farmur_bermuda_seed3.json`

验证命令：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests
```

如生成 outputs，再额外跑对应 runner command，例如：

```bash
PYTHONPATH=src python -m rehearsal.experiments.run examples/mur/mur_toy_experiment.py \
  --method mur \
  --seeds 3 \
  --method-params variant=gmur,horizon=0 \
  --eval-samples 100 \
  --compact
```

```bash
PYTHONPATH=src python -m rehearsal.experiments.run examples/mur/mur_toy_experiment.py \
  --method mur \
  --seeds 3 \
  --method-params variant=farmur,horizon=2 \
  --eval-samples 100 \
  --compact
```

README 对齐的 Bermuda reference run 使用 window length `1` / `horizon=0`：

```bash
PYTHONPATH=src python -m rehearsal.experiments.run examples/mur/bermuda_example.py \
  --method mur \
  --seeds 3 \
  --method-params variant=gmur,horizon=0 \
  --params n_data=2000 \
  --eval-samples 1000 \
  --output outputs/mur_bermuda_seed3.json \
  --compact
```

如果还跑 FarMuR 版本，只用于确认退化一致性，不需要把 CLI 暂时放进 README。

完成时请报告：
- 修改了哪些文件
- NeurIPS 2025 legacy code 中哪些部分已经被接入
- `GMuR` 和 `FarMuR` 如何通过同一个 `mur` method 的 `variant` 参数选择
- 是否实现了 toy / Bermuda example，以及对应 runner 命令结果
- 哪些 legacy 实验脚本、plotting、CVX/Sklearn 相关部分只作为参考或保留为 TODO/backlog
- pytest 结果

你不是唯一一个 agent。不要回滚或重写其他 agent 的修改；如果发现 framework API
已经变化，请适配现有接口，而不是重建一套接口。

本任务只实现 NeurIPS 2025 MUR adapter，不实现或重写 NeurIPS 2023、NeurIPS
2024、AAAI 2025、ICML 2025、IJCAI 2025、ICLR 2026、CME、OLEM-Rh。不要扩大范围。

关键是：不要只写“帮我实现 GMuR/FarMuR”。要把输入旧代码、目标新 API、同一
`mur` registry 名、`variant` 参数、允许修改的文件、测试验收、依赖边界和禁止事项
全部落实。这样 worker 更容易产出能合并的代码，而不是又生成一套新的研究脚本。
