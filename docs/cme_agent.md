你可以直接用下面这个 prompt 作为 CME-Rh worker agent 的任务说明：

你是 Rehearsal 项目的实现型 worker agent，负责把未发表工作
`Non-Parametric Rehearsal Learning via Conditional Mean Embeddings` 接入当前统一
Python package。

项目根目录：当前 Rehearsal 仓库根目录

背景：
- 当前项目已经有 `rehearsal.core.AUFTask`、`rehearsal.core.DecisionResult`、
  `rehearsal.methods.registry`、seeded batch experiment runner 和 ICML 2025 CARE
  adapter。请适配这些现有接口，不要重新设计 framework。
- 旧代码位于 `previous_works/unpublished/code/`。
- 重点参考：
  - `previous_works/unpublished/code/Rh_Solver.py`
  - `previous_works/unpublished/code/main_lin_syn1.py`
  - `previous_works/unpublished/code/main_nonlin_syn1.py`
  - `previous_works/unpublished/code/main_nonlin_syn2.py`
  - `previous_works/unpublished/code/main_bank.py`
  - `previous_works/unpublished/code/main_bermuda.py`
  - `previous_works/unpublished/code/base_*.py` 只作为 no-action baseline 参考
- 旧代码只能作为只读参考，不要修改 `previous_works/` 下的任何文件。
- 同一目录里的 `Order-Based Rehearsal Learning.pdf` 和 order-based 方法不在本任务范围内。

你的 ownership：
- `src/rehearsal/methods/cme.py`
- `src/rehearsal/optimizers/cme.py`
- `src/rehearsal/metrics/cme.py`，仅在确实需要共享指标/核函数时新增
- `src/rehearsal/datasets/cme.py` 或 `examples/cme/` 中的轻量实验配置
- `tests/test_cme_rh.py`
- 如需让 CLI 可用，可以小范围修改：
  - `src/rehearsal/methods/registry.py`
  - `src/rehearsal/methods/__init__.py`
- 如有必要，可以对 `src/rehearsal/core/` 或 `src/rehearsal/models/` 做很小的兼容性补充；
  不要重构现有 framework。

目标：
实现 CME-Rh / kernel conditional mean embedding rehearsal 方法的 package
adapter，使其通过当前统一接口工作：

- `fit(data, task, config=None)`
- `suggest(observation, task)`
- `evaluate(task, n_samples)`

`task` 必须使用 `rehearsal.core.AUFTask`。`suggest` 必须返回当前
`rehearsal.core.DecisionResult` 实例；不要在实现或文档中重新定义
`DecisionResult` 字段，以当前 `rehearsal.core` 的定义为准。

方法注册名使用一个稳定名称：

```python
"cme-rh": CMERehearsal
```

不要额外添加 `"cme"`、`"kernel-cme"` 等重复 alias。

请优先实现一个 CPU 可运行、依赖轻量、可测试的版本：
- 使用 `numpy` 实现 RBF kernel、带界投影和线性系统求解。
- 不要添加新的 production dependency。
- 不要把 legacy 脚本中的 `pandas`、`sklearn`、`tqdm`、`matplotlib`、`mpmath`
  变成 package runtime 依赖。
- 如果 `scipy` 在本地可用，可以只在数据加载或可选路径中使用；核心 solver 不应依赖它。
- 不要引入 torch、cvxopt、rpy2、FrEIA 等新依赖。

实现要求：
1. 阅读 legacy CME solver，抽取核心概念，而不是复制脚本式全局代码：
   - observed/context variables `X`
   - alteration/action candidate set `Rh`
   - unaltered stage variables / environment variables `U`
   - outcome variables `Y`
   - desired region `M y <= d`
   - surrogate target weights `W`
   - RBF kernel bandwidth selection
   - kernel ridge weights `gamma` 与 `alpha`
   - `omega` action objective
   - bounded projected-gradient action optimization
   - no-action baseline evaluation
2. 新代码必须围绕现有 `AUFTask`：
   - `task.observed_variables` 对应 legacy `nodes_X`
   - `task.alterable_variables` 对应真正可干预、有 bounds 的 action variables
   - `task.outcome_variables` 对应 legacy `nodes_Y`
   - `task.candidate_alteration_sets` 对应 legacy `candidate_Rh_list`
   - `task.desired_region` 对应 legacy `(M, d)`
   - `task.variable_order` 用于恢复拓扑顺序
   - 非 action 的中间变量 `U` 优先从 `task.metadata["cme_environment_variables"]`
     读取；如果没有，则从 `data` / `task.variable_order` 中推断为既不在 `X`、
     action、`Y` 中的变量
3. `fit(...)` 应完成并缓存：
   - 训练数据矩阵 `X_hist`、`Y_hist`、每个候选 action 的历史矩阵
   - surrogate weights
   - bandwidths 或 bandwidth metadata
   - 必要的 kernel matrices 或可复用分解
4. `suggest(...)` 应：
   - 根据当前 observation 构造 `x_t`
   - 对每个 candidate `Rh` 计算 CME/KRR objective
   - 在 alteration bounds 内优化 action
   - 选择 objective 最优的 candidate
   - 返回可通过当前 `DecisionResult` 校验的结果对象
5. `evaluate(...)` 可以先实现轻量版本：
   - 对最近一次 `suggest` 的 action 做 empirical/surrogate success estimate；
   - 如果 experiment config 提供 true simulator，则由 config 的 `evaluate` 函数负责
     计算 true AUF success rate。
6. 结果必须 deterministic under fixed seed。PGD 的随机 restart fallback 必须走
   `np.random.default_rng(seed)` 或 method-local RNG，不要使用全局 `np.random`。
7. 诊断信息至少包含：
   - `selected_candidate`
   - `objective_value`
   - `solver_status`
   - `n_candidates`
   - `n_training_samples`
   - bandwidth / regularization / iteration 相关信息
8. 不要修改 `previous_works/` 下的任何文件。
9. 不要做 unrelated refactor。

建议实现结构：
- `src/rehearsal/metrics/cme.py`
  - `rbf_kernel`
  - `median_or_mean_bandwidth`
  - `desired_region_surrogate_weights`
- `src/rehearsal/optimizers/cme.py`
  - `CMEOptimizationResult`
  - `optimize_action_projected_gradient`
  - 一个候选 `Rh` 的 objective / gradient 计算
- `src/rehearsal/methods/cme.py`
  - `CMERehearsal`
  - legacy data dict / ndarray 到矩阵的转换
  - candidate enumeration
  - `fit` / `suggest` / `evaluate`
- `examples/cme/cme_toy_experiment.py`
  - 一个很小的 seeded batch config，用现有 `rehearsal.experiments.run` 跑通

测试要求：
添加 `tests/test_cme_rh.py`，至少覆盖：
- tiny synthetic AUF task 可以 `fit`、`suggest`、`evaluate`
- 输出 alteration 尊重 `task.alteration_domain`
- 固定 seed 下结果 deterministic
- surrogate weights 对 desired region 有正确方向性
- RBF kernel shape、对称性和对角线行为正确
- PGD optimizer 不越界，objective 非 NaN
- 有 `U` 中间变量和没有显式 `U` metadata 的 fallback 都能工作
- 多个 candidate 时能返回合法 selected candidate
- diagnostics 字段存在
- registry 可以通过 `"cme-rh"` 创建 method

如果新增 example config，再添加一个轻量 runner smoke test：
- `run_experiment_configs("examples/cme/cme_toy_experiment.py", seeds=(1,), method_name="cme-rh", ...)`
  返回 batch payload。

验证命令：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests
```

完成时请报告：
- 修改了哪些文件
- legacy CME code 中哪些部分已经被接入
- 哪些 legacy 实验脚本只作为参考或保留为 TODO/backlog
- pytest 结果

你不是唯一一个 agent。不要回滚或重写其他 agent 的修改；如果发现 framework API
已经变化，请适配现有接口，而不是重建一套接口。

本任务只实现 CME-Rh Conditional Mean Embeddings / non-parametric rehearsal
adapter，不实现 order-based rehearsal，不实现 ICML 2025 CARE，不实现 NeurIPS
2023、NeurIPS 2024、AAAI 2025、IJCAI 2025、ICLR 2026。不要扩大范围。

关键是：不要只写“帮我实现 CME”。要把输入旧代码、目标新 API、允许修改的文件、
测试验收、依赖边界和禁止事项全部写清楚。这样 worker 更容易产出能合并的代码，
而不是又生成一套新的研究脚本。
