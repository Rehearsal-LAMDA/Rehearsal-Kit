  你可以直接用下面这个 prompt 作为 ICML 2025 worker agent 的任务说明：

  你是 Rehearsal 项目的实现型 worker agent，负责把 ICML 2025 rehearsal learning 方法接入新的统一 Python package。

  项目根目录：当前 Rehearsal 仓库根目录

  背景：
  - 旧代码位于 previous_works/05-ICML 2025/code/
  - 重点参考：
    - previous_works/05-ICML 2025/code/Rh_Solver.py
    - previous_works/05-ICML 2025/code/utils.py
    - previous_works/05-ICML 2025/code/main_syn.py
    - previous_works/05-ICML 2025/code/main_bermuda.py
  - 旧代码只能作为只读参考，不要直接把脚本式全局代码整段搬进 package。
  - previous_works/01-FCS 2022 和 previous_works/07-NeurIPS 2025 不在本轮实现范围内。

  你的 ownership：
  - src/rehearsal/methods/care.py
  - src/rehearsal/metrics/care.py 或已有 metrics 模块中的相关函数
  - tests/test_icml2025_care.py
  - 如有必要，可以对 src/rehearsal/core/ 或 src/rehearsal/models/ 做很小的兼容性补充，但不要重构 framework。

  目标：
  实现 ICML 2025 CARE / circular-region rehearsal 方法的 package adapter，使其通过统一接口工作：

  - fit(data, task, config=None)
  - suggest(observation, task)
  - evaluate(task, n_samples)

  其中 suggest 必须返回 rehearsal.core.DecisionResult，包含：
  - alterations: dict[str, float] 或等价结构
  - estimated_success_probability: float，必须在 [0, 1]
  - cost: float
  - diagnostics: dict
  - runtime_seconds: float

  请优先实现一个 CPU 可运行、依赖轻量的版本：
  - 使用 numpy / scipy，如果 scipy 已存在可以使用。
  - 不要添加新的 production dependency。
  - 如果旧代码依赖复杂数值优化，先实现可测试的核心 CARE solver，再把更重的 solver 留成清晰 TODO/backlog。
  - 不要引入 torch、cvxopt、rpy2、FrEIA 等新依赖。

  实现要求：
  1. 阅读 legacy ICML 2025 solver，抽取核心概念：
     - linear SEM / SRM 参数估计
     - candidate alteration set Rh
     - alteration bounds
     - desired region M y <= d
     - CARE success evaluation
     - circular-region special case，如果可轻量实现则实现，否则设计接口并给出 TODO
  2. 新代码必须围绕 rehearsal.core.AUFTask 和 rehearsal.core.DecisionResult。
  3. 输出 alterations 必须尊重 task 的 alteration bounds。
  4. 结果必须 deterministic under fixed seed。
  5. 诊断信息至少包含：
     - selected_candidate
     - objective_value 或 estimated_care_success
     - solver_status
     - n_candidates
  6. 不要修改 previous_works/ 下的任何文件。
  7. 不要做 unrelated refactor。

  测试要求：
  添加 tests/test_icml2025_care.py，至少覆盖：
  - bounded alteration output
  - estimated_success_probability in [0, 1]
  - deterministic result under fixed seed
  - works on tiny synthetic AUF task
  - circular-region helper or fallback behavior
  - solver diagnostics fields exist

  验证命令：
  - pytest

  完成时请报告：
  - 修改了哪些文件
  - ICML 2025 legacy code 中哪些部分已经被接入
  - 哪些部分保留为 TODO/backlog
  - pytest 结果

  如果你要让多个 agent 并行，最好再加一句：

  你不是唯一一个 agent。不要回滚或重写其他 agent 的修改；如果发现 framework API 已经变化，请适配现有接口，而不是重建一套接口。

  更强一点的版本可以把交付边界写死：

  本任务只实现 ICML 2025 adapter，不实现 NeurIPS 2023、NeurIPS 2024、AAAI 2025、IJCAI 2025、ICLR 2026。不要扩大范围。

  关键是：不要只写“帮我实现 ICML 2025”。要把 输入旧代码、目标新 API、允许修改的文件、测试验收、禁止事项 全部写清楚。这样 agent 更容易产出能合并的代码，而不是又生成一套新的研究脚本。
