# Rehearsal


**排演学习**由南京大学周志华教授提出，目标是识别服务于决策的**影响关系**。排演学习用于处理 **AUF** 任务：给定观测到的上下文 **X** 和预测出的非期望结果 **Y**（**Y** 位于预先指定的期望区域 *S* 之外），目标是确定一个决策，使 **Y** 朝 *S* 移动。参考资料见 [参考 PDF](https://www.lamda.nju.edu.cn/publication/fcs22_rehearsal.pdf)。

`rehearsal` 包为近期进展中迁移而来的方法提供统一接口：共享任务契约、结构模型接口、方法适配器、优化器、指标、影响关系度量、数据集，以及用于在统一 CLI 形式下比较排演学习方法的带随机种子的实验运行器。

## 已实现方法来源

下表覆盖当前已注册的 `rehearsal-run --method` 适配器和独立度量演示。形如 `--method ...` 的值是稳定的方法注册表名称；InP 是一个度量 API，提供独立示例 CLI，而不是 `RehearsalMethod` 适配器。

| Registry | Venue | Paper | Example configs | Setting |
| --- | --- | --- | --- | --- |
| `qwz23` | 2023 NeurIPS | 用于避免非期望未来的排演学习 | `examples/qwz23/` | 基于交互，显式图 - 线性 |
| `micns` | 2024 NeurIPS | 非平稳环境中以最小成本避免非期望未来 | `examples/micns/` | 无需交互，显式图 - 线性，非平稳 |
| `grad-rh` | 2025 AAAI | 带多变量变更的基于梯度的非线性排演学习 | `examples/grad_rh/` | 无需交互，显式图 - 非线性 |
| `care` | 2025 ICML | CARE condition 下排演学习中的最优决策 | `examples/care/` | 无需交互，显式图 - 线性，在 CARE condition 下最优 |
| `msr` | 2025 IJCAI | 通过序贯决策避免非期望未来 | `examples/msr/` | 无需交互，显式图 - 多阶段 |
| `mur` | 2025 NeurIPS | 通过二次规划重构降低方差的长期排演学习 | `examples/mur/` | 无需交互，显式图 - 线性，长期 |
| `olem-rh` | arXiv 2026 | 基于序结构的排演学习 | `examples/olem_rh/` | 无需交互，无显式图 - 基于序结构 |
| `cme-rh` | arXiv 2026 | 通过条件均值嵌入实现的非参数排演学习 | `examples/cme/` | 无需交互，无显式图 - 非参数 |
| InP measure demos | 2026 ICLR | 关于 AUF 中影响关系度量 | `examples/inp/` | 影响关系度量 |

请注意，这些方法的原始设定**并不完全相同**。

## Bermuda 示例

[Bermuda](http://lod.bco-dmo.org/id/dataset/720788) 是一个用于 AUF 任务的标准化连续 SEM：`Light`、`Temp` 和 `Sal` 是观测上下文变量；`DIC`、`TA`、`Omega`、`Chla` 和 `Nutrients_PC1` 是带边界的可变更变量；`NEC` 是结果变量；成功意味着在真实模拟器下，以高概率使 `NEC` 落入期望区间。

每个带随机种子的示例都会采样一个 Bermuda 上下文，从 `n_data=2000` 个观测样本中学习结构模型，选择带边界的变更，并用 `eval_samples=1000` 评估所选动作。观测到的 Bermuda 上下文在每个带随机种子的实验配置内部采样。不要通过 `--params` 传入观测变量；请使用 `--seeds` 保证采样观测可复现。

### 度量与方法 CLI 示例

请在仓库根目录运行以下命令。它们会在 `outputs/` 下生成已跟踪的 Bermuda 参考输出。

InP / ICLR 2026 度量示例：

```bash
env PYTHONPATH=src python examples/inp/bermuda_inp_example.py \
  --n-data 2000 \
  --num-samples 1500 \
  --n-bins 3 \
  --start-node TA \
  --output outputs/inp_bermuda_measures.json \
  --quiet
```

QWZ23 / NeurIPS 2023：

```bash
env PYTHONPATH=src python -m rehearsal.experiments.run examples/qwz23/bermuda_example.py \
  --method qwz23 \
  --seeds 3 \
  --params n_data=2000 \
  --eval-samples 1000 \
  --output outputs/qwz23_bermuda_seed3.json \
  --compact
```

CARE / ICML 2025：

```bash
env PYTHONPATH=src python -m rehearsal.experiments.run examples/care/care_bermuda_example.py \
  --method care \
  --seeds 3 \
  --params n_data=2000 \
  --eval-samples 1000 \
  --output outputs/care_bermuda_seed3.json \
  --compact
```

OLEM-Rh / arXiv 2026：

```bash
env PYTHONPATH=src python -m rehearsal.experiments.run examples/olem_rh/bermuda_example.py \
  --method olem-rh \
  --seeds 3 \
  --params n_data=2000 \
  --eval-samples 1000 \
  --output outputs/olem_rh_bermuda_seed3.json \
  --compact
```

### 排演学习结果

已跟踪的方法输出是使用 seed `3` 的单随机种子 Bermuda 参考结果。下表报告由各示例的真实模拟器测得的真实 AUF 概率。

| Method | Venue | Output | True AUF probability |
| --- | --- | --- | ---: |
| `qwz23` | 2023 NeurIPS | `outputs/qwz23_bermuda_seed3.json` | 0.833 |
| `micns` | 2024 NeurIPS | `outputs/micns_bermuda_seed3.json` | 0.837 |
| `grad-rh` | 2025 AAAI | `outputs/grad_rh_bermuda_seed3.json` | 0.827 |
| `care` | 2025 ICML | `outputs/care_bermuda_seed3.json` | 0.840 |
| `mur` (GMuR/FarMuR, same when horizon 0) | 2025 NeurIPS | `outputs/mur_bermuda_seed3.json` | 0.840 |
| `msr` | 2025 IJCAI | `outputs/msr_bermuda_seed3.json` | 0.830 |
| `cme-rh` | arXiv 2026 | `outputs/cme_bermuda_seed3.json` | 0.831 |
| `olem-rh` | arXiv 2026 | `outputs/olem_rh_bermuda_seed3.json` | 0.808 |

请注意，这些方法主要是在统一的 Bermuda 场景下提供可运行的并列实现参考。然而，它们的原始设定和输入要求并不相同，因为这些方法面向 AUF 决策问题的不同变体。因此，这些结果用于功能演示，而不是直接的性能比较。例如，`olem-rh` 只使用观测数据作为输入，而其他方法可能依赖额外的结构信息。

### 序学习与变量 InP

Bermuda 度量示例会从连续观测数据中学习变量序结构，将每个变量离散化为 `3` 个 bin 以执行递归 MEP / InP 计算，并对从所选 `start_node` 到 `NEC` 的有效路径上的每个可变更变量评估 InP。

已跟踪的 `outputs/inp_bermuda_measures.json` 运行使用 `n_data=2000`、`num_samples=1500`、`n_bins=3` 和 `start_node=TA`。学习到的序结构为 `Temp -> pHsw -> TA -> DIC -> CO2 -> Sal -> Light -> Omega -> NEC -> Nutrients_PC1 -> Chla`；在该序结构下，被评估变量的取值如下：

| Variable | InP | MEP-alter | MEP-observe |
| --- | ---: | ---: | ---: |
| `DIC` | 0.469 | 0.650 | 0.181 |
| `TA` | 0.322 | 0.982 | 0.659 |
| `Omega` | 0.186 | 0.190 | 0.004 |

偏序演示会选择一个兼容的 Bermuda 序结构，使得以 `TA` 为起点时 MEP 达到最大值 `0.980`；在该序结构下，`DIC`、`TA` 和 `Omega` 的 InP 值分别为 `0.473`、`0.346` 和 `0.174`。

## 项目结构

- `src/rehearsal/`：可安装的 Python 包。它包含共享任务契约、模型接口、方法适配器、度量 API、优化器、指标、数据集和实验运行器。
- `tests/`：面向该包的聚焦回归测试和契约测试。
- `examples/`：README 命令使用的可运行方法与度量示例。
- `docs/`：架构说明和方法迁移指南。
- `previous_works/`：只读历史代码、论文源文件、数据和 PDF，在将方法迁移到统一包时作为参考材料。
- `ExecPlan.md`、`OnGoing.md`、`code_idea.md`：项目计划和剩余迁移任务。
- `outputs/`：由 README 风格命令生成并被跟踪的示例实验结果 JSON 文件，用作可复现参考输出。

有意不纳入 Git 的文件包括 Python 字节码、pytest/cache 目录、`.DS_Store` 等操作系统元数据、本地 agent/editor 状态、打包/构建产物、LaTeX 辅助文件和本地运行时产物。

当前实现包含上方来源表中列出的度量 API 和方法适配器，同时保持共享模型、任务、优化器和实验运行器接口可在多篇论文之间复用。

## 包布局

- `rehearsal.core`：AUF 任务对象、期望区域、变更域、结果契约和校验。
- `rehearsal.models`：结构学习模型。`LinearGaussianSRM` 和 `LinearGaussianSRMLearner` 是 CARE 以及未来 NeurIPS 2023 / NeurIPS 2024 适配器的共享组件。
- `rehearsal.optimizers`：在拟合后的结构模型上执行的排演阶段优化器。
- `rehearsal.methods`：轻量方法适配器，对外暴露 `fit`、`suggest` 和 `evaluate`。
- `rehearsal.measures`：InP、MEP、ACE、CACE 以及用于评估拟合后排演学习模型影响关系属性的偏序工具。
- `rehearsal.datasets`：可复用的数据集和 SEM 工厂，包括跨方法共享的通用 Bermuda 与 Manage 数据集模块。
- `rehearsal.experiments`：用于带随机种子的批量实验的命令行运行器。

## 安装与打包演示

包发布后，可使用以下命令安装基础包：

```bash
python -m pip install rehearsal
```

基础安装包含基于 NumPy 的核心 API、方法适配器、实验运行器和一个已安装的玩具演示。Bermuda `.mat` 加载需要 SciPy，并通过一个可选 extra 暴露：

```bash
python -m pip install "rehearsal[bermuda]"
```

QWZ23 使用采样式多变量最大化。它可以使用 NumPy 随机搜索 fallback 运行，也可以通过以下命令安装 SciPy 以使用首选的 MILP 优化器：

```bash
python -m pip install "rehearsal[qwz23]"
```

如果要从本地 checkout 发布，请安装 release tools extra：

```bash
python -m pip install "rehearsal[publish]"
```

该包附带一个自包含的 smoke demo，不需要仓库的 `examples/` 目录：

```bash
rehearsal-demo \
  --seed 3 \
  --n-samples 40 \
  --eval-samples 6 \
  --max-iters 5 \
  --output outputs/care_demo_from_package.json \
  --compact
```

同一个 demo 也可以从 Python 中导入并运行：

```python
from rehearsal.experiments.demo import run_demo

result = run_demo(seed=3, n_samples=40, eval_samples=6, max_iters=5)
print(result["name"], result["method"], result["n_runs"])
print(result["runs"][0]["evaluation"])
```

## 运行器契约

通用运行器只有一种执行形式：带随机种子的批量运行。不存在单独的单随机种子输出模式。如果只需要一个随机种子，请传入只含一个元素的 seed 列表：

```text
--seeds 3
```

输出始终包含 `runs` 和 `summary`。即使只有一个随机种子，`summary` 仍包含 `mean`、`std`、`min` 和 `max`；标准差为 `0.0`。

实验配置应定义：

```python
def build_experiment(params, seed):
    # seed is supplied only by rehearsal-run --seeds.
    # Build task, generated training data, and the observed individual here.
    return {
        "name": "my_experiment",
        "task": task,
        "data": data_for_this_seed,
        "observation": observation_for_this_seed,
        "method_params": {"pgd_steps": 60},
        "default_eval_samples": 500,
        "evaluate": evaluate_true_auf,
        "metadata": {"n_samples": n_samples},
    }
```

不要通过 `--params`、`--method-params` 或配置返回的 `method_params` 传递 `seed`。seed 列表是运行随机性的唯一来源。运行器会将每个 seed 传给配置和方法构造器。

`data` 和 `observation` 不应被视作全局常量。在已提供的演示中，`n_samples=100` 表示：对每个 seed，在 `build_experiment(params, seed)` 内生成 100 个训练样本，然后将生成的字典作为 `data` 返回。观测到的个体也在同一个带 seed 的工厂函数中采样。

## CLI 参数

仅使用以下形式：

| Argument | Meaning |
| --- | --- |
| `--seeds 3,4,5` | 必需。精确指定运行 seed。`--seeds 3` 是一个单 seed 批量运行。 |
| `--method NAME` | 方法注册表名称。当前已注册：`care`、`cme-rh`、`grad-rh`、`micns`、`msr`、`mur`、`olem-rh`、`qwz23`。 |
| `--params KEY=VALUE` | 传给 `build_experiment(params, seed)` 的实验配置参数。 |
| `--method-params KEY=VALUE` | 方法构造器参数。不要在这里放 `seed`。 |
| `--fit-params KEY=VALUE` | 传给 `method.fit(...)` 的额外选项；通常很少需要。 |
| `--eval-samples N` | 配置评估器使用的真实 AUF Monte Carlo 样本数。 |
| `--output path.json` | 可选 JSON 输出路径。 |
| `--compact` | 打印紧凑 JSON。 |

提供 `--output` 时，完整 JSON payload 会写入该文件，运行器会打印一行简短的完成信息，例如 `wrote outputs/cme_bermuda_seed3.json (n_runs=1, method=cme-rh)`。

已移除的单数别名 `--param`、`--method-param` 和 `--fit-param` 会被有意拒绝。

## 输出形状

单 seed 运行仍会返回一个批量结果：

```json
{
  "name": "cme_bermuda",
  "method": "cme-rh",
  "seeds": [3],
  "n_runs": 1,
  "runs": [
    {
      "seed": 3,
      "observation": {"Light": 0.01, "Temp": -0.02, "Sal": 0.03},
      "structural_learning": {
        "runtime_seconds": 0.004
      },
      "decision": {
        "alterations": {"DIC": 0.2, "TA": 0.1},
        "estimated_success_probability": 0.8,
        "cost": 0.3,
        "runtime_seconds": 0.001
      },
      "evaluation": {
        "true_auf_success_rate": 0.82,
        "no_action_true_auf_success_rate": 0.12,
        "eval_samples": 1000
      }
    }
  ],
  "summary": {
    "structural_learning.runtime_seconds": {
      "mean": 0.004,
      "std": 0.0,
      "min": 0.004,
      "max": 0.004
    },
    "decision.runtime_seconds": {
      "mean": 0.001,
      "std": 0.0,
      "min": 0.001,
      "max": 0.001
    },
    "evaluation.true_auf_success_rate": {
      "mean": 0.82,
      "std": 0.0,
      "min": 0.82,
      "max": 0.82
    },
    "evaluation.no_action_true_auf_success_rate": {
      "mean": 0.12,
      "std": 0.0,
      "min": 0.12,
      "max": 0.12
    }
  }
}
```

批量 `summary` 有意只包含真实 AUF Monte Carlo 成功指标，以及结构学习和决策阶段运行时间。它不会汇总决策成本、方法内部估计值或 `eval_samples`。

已提供的示例报告以下逐次运行评估字段：

- `true_auf_success_rate`：在实验配置提供的真实数据生成过程下，采用建议变更后的成功率。
- `no_action_true_auf_success_rate`：同一个观测样本在同一真实数据生成过程下、不进行变更时的成功率。
- `eval_samples`：来自 `--eval-samples` 或示例 `default_eval_samples` 的 Monte Carlo 数量。
- `structural_learning.runtime_seconds`：针对该 seed 的训练数据执行 `method.fit(...)` 所花费的逐次运行 wall-clock 时间。
- `decision.runtime_seconds`：方法的 `suggest(...)` 决策步骤所花费的逐次运行 wall-clock 时间。它不包含结构拟合或真实 AUF Monte Carlo 评估时间。

## 方法注册表

`--method ...` 由 `rehearsal.methods.registry` 解析。注册表允许运行器通过稳定的 CLI 名称实例化方法，而不需要每个实验配置自行导入并构造适配器。

```python
"grad-rh": GradRhRehearsal
"care": ICML2025CARERehearsal
"micns": MICNSRehearsal
"msr": MSRRehearsal
"mur": MURRehearsal
"olem-rh": OLEMRhRehearsal
"qwz23": QWZ23Rehearsal
"cme-rh": CMERehearsal
```

除上方列出的稳定注册表名称外，没有旧版方法名称别名。

## 协作指南

本仓库是一个研究代码迁移项目。请保持改动小而可审查，并与共享的 `src/rehearsal/` 包接口保持一致，而不是新增一次性实验脚本。

### Commit 前缀

每个 commit subject 开头都应使用简短的方括号前缀：

| Prefix | Use for |
| --- | --- |
| `[ENH]` | 新功能、方法适配器、实验运行器或受支持能力。 |
| `[FIX]` | bug 修复、数值修正、CLI 契约修复或损坏测试修复。 |
| `[DOC]` | README、架构说明、方法迁移说明、注释，或不改变行为的示例。 |
| `[TST]` | 新增或更新测试、fixtures、smoke checks 或回归覆盖。 |
| `[REF]` | 保持行为不变，同时改善结构或可读性的重构。 |
| `[EXP]` | 可复现实验配置、结果 JSON 文件或 benchmark 输出更新。 |
| `[DATA]` | 数据集加载器、小型已跟踪数据 fixtures 或元数据变更。 |
| `[DEP]` | 依赖、打包或环境变更。生产依赖需要事先确认。 |
| `[CHORE]` | 仓库维护、仅格式化改动，或无面向用户行为变化的清理。 |

Commit subject 应使用祈使句并保持具体，例如 `[ENH] Add CME Bermuda batch runner` 或 `[FIX] Preserve one-seed batch summary shape`。

### 分支与 Review

- 使用类似 `enh/cme-runner`、`fix/seed-summary`、`doc/collaboration-guidelines` 或 `exp/care-bermuda-smoke` 的分支名。
- 每个 pull request 应聚焦于一个方法、运行器契约、数据集或文档主题。
- 对复杂功能或重要重构，应在实现前编写或更新 ExecPlan，并在工作变化时保持计划同步。
- 将 `previous_works/` 视为只读历史参考材料。应将行为迁移到 `src/rehearsal/`，在 `tests/` 中添加聚焦测试，并在 `docs/` 下记录方法特定说明。

### 测试与验证

- 修改 Python 包、示例或测试后，运行：

  ```bash
  env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests
  ```

- 修改 JavaScript 文件后，运行 `npm test`。
- 修改 CLI 行为时，在 `tests/test_experiment_runner.py` 中包含或更新回归测试。
- 修改数值方法时，优先使用固定 seed 的确定性 toy tests，再添加更大的实验输出。
- 保持已跟踪的 `outputs/` 文件可通过 README 风格命令复现，并避免提交本地缓存、临时产物或探索性产物。

### 依赖与数据

- 保持运行时依赖精简。添加任何新的生产依赖前都要先确认。
- 对重量级研究依赖优先使用可选导入，并保持 CPU smoke tests 在不下载历史数据的情况下可运行。
- 安装 JavaScript 依赖时优先使用 `pnpm`。
- 只跟踪小型且必要的数据 fixtures。大型生成产物应留在 Git 之外，除非它们被明确接受为可复现参考。

## 验证

运行 `Testing And Verification` 中列出的 pytest 命令。
