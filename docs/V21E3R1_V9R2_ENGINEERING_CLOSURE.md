# V21E3R1 V9R2 工程闭合裁决

日期：2026-08-24

## 总裁决

```text
V8 exact108 history                         = PRESERVED / UNCHANGED
V9R2 scoped engineering code closure        = PASS
V9R2 targeted regression                    = PASS (230/230)
V9R2 reproducible wheel                     = PASS (two byte-identical builds)
Installed exposed-case engineering smoke    = PASS (MOKP + MOTSP, 4/4 arms each)
Full development matrix                     = NOT AUTHORIZED / NOT STARTED
Development study readiness                 = PRE_DEVELOPMENT_HOLD
Selection / confirmation / formal study     = PROHIBITED
IJOC submission                             = HOLD / NO-SUBMIT
```

V9R2 的最终发行身份为 `mo-nco==0.21.3.13`。本裁决只闭合当前可授权的工程实现、打包和有限回归，不把代码通过、同实现重放或 B=8 smoke 解释为科学有效性、独立复现或候选晋级。`.11/_001` 和 `.12/_002` 在攻击性复核中被否决并保留为负证据；它们不是可交付版本。

## 已闭合的代码缺口

1. 独立诊断不再只信任 SQLite 内部字段。它要求外部绑定的 detached `terminal.json`，验证 attempt/evaluation/decision 三条语义链、terminal 链和精确文件绑定，并重建逐 evaluation 左连续 HV AUC、初始化后增量 HV 和 Lyapunov 见证算术。
2. branch replay 的 source manifest 现在覆盖执行环境中完整的 `mo_nco/**/*.py` 集合；缺文件、多文件、重复路径或任一 hash 漂移均 fail closed。四个臂都执行同实现 full branch re-execution，而不是只验证 BOTH。
3. screen witness 与 Lyapunov witness 升级为可重放 schema。trace verifier 从持久化 solution/objective/population 重新推导 first-unseen screening、确定性 target subset 和 population replacement 决策，不再只核验 witness 自洽。
4. 新增 canonical V9R2 pre-development protocol，固定四臂菜单、暴露 development 分区、B/A/S/T 语义和所有后续阶段的禁止状态。runner 启动前必须加载、hash 绑定并验证该 protocol。
5. 新增机器可执行 readiness gate。当前 protocol 下它只会生成 `PRE_DEVELOPMENT_HOLD` 收据并以退出码 2 停止；不会 materialize development rows、执行 simultaneous inference 或输出 promotion PASS。
6. 四臂 runner 把 protocol、resource caps、外部 terminal hash、只读诊断、完整源码闭包和每臂 branch replay 全部绑定进 self-hashed summary。公开 V9R2 调用名为 `run_v9r2_development_case`，V9R1 名称仅作为兼容别名保留。
7. wheel 现在包含 `mo_nco/specs/*.json`，版本和 CLI 身份统一为 `0.21.3.13`；新增 `mo-nco-v21e3r1-v9-gate`。

## 当前仍然明确未实现或未满足的科学前置项

- `full_algorithm_decision_replay` 仍明确标记为 `NOT_IMPLEMENTED`；现有 branch replay 是相同 Python 实现的重执行，`implementation_independence=false`、`scientific_independence=false`。
- 当前 protocol 机器列出的 10 项 full-matrix artifact 均未绑定到新的授权 identity：current-source test receipt、environment lock、full-development algorithm spec、full source freeze、独立算法决策重放、metric/reference manifest、rights ledger、strong baseline registry、target-scale resource-capacity receipt 和 trace/replay spec。部分工程版本已在本交付中生成，但不自动满足科学冻结/独立验收。
- full-development B/A/S/T caps 尚未冻结；当前参数只用于单 case、B=8 的非科学工程 smoke。
- 未执行 full development matrix，也未产生 CI95 或 W/T/L；未 materialize selection、confirmation 或 formal cases。
- 本轮没有声称全仓 green。历史 V8 frozen recovery 会按设计检测新源码 drift；不得修改历史冻结 manifest 来消除该信号。

因此唯一允许的状态仍是：

```text
AUTHORIZED_NEXT_PHASE = NONE
STOP_AFTER_EXPOSED_DEVELOPMENT_ENGINEERING_SMOKE
FULL_DEVELOPMENT_MATRIX = PROHIBITED_UNDER_CURRENT_PROTOCOL_IDENTITY
SELECTION = PROHIBITED
CONFIRMATION = PROHIBITED
FORMAL_STUDY = PROHIBITED
IJOC = HOLD_NO_SUBMIT
```

## 工程证据

- Python 3.11 targeted suite：`230 passed in 34.05s`，JUnit XML 与完整 test-file manifest 由发行收据 hash 绑定。
- `compileall`：当前源码树和 clean-installed wheel 均通过。
- Python 3.13.12 / setuptools 80.10.2 / pip 26.0.1，`SOURCE_DATE_EPOCH=1700000000`：两个 wheel 均为 877263 bytes，逐字节相同，SHA-256 `589dc10657fd14c65008da6da8bc1111d24fe05e866d7fba8200ff031f50df6e`。
- 仓库外 clean venv：import 来自 `site-packages`，版本、package-data protocol、三个 V9 CLI、`pip check` 与 installed compileall 均通过。
- installed MOKP/MOTSP exposed-development smoke：每个 case 的 LEGACY/SCREEN/LYAP/BOTH 均通过 objective/archive/resource replay、policy witness replay、外部 terminal 绑定、完整 source closure 和 same-implementation branch re-execution；所有后续授权字段均为 false。
- installed readiness gate：预期退出码 2，`development_rows_materialized=0`，`full_development_matrix_authorized=false`。

可执行命令见 `docs/V21E3R1_V9R2_RUNBOOK.md`；机器收据和不可变 source bundle 位于 `artifacts/v21e3r1_v9r2_engineering_closure_20260824_003/`。
