# V21E3R1 V9R2 Trace and Replay Specification

状态：`ENGINEERING_ONLY / PRE_DEVELOPMENT_HOLD`

本规范说明 V9R2 当前实现能验证什么，以及明确不能验证什么。它不授权 full development matrix 或任何后续科学阶段。

## 1. 证据层级

| 层级 | 输入 | 当前结论 | 独立性边界 |
| --- | --- | --- | --- |
| Trace verifier | `trace.sqlite3`、外部 `terminal.json`、精确 problem artifact | objective/solution、archive、B/A/S/T、确定性 screen/Lyapunov/population policy replay | 使用项目中的独立 verifier 路径，但不是完整算法的独立实现 |
| Read-only diagnostic | `trace.sqlite3`、外部绑定的 `terminal.json` | ledger/chain/terminal 算术、all-evaluated HV、operator productivity、Lyapunov durable-state arithmetic | 不接收 problem input，因此 `objective_function_replay=NOT_IMPLEMENTED_NO_PROBLEM_INPUT` |
| Branch re-execution | trace、problem artifact、完整 `mo_nco/**/*.py` manifest | 相同随机程序、相同源码身份的 full branch re-execution | `implementation_independence=false`、`scientific_independence=false` |
| Independent algorithm replay | 独立 producer/implementation/custody | 当前不存在 | `full_algorithm_decision_replay=NOT_IMPLEMENTED` |

前三层全部 PASS 也不能替代第四层，不能生成 selection 或 scientific promotion。

## 2. 必需外部绑定

成功 trace 的 verifier 与 diagnostic 都必须读取 detached `terminal.json`。runner 计算该文件的 SHA-256，并将路径与预期 hash 显式传入两个消费者。SQLite 内嵌 terminal row 必须与 detached bytes 精确一致；只验证数据库内部 self-hash 不足以通过。

必须验证：

- SQLite `integrity_check=ok`，diagnostic 使用 read-only URI 与 `query_only=ON`；
- run context digest；
- attempt、evaluation、decision 三条连续语义 hash chain；
- terminal 中三条末端 hash、记录数、资源账本和 run status；
- detached receipt file SHA-256、receipt payload SHA-256 与 SQLite terminal row 的精确绑定；
- 不允许未解析 decision、缺失 evaluation 前缀或 cache hit 指向未来/未知 evaluation。

## 3. Candidate-screen witness v2

schema：`v21e3r1_information_time_candidate_screen_v2`。

每个 membership check 必须持久化 exact solution、solution SHA-256、候选 rank、operator 和 `seen_before_attempt`。verifier 使用 problem canonicalization 与此前已 charged 的 durable solution hashes 重新推导 membership 序列，并验证提交的 proposal 正是 deterministic first-unseen candidate；若全部 seen，则验证 exhausted 语义。不能只相信 witness 内的 seen 标志或 selected hash。

## 4. Archive-compensated replacement witness v2

schema：`v21e3r1_archive_compensated_replacement_v2`；policy：`archive_compensated_information_lyapunov_development_v1`。

verifier 必须从 durable typed-population state、reference directions、objective box、candidate objectives 和 all-evaluated archive 独立推导：

- considered target neighborhood；
- preselected empty targets；
- existing targets 的 finite scalar deltas；
- finite selection capacity；
- normalized HV before/after/gain；
- deterministic nonpositive-then-credit-bounded positive subset；
- final replacement targets、paid-worsening count、archive credit 和 composite potential change。

任何 witness 字段即使被攻击者连同 decision chain、terminal row 和 detached receipt 一起重新封链，只要与上述 durable state 不一致，都必须 fail closed。

## 5. Population-policy replay

初始化 decision 必须只填充本次 type。search decision 的 target neighborhood 由冻结 reference directions 和 neighborhood size 重建；nonworse 与 Lyapunov 两种 replacement policy 分别独立推导 targets。每次 decision 的 `accepted_into_population`、replacement count 和 target ids 必须与推导结果一致，随后才更新 verifier 的 typed-population state。

这只重放已经持久化 proposal 之后的确定性 policy decision；它不重建 RNG、operator choice 或 proposal generation，因此总字段继续为 `full_algorithm_decision_replay=NOT_IMPLEMENTED`。

## 6. 完整源码闭包

branch replay manifest 的 scope 是执行环境内全部 regular `mo_nco/**/*.py`。验证时 live set 与 manifest set 必须完全相等；缺文件、多文件、重复/case-colliding path、非 canonical path、size/hash drift 均拒绝。manifest 只绑定同实现重放，不能把 `source_closure_verified=true` 改写为第三方独立复现。

## 7. Runner 最低验收

每个臂完成后，runner 必须逐字段要求：

- trace overall status、objective/archive replay、B/A/S/T replay；
- population-policy replay PASS，screen/Lyapunov replay 状态与 arm 启用状态一致，witness counts 与 durable accounting 一致；
- diagnostic schema、semantic chains、terminal/detached binding、HV metrics 的 finite/range 约束和 arm-specific Lyapunov arithmetic；
- diagnostic 保留两个明确的 `NOT_IMPLEMENTED` 边界；
- branch status 为 `PASS_SAME_IMPLEMENTATION_FULL_BRANCH_REEXECUTION`，所有 semantic checks true，完整 source closure true，三个 independence 字段 false。

任一子字段缺失、类型漂移、NaN/Infinity、状态降级或 count 不一致，都不得生成顶层 `SUCCESS_ENGINEERING_ONLY` summary。

## 8. 授权边界

所有 receipt 必须保留：

```text
scientific_claim_authorized=false
implementation_independence=false
scientific_independence=false
third_party_independence=false
selection_authorized=false
confirmation_authorized=false
formal_authorized=false
ijoc_submission_authorized=false
```

当前 canonical protocol 还缺 environment lock、independent algorithm-decision replay、metric/reference manifest、rights ledger 和 strong-baseline registry，所以 full matrix 必须在 materialization 前停止。
