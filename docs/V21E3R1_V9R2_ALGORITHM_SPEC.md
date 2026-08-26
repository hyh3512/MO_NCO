# V21E3R1 V9R2 Engineering Algorithm Specification

状态：`FROZEN_FOR_EXPOSED_SINGLE_CASE_ENGINEERING_SMOKE_ONLY`

发行身份：`mo-nco==0.21.3.13`

canonical pre-development protocol：`mo_nco/specs/V21E3R1_V9R2_PREDEVELOPMENT_PROTOCOL.json`

```text
protocol file SHA-256     = 112bbe405c64fbe598275f27a0ae7262a4e68e85469052e559de722066ef15ad
protocol payload SHA-256  = b051bdeb20b949e3b09ab30af6517259a222fe05b322eb450039e713f2148db2
resource contract SHA-256 = d6525669b723dd104004949fad202263060d211c9477e8d2fec9ea6657126479
```

本文件冻结当前可执行工程对象的语义，但不冻结或授权 full-development 参数值。full matrix 的环境、B/A/S/T caps、metric/reference、baseline 和 rights 仍缺失，因此必须使用新的 protocol identity 才能开启。

## 1. 问题与阶段范围

- families：MOKP、MOTSP；目标维度为 2。
- evidence partition：仅 canonical exposed-development manifest 中已有 case bytes。
- phase：严格为 `development`；calibration、selection、confirmation 和 formal 输入均 fail closed。
- candidate：`C0`；reference directions、seed 和所有 cap 必须进入 run-context hash。
- all-evaluated archive：每次 first true evaluation 都在 population decision 前进入 append-only nondominated archive；population rejection 不删除已经付费得到的 objective information。

## 2. 冻结四臂菜单

| Arm | Screening | Replacement |
| --- | --- | --- |
| LEGACY | `disabled_v1` | `bounded_reference_neighborhood_nonworse_replacement_v1` |
| SCREEN | `bounded_cache_aware_structural_screen_development_v1` | `bounded_reference_neighborhood_nonworse_replacement_v1` |
| LYAP | `disabled_v1` | `archive_compensated_information_lyapunov_development_v1` |
| BOTH | `bounded_cache_aware_structural_screen_development_v1` | `archive_compensated_information_lyapunov_development_v1` |

同一 single-case runner 必须按 LEGACY/SCREEN/LYAP/BOTH 全部运行，不能只运行或报告胜出的臂。

## 3. 多资源合同

一次 invocation 显式给出：

- `B = charged_evaluations`：first true objective evaluations；positive exact built-in int。
- `A = attempt_cap`：提交给 durable attempt ledger 的总 attempts；positive exact built-in int 且 `A >= B`。
- `S = structural_screening_cap`：`structural_candidate_generations + cache_membership_probes`；nonnegative exact built-in int。
- `T = wall_time_cap_seconds`：当前 Python process 的 `time.perf_counter()` monotonic elapsed time；finite positive exact built-in int/float。

超过 A/S/T 时必须先提交持久化 `V9_RESOURCE_CAP_EXHAUSTED` FAILURE terminal receipt，再抛出异常。B/A/S/T 不等于完整资源证明；process-tree peak RSS、trace bytes 和 replay time 仍须在 target-scale receipt 中另行报告。

## 4. Screening 语义

SCREEN/BOTH 对冻结的 deterministic structural generator 产生的 bounded prefix 逐项执行 exact cache membership query：

1. 每个 candidate generation 与 membership probe 先计入 S；
2. canonical solution、hash、rank、operator 和 durable `seen_before_attempt` 写入 witness v2；
3. 提交首个 unseen candidate；
4. 若 bounded prefix 全部 seen，写 exhausted witness 并进入原 retry/fallback 路径；
5. screening 本身不得调用 objective，也不得读取未来状态。

MOKP 使用 deterministic feasible add/drop/swap 序列；MOTSP 使用 bounded deterministic 2-opt alternative sequence。首个 unseen 只证明对已检查 prefix 的 cache novelty，不证明 quality 改善。

## 5. Replacement 语义

LEGACY/SCREEN 在 reference-neighborhood 中仅替换 empty target 或 scalar nonworse target。

LYAP/BOTH 对当前 candidate 重建 all-evaluated normalized 2-D HV gain，先无条件选择 empty targets，再对已有 targets 计算 finite scalar deltas。deterministic subset rule 先按 `(delta,target_id)` 选择 nonpositive deltas，再在 `lambda * normalized_hv_gain` credit 内选择 positive deltas，并受剩余 target capacity 限制。

每个 decision 持久化 witness v2；positive worsening sum 不得超过 archive credit，composite potential change 不得超过数值 tolerance。该有限事件不变量不推出 convergence、superiority 或 lambda 最优。

## 6. Durable ledger 与完成条件

- attempts、first evaluations 和 decisions 使用连续 1-based index 与各自语义 hash chain。
- exact cache hit 必须指向更早的已完成 evaluation 和同一 canonical solution/hash。
- SUCCESS 前必须没有 unresolved decision，WAL 必须封口，并生成与 SQLite terminal row 逐字节一致的 detached `terminal.json`。
- 每臂必须通过 problem-backed trace verifier、external-terminal-bound read-only diagnostic 和完整源码闭包的 same-implementation branch re-execution。
- 顶层 summary 仅在四臂全部通过后以 exclusive create 写入，并包含 self hash；不得覆盖既有目录。

详细 witness/replay 合同见 `docs/V21E3R1_V9R2_TRACE_REPLAY_SPEC.md`。

## 7. 明确未冻结的 full-development 对象

当前没有冻结 full-matrix B/A/S/T 数值、lambda menu、case×seed matrix、simultaneous promotion thresholds、metric/reference manifest、environment lock、strong-native baseline registry、rights ledger 或 independent algorithm implementation。因此：

```text
FULL_DEVELOPMENT_MATRIX_AUTHORIZED=false
SCIENTIFIC_DEVELOPMENT_CLAIMS_AUTHORIZED=false
AUTHORIZED_NEXT_PHASE=NONE
```

不能通过修改本文件、当前 JSON 的 false 字段或 CLI 参数来产生授权；任何未来 full-development 对象必须采用新的 canonical protocol identity 并重新审计。
