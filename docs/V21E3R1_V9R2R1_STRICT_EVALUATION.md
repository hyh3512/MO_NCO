# V21E3R1 V9R2R1 严格评估与工程维护裁决

日期：2026-08-25

身份：`mo-nco==0.21.3.14` / `V21E3R1_V9R2R1`

## 1. 当前实验的证据结论

V9R2 已完成的定向测试、双 wheel/source 复建、clean install、两 family
四臂 B=8 smoke、SQLite/WAL/资源/HV/同实现 branch replay，以及最终全仓
JUnit 在其声明范围内自洽。最终全仓不是绿色：8 个失败均是冻结 V8 流程
检测到 `Frozen diagnostic source manifest drifted`，该信号必须保留。

8 臂合计为 2 个已暴露 development cases、1 seed/case、64 charged
evaluations、64 physical calls、80 attempts、16 cache hits、40 candidate
screens、1,236 structural candidate generations 和 1,276 structural work。
这些结果只证明有限工程执行、计账和相同实现重放，不证明算法有效性。

MOKP 上 BOTH/SCREEN 的输出相同，LEGACY/LYAP 的输出相同；两组最终
normalized HV 只差约 `1.1948588e-6`。MOTSP 四臂的最终 HV 和 AUC 完全
相同。单 case、单 seed、B=8 不能支持 Lyapunov 或 screening 的总体效果、
算法优越性、泛化或 SOTA 结论。

## 2. 发现的工程身份缺口

`0.21.3.13` wheel 两次复建逐字节一致，但在 wheel 冻结之后修复的
`run_v21e3r1_same_implementation_branch_replay_coverage.py` 不属于该 wheel，
也不在原 192 文件 source ZIP 中。原 source ZIP 对它所声明的 scoped closure
是可复建的，却不能称为“修复后完整源码冻结”。

V9R2R1 因此只做三项维护：

1. 使用 lazy package exports 消除 gate/diagnostic 的 `python -m` 预导入
   RuntimeWarning，同时保留原 root-level import API；
2. 新 source-freeze candidate 纳入 live `mo_nco` 源码、package specs、V9
   回归、修复后的 branch-coverage writer、diagnostic runner、相关测试和文档，
   并增加 strict canonical manifest、deterministic ZIP 与 live-tree verifier。
3. 新增只读 full-suite environment preflight，精确绑定 Python/pymoo/moocore
   身份并实际导入 pymoo backend；native PYD/DLL 被系统策略拒绝时生成 self-hashed
   HOLD 收据和 exit 2，而不是把不可用强基线静默 skip 或等 12 分钟后才混入
   全仓失败。

算法、四臂菜单、V9R2 canonical HOLD protocol、case partition 和科学门禁均未
改变。新 source bundle 明确写入
`full_source_freeze_requirement_satisfied=false`，因为它尚未被新的有权 protocol
接受。

## 3. 十项 pre-development artifact 的严格状态

当前 10 项必须继续是 `0/10 satisfied`。其中 source/test/environment 三项可在
最终源码身份上重新生成候选；metric/reference、full algorithm spec 和 trace
spec 需要新的前瞻性科学冻结；independent replay、rights、strong baselines 需要
外部/治理证据；target-scale capacity 必须在先冻结目标规模后实际测量。不能把
文件存在或 SHA-256 命中等同于 semantic acceptance 或 protocol acceptance。

现有算法规范明确没有冻结 full-development B/A/S/T、case×seed、lambda menu、
simultaneous inference、promotion thresholds 和 stop rules，所以不能通过重命名
或补哈希把它变成 full-development algorithm specification。

## 4. 裁决

V9R2R1 最终扩展定向集为 `271 passed in 94.07s`，其中包含 environment
preflight 的五项 fail-closed 回归。双 wheel 为 877,755 bytes、逐字节相同，
SHA-256 `b8813e632888eb80c65f165f90e622c2973623c4a2d814218111bcbd24165b27`。
仓库外 clean install、`pip check`、compileall、无 RuntimeWarning 的 gate/diagnostic
CLI、canonical gate exit 2，以及 MOKP/MOTSP 共 8 臂 installed B=8 smoke 均通过。

第一次 V9R2R1 全仓快照为
`1335 passed, 4 skipped, 265 subtests passed, 14 failed in 712.16s`。其中 8 个
仍是冻结 V8 manifest drift；另外 6 个来自同一个外部环境根因：四个 pymoo
baseline tests 和两个 pymoo adapter subtests 在导入未签名
`C:\miniconda3\Lib\site-packages\moocore\_libmoocore.pyd` 时被 Windows Code
Integrity 企业策略拒绝。系统 Operational log 的 3033/3077 事件绑定 policy ID
`{0283ac0f-fff1-49ae-ada1-8a933130cad6}` 与该精确文件；其 SHA-256 为
`df7dde6e737fe1e1296665e9c17994041ec2a57f214b14a6b595a55ab2b2d8b3`。

新 preflight 对该环境正确返回 `HOLD_FULL_SUITE_ENVIRONMENT`、exit 2；收据
SHA-256 为 `a1d0abb2fe7abd3eca813bebc6f7aa5d6c6a586c546bfb7c533ad495af4a6a3b`。
这 6 个失败不涉及 V9R2R1 scoped code path，但它们使当前 full-repository
validation 不合格。不能通过 skip、修改测试或借用 `consumer_use_authorized=false`
的旧 durable runtime 制造绿色；需要有权管理员提供符合企业签名策略的冻结
pymoo/moocore 环境后再跑。

```text
V9R2 completed experiment integrity       = PASS_WITH_DECLARED_SCOPE
V9R2 scoped engineering                   = PASS
V9R2 8-arm exposed-case smoke             = PASS_ENGINEERING_ONLY
V9R2R1 scoped code/packaging maintenance  = PASS
V9R2R1 full-repository validation         = HOLD_ENVIRONMENT_PLUS_FROZEN_HISTORY
full-suite pymoo environment              = HOLD_WDAC_UNSIGNED_NATIVE_MODULE
repository-wide green                     = FALSE
scientific effectiveness                  = NOT ESTABLISHED
implementation/scientific independence    = FALSE
CI95 / W-T-L / simultaneous inference     = NOT PRODUCED
development study readiness               = PRE-DEVELOPMENT HOLD
full development matrix                   = NOT AUTHORIZED / NOT STARTED
selection / confirmation / formal         = PROHIBITED
IJOC                                      = HOLD / NO-SUBMIT
```

完整复制命令见 `docs/V21E3R1_V9R2R1_RUNBOOK.md`。管理员修复 native-code
信任环境后，必须先让 environment preflight 变为 PASS，再重跑 6 个 pymoo
回归和全仓；验收仍只允许原 8 个冻结 V8 drift。即便最终全仓回到该预期失败
集合，科学门禁仍保持 HOLD。
