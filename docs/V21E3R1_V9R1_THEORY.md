# V21e3r1 V9R1 理论边界与数值合同

## 0. 当前授权状态

```text
V8 development promotion             = HOLD
V9R1 theory-helper repair             = DEVELOPMENT-ONLY CANDIDATE
V9R1 development study readiness      = PRE-DEVELOPMENT HOLD
selection / confirmation / formal     = PROHIBITED
IJOC submission                       = HOLD / NO-SUBMIT
```

V9R1 只修复局部恒等式、数值输入和证明假设。它不继承 V8 的 promotion receipt，也不证明 V9R1 已完成运行器资源门禁、跨臂诊断、独立 replay、两族回归或 prospective development protocol。任何一次 helper 测试通过都不能改变上述 HOLD。

## 1. Information-time equivalence 的完整假设

令有限可行集为 \(\Omega\)，canonical map 为 \(c:\Omega\to\mathcal C\)，确定性目标为 \(F:\Omega\to\mathbb R^d\)。必须显式要求 canonical identity 与目标一致：

\[
c(x)=c(y)\Longrightarrow F(x)=F(y).
\]

实现 helper 接收的对象必须已经是 hashable canonical identities；helper 不推断 tour rotation、reversal 或其他领域等价关系。对非空有限 attempt history \(x_1,\ldots,x_A\)，首次访问按 \(c(x_t)\) 判定，ordered first-visit path 是有序 tuple，而不是无序 set。

两条 histories 的 quality-time 等价还要求它们共享同一个 \(F\)、canonical contract、目标方向、固定 objective box、固定 HV reference point、固定 IGD\(^+\) reference set、固定 additive-epsilon reference set，以及同一个 charge-time AUC 插值和端点约定。只在这些对象全部冻结且 objective calls 成功时，相同 ordered first-visit path 才推出相同 ordered objective history、all-evaluated archive 和 charge-indexed metrics。该结论不推出 attempt time、wall time、RSS、operator attribution 或 failure behavior 相同。

## 2. Operator productivity 的解释边界

令 \(N_t\) 表示一次 attempt 是否产生 first-evaluated canonical state，令 \(G_t\in[0,1]\) 为该事件带来的 normalized all-evaluated archive gain，duplicate 时 \(G_t=0\)。当 \(q_{o,t}=\Pr(N_t=1\mid\mathcal F_{t-1},o)>0\) 时，

\[
\mathbb E[G_t\mid\mathcal F_{t-1},o]
=q_{o,t}\,\mathbb E[G_t\mid N_t=1,\mathcal F_{t-1},o].
\]

若 \(q_{o,t}=0\)，条件事件概率为零，conditional gain 不可统计识别。代码为保持经验分解而返回 0；该 0 是计算约定，不是“已证明质量为零”。经验恒等式是描述性 accounting identity，不是 operator 的随机化因果效应，也不授权跨臂 superiority。

## 3. 多资源合同与 \(S\) 的精确定义

V9R1 中 \(B\) 是 first true objective evaluations，\(A\) 是提交给 durable ledger 的 attempts。可执行合同中 `S = structural_candidate_generations + cache_membership_probes`：前者在每个受限候选生成事件前计账，后者在每个 exact cache-membership query 前计账。该 \(S\) 仍不直接计量排序、hash construction、内存分配或数据库 I/O，因此不得将 \(S\) cap 误解为完整资源证明。\(T\) 是当前 Python 进程用 `time.perf_counter()` 测得的 monotonic elapsed wall time，不是 process-tree CPU/RSS 计量。`DualResourceBudget` 已在 V9R1 主运行器中 fail-closed 接入；超过 \(A/S/T\) cap 时会先持久化 `V9_RESOURCE_CAP_EXHAUSTED` FAILURE receipt 再停止。任何科学运行仍必须预先冻结 \(B/A/S/T\) caps，并另行报告 process-tree peak RSS、trace bytes 和 replay time。

## 4. Bounded first-unseen screening

合法 screening 对象是运行前冻结的 deterministic generator rule，而不必是运行前物化的完整 candidate list。每次调用生成的有序候选可以依赖 \(\mathcal F_{t-1}\) 中已知的 parent、seed、problem data 和 search index，但不得调用候选 objective 或读取未来信息。`is_seen` 必须对每个已 canonicalized candidate 返回 exact Python `bool`；truthy integer、字符串和第三方布尔标量均不属于该合同。

“首个 unseen”只保证在被检查的 bounded prefix 内不命中 exact cache，不保证 quality、runtime improvement 或全局 novelty。可独立 replay 的正式 witness 仍需绑定候选 canonical hashes、逐项 seen results、选中 rank 和 generator semantics。

## 5. Archive-compensated local invariant

设 typed population 为 \(X=(x_1,\ldots,x_R)\)，每个 \(U_r\) 是运行期间固定、有限且使用同一 frozen normalization 的 scalar energy。设 \(A_t\) 是 append-only all-evaluated nondominated archive，\(H(A_t)\in[0,1]\) 使用同一 frozen box/reference。完成初始化后，对一次 candidate evaluation 定义

\[
\Delta H_t=H(A_t)-H(A_{t-1})\in[0,1],
\qquad
\delta_{s,t}=U_s(y_t)-U_s(x_{s,t-1}).
\]

若 exact-arithmetic rule 满足

\[
\sum_{s\in S_t}[\delta_{s,t}]_+\le\lambda\Delta H_t,
\]

则

\[
\Psi_\lambda(X_t,A_t)-\Psi_\lambda(X_{t-1},A_{t-1})\le0,
\quad
\Psi_\lambda(X,A)=\sum_r U_r(x_r)-\lambda H(A).
\]

浮点实现使用显式 \(\varepsilon\in[0,10^{-6}]\)。只有落在 \([-\varepsilon,1+\varepsilon]\) 的 normalized HV/gain 才可按边界 roundoff clamp 到 \([0,1]\)。每次 replacement 的可验证结论是

\[
\Psi_t-\Psi_{t-1}\le\varepsilon.
\]

因此 \(n\) 次 replacement 的累计结论必须写成

\[
\sum_{t=1}^{n}\sum_{s\in S_t}[\delta_{s,t}]_+
\le \lambda\{H(A_n)-H(A_0)\}+n\varepsilon
\le \lambda+n\varepsilon,
\]

而不能在使用非零 tolerance 时继续声称 exact nonincrease。正式 receipt 应累计核对实际 slack，而不是只检查逐事件 Boolean。

## 6. 数值类型合同

所有 counts 和 identifiers 必须是 exact built-in `int`，排除 `bool`。所有连续数值必须是 finite exact built-in `int` 或 `float`，排除 `bool`、字符串、`Decimal`、NumPy scalar 和任何依赖隐式 `float(...)` 的对象。Normalized HV、normalized HV gain 与累计 normalized archive gain 限定在 \([0,1]\)（只允许上节声明的边界 tolerance）。`is_seen` 必须返回 exact built-in `bool`。

## 7. 结论边界

上述结果是 finite deterministic search 的局部恒等式和 accounting invariant。它们不证明 global Pareto convergence、approximation ratio、final HV superiority、\(\lambda\) 最优、screening 有效、operator 因果效应、跨问题泛化、独立复现或投稿资格。V9R1 只有在代码、资源 cap、metric/reference manifest、两族 matched development protocol、simultaneous inference 和独立 verifier 全部重新冻结后，才可以离开 PRE-DEVELOPMENT HOLD。
