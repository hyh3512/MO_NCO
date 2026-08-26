# V9R2R1 theory boundary

V9R2R1 preserves the V9R2 engineering algorithm semantics while repairing the
source/package identity and validation closure. The detailed mechanism is
specified in [`V21E3R1_V9R2_ALGORITHM_SPEC.md`](V21E3R1_V9R2_ALGORITHM_SPEC.md)
and the earlier finite-event argument is recorded in
[`V21E3R1_V9R1_THEORY.md`](V21E3R1_V9R1_THEORY.md).

The implemented Lyapunov witness is a finite replacement-event invariant: the
sum of accepted positive target worsening is bounded by the declared archive
credit, subject to target capacity and numeric tolerance. It is not a proof of
global convergence, superiority, optimality of the tradeoff coefficient, or
scientific effectiveness. Structural screening, cache accounting, durable
B/A/S/T caps, and all-evaluated normalized 2-D hypervolume reconstruction are
engineering contracts.

The current source identity is the 203-file manifest with source-tree SHA-256
`50ad30da8670eb488848e6db084084185fea7725e86c7fea480639caa193d9eb`.
That identity remains a pre-development engineering candidate. Scientific
independence, full-development execution, selection, confirmation, formal
study, and IJOC submission are not authorized.
