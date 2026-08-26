# Independent reproduction boundary

This directory contains small, inspectable utilities and synthetic fixtures
for scoped metric recomputation, operator-accounting reanalysis, and neutral
algorithm-event comparison. Most utilities use only the Python standard
library and do not import `mo_nco`.

## Exact capability boundary

The included material can establish only narrowly stated properties:

- `recompute_v21e3r1_metrics.py` independently reimplements the biobjective
  nondominated archive, normalized hypervolume, and exact per-evaluation
  left-continuous AUC for a supplied SQLite trace;
- `recompute_v21e3r1_successor_metrics.py` recomputes scoped successor metrics
  from a supplied trace and preserves all later-phase authority fields as
  false;
- `reanalyze_v21e3r1_operator_accounting.py`, together with its frozen JSON
  specification, performs read-only accounting checks over supplied completed
  development diagnostics;
- `recompute_v21e3r1_simultaneous_bounds.py` independently implements a frozen
  paired-case statistical calculation and validates explicit phase bindings;
- `compare_v21e3r1_algorithm_events.py` is a neutral validator/comparator for
  two supplied canonical event streams;
- `golden/*.jsonl` are synthetic development-only fixtures for exercising the
  comparator, not captured scientific runs.

These files do **not** provide an independently implemented optimizer,
algorithm-execution independence, authenticated producer identity, external
custody, a third-party investigator, or scientific independence. In
particular, `golden/external_valid.jsonl` is a differently labelled synthetic
placeholder and is not evidence that an external producer exists.

## Authority remains closed

```text
SELECTION_AUTHORITY = false
CONFIRMATION_AUTHORITY = false
FORMAL_STUDY_AUTHORITY = false
IJOC_SUBMISSION_AUTHORITY = false
SCIENTIFIC_INDEPENDENCE = false
```

The presence of code that can validate a future selection or confirmation
record does not authorize creating that record or materializing later-stage
case bytes. Use only already exposed development material unless a separately
documented predecessor gate explicitly changes this status.

## Included files

```text
README.md
V21E3R1_ALGORITHM_REPLAY_SPEC_V1.md
compare_v21e3r1_algorithm_events.py
reanalyze_v21e3r1_operator_accounting.py
recompute_v21e3r1_metrics.py
recompute_v21e3r1_simultaneous_bounds.py
recompute_v21e3r1_successor_metrics.py
v21e3r1_operator_accounting_reanalysis_spec_v1.json
golden/external_valid.jsonl
golden/negative_decision_mismatch.jsonl
golden/reference_valid.jsonl
```

## Examples

Metric recomputation over a caller-supplied development trace:

```bash
python independent_reproduction/recompute_v21e3r1_metrics.py \
  --trace path/to/development-trace.sqlite3 \
  --lower=-5004,-5352 \
  --upper=0,0 \
  --output independent_metric.json
```

Inspect each utility's accepted arguments without generating evidence:

```bash
python independent_reproduction/compare_v21e3r1_algorithm_events.py --help
python independent_reproduction/reanalyze_v21e3r1_operator_accounting.py --help
python independent_reproduction/recompute_v21e3r1_simultaneous_bounds.py --help
python independent_reproduction/recompute_v21e3r1_successor_metrics.py --help
```

Outputs remain engineering receipts whose meaning depends on the exact input
provenance. They cannot be relabelled as external or scientific reproduction.
