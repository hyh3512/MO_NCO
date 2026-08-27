# V9R2R1 public-release boundary

## Status

This repository is an engineering-reproduction candidate for
`mo-nco==0.21.3.14`, revision `V21E3R1_V9R2R1`. Public visibility is not a
scientific phase transition, a license grant, or a claim of complete
repository or scientific reproducibility.

The frozen 203-file source candidate remains identified by:

```text
source_tree_sha256 = 50ad30da8670eb488848e6db084084185fea7725e86c7fea480639caa193d9eb
source_identity_changed_by_public_metadata = false
```

The checked-in public reference observation at commit
`f6ad6a73ea9e2c46eeadded3f4446775097fdc48` used 136 test modules and
collected 1408 JUnit testcases. The current checkout closure contains 137 test
modules after adding the generic CI sanitizer security regression module; this
does not reinterpret the byte-bound reference observation. That observation
was:

```text
pytest = 78 failed, 1328 passed, 4 skipped, 267 subtests passed
junit = 1408 testcases, 77 failed/error testcases, 1327 passed, 4 skipped
repository_wide_green = false
```

One JUnit testcase carries two failed parametrized subtests, which explains the
78-versus-77 representation difference. All 78 pytest outcomes are classified
as 70 held/rights-sensitive dependencies, seven frozen-V8 fail-closed signals,
and one withheld sealed-output dependency. See
[`../provenance/V9R2R1_PUBLIC_CHECKOUT_VALIDATION_RECEIPT.json`](../provenance/V9R2R1_PUBLIC_CHECKOUT_VALIDATION_RECEIPT.json).

## Material intentionally public in this snapshot

- the frozen 203-file V9R2R1 engineering source candidate;
- tests and small engineering-evidence records needed to inspect the scoped
  validation boundary;
- the 12 authors-generated, exposed-development instances already frozen for
  development-only regression;
- protocol documents, read-only metric/accounting analyzers, and a neutral
  event-stream comparator;
- synthetic development-only golden comparator fixtures;
- citation, copyright, and third-party inventory metadata.

The golden files are constructed protocol fixtures. Labels such as
`external_valid` identify a comparator role, not an external investigator,
independent implementation, or real external custody.

## Material prohibited from this public tree

- secrets, credentials, private keys, access tokens, or user-home paths;
- future entropy or unreleased random material;
- selection, confirmation, or formal-study case bytes, partitions, receipts,
  outcomes, or materialized protocols;
- submission artifacts or language claiming that submission is authorized;
- SQLite databases and their WAL/SHM companions;
- wheels, source archives, virtual environments, caches, build trees, or large
  trace/result warehouses;
- third-party data or solver binaries lacking verified redistribution rights.

## Independent-reproduction boundary

The standard-library-only utilities can independently recompute selected
metrics or validate supplied records without importing `mo_nco`. The neutral
algorithm-event comparator validates schemas, hash chains, and equality of
projected events. Neither capability supplies an independently written
algorithm producer.

Accordingly:

```text
metric_implementation_independence = PARTIAL_AND_SCOPED
same_implementation_read_only_reanalysis = AVAILABLE
neutral_event_stream_comparator = AVAILABLE
independent_algorithm_producer = NOT_AVAILABLE
external_custody = NOT_ESTABLISHED
third_party_scientific_reproduction = NOT_COMPLETED
scientific_independence = false
```

The simultaneous-bounds utility contains code paths that can validate
selection or confirmation inputs. Shipping that validator does not create
those inputs, authorize either phase, or permit materialization of later-stage
cases.

## Scientific and publication gate

The current terminal classification remains:

```text
repository_wide_green = false
environment_lock_requirement_satisfied = false
full_algorithm_decision_replay = NOT_IMPLEMENTED
selection_authorized = false
confirmation_authorized = false
formal_authorized = false
scientific_independence = false
ijoc_submission_authorized = false
PUBLIC_RELEASE_SCIENTIFIC_STATUS = HOLD
```

Passing a source hash check, unit test, metric recomputation, trace replay, or
synthetic golden comparison is engineering evidence only. It must not be
reported as formal proof, scientific effectiveness, independent reproduction,
or submission readiness.

## Licensing boundary

The repository uses a conservative all-rights-reserved, no-license-grant
notice. It is publicly inspectable but is not openly licensed and does not yet
satisfy open-source or open-science licensing expectations. A future license
change requires an explicit decision by the actual rights holders.
