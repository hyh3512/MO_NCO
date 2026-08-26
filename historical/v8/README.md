# V21E3R1 V8 historical identity

The recommended `v21e3r1-v8-frozen` tag is intentionally absent. The frozen
source receipt available under internal custody says both:

```text
public_redistribution_authorized = false
source_archive_scope = SOURCE_INVENTORY_ONLY_INTERNAL_CUSTODY_NO_REDISTRIBUTION_AUTHORITY
```

Consequently, neither the historical archive nor a Git tag containing its
bytes may be published from this public repository without a new, explicit
rights-and-custody authorization. This is not a missing-file accident and must
not be bypassed by rebuilding the old tree from the current V9 sources.

Machine-readable hashes and the fail-closed state are recorded in
`provenance/V21E3R1_V8_FROZEN_TAG_STATUS.json`. The historical CI job must
remain blocked until the exact tag is separately authorized, materialized, and
verified. Current-tree tests must not be changed to `skip` or `xfail` to hide
the eight source-identity failures.

This HOLD does not authorize selection, confirmation, a formal study, a
scientific claim, or IJOC submission.
