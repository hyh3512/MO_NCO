# GitHub chat-access snapshot

This private-repository snapshot exposes the smallest reviewed V9R2R1 source,
test, protocol, and engineering-evidence closure needed for remote inspection.

Identity: `mo-nco==0.21.3.14`, `V21E3R1_V9R2R1`.

Included:

- the deterministic 203-file V9R2R1 engineering source candidate;
- the complete frozen 12-case exposed-development packet and its bound
  reference/config manifests, so the repository-level V9R2R1 regression suite
  can validate the packet without falling back to the local experiment tree;
- the current environment-recovery, targeted, and full-repository JUnit files;
- the frozen V8 plan/wrappers and tests needed to explain the exact eight
  fail-closed source-manifest failures;
- the gate receipt and both complete smoke JSON evidence chains; SQLite traces
  remain excluded.

Excluded:

- virtual environments, caches, temporary directories, wheels, source ZIPs,
  SQLite/WAL files, and superseded diagnostic JUnit files;
- selection, confirmation, formal-study, or submission artifacts.

Strict boundaries:

- `V9R2R1 scoped engineering = PASS`;
- full-repository green is false: the current full JUnit contains exactly eight
  frozen-V8 source-manifest failures and no unexpected failure family;
- `environment_lock_requirement_satisfied=false`;
- scientific independence is false, full algorithm-decision replay is not
  implemented, and later scientific phases remain prohibited;
- original evidence files are preserved byte-for-byte and can contain local
  Windows paths; they contain no detected credentials or user-home paths in the
  uploaded current-recovery set.

`GITHUB_EXPORT_CONTENTS.json` is the detached allowlist/hash manifest for this
snapshot. It excludes itself and `.git` from its file inventory.
