# GitHub public engineering-inspection scope

This public repository exposes the reviewed V9R2R1 source, test, protocol, and
engineering-evidence material that is currently authorized for remote
inspection. Public visibility is not an open license or a scientific release.

Identity: `mo-nco==0.21.3.14`, `V21E3R1_V9R2R1`.

Included:

- the deterministic 203-file V9R2R1 engineering source candidate;
- the 12 author-generated exposed-development instance files and the small
  manifests explicitly cleared for this engineering snapshot;
- the current environment-recovery, targeted, and full-repository JUnit files;
- the current wrappers and tests needed to explain the exact eight historical
  fail-closed source-manifest failures, plus a registry of their exact node-ids;
- the gate receipt and both complete smoke JSON evidence chains; SQLite traces
  remain excluded.

Excluded:

- virtual environments, caches, temporary directories, wheels, source ZIPs,
  SQLite/WAL files, and superseded diagnostic JUnit files;
- selection, confirmation, formal-study, or submission artifacts;
- the historical V8 archive/tag, for which the custody receipt explicitly says
  public redistribution is not authorized;
- large historical result ZIPs and wheelhouses, which require a separately
  authorized Release/DOI/LFS publication path and verified third-party rights.

Strict boundaries:

- `V9R2R1 scoped engineering = PASS`;
- the checked-in recovery JUnit is a byte-bound historical engineering
  reference with exactly eight frozen-V8 failures; it is not a claim that a
  fresh public checkout reproduces that full-suite result;
- the fresh public-checkout full suite fails beyond the exact-eight registry
  because prohibited, large, ambiguous-rights, and otherwise unpublished
  historical fixtures are absent; repository-wide green is false;
- `environment_lock_requirement_satisfied=false`;
- scientific independence is false, full algorithm-decision replay is not
  implemented, and later scientific phases remain prohibited;
- original evidence files are preserved byte-for-byte and can contain local
  Windows paths; they contain no detected credentials or user-home paths in the
  uploaded current-recovery set.

`GITHUB_EXPORT_CONTENTS.json` is the canonical Git-index blob inventory for
this engineering candidate. It excludes itself and `.git` to avoid recursive
self-hashing.
