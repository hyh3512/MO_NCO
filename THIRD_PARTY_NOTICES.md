# Third-party notices

This file is an inventory aid, not a complete legal review and not a license
grant. No third-party package is vendored as a wheel, archive, virtual
environment, or binary in this snapshot. Users who install or redistribute
dependencies must review the exact versions and licenses they obtain.

## Declared build and optional dependencies

The package metadata names or enables the following separately distributed
projects:

- Setuptools, used as the build backend;
- pytest, used by the test suite;
- NumPy and Numba, used by optional acceleration paths;
- cryptography, used by optional receipt/signature paths.

## Optional integrations referenced by source code

Some modules contain optional adapters or execution paths for separately
installed software, including pymoo, PyTorch, LKH-family executables, and
Paquete-related tooling or data conventions. Their presence as names or
interfaces in source code does not mean those projects, binaries, datasets, or
licenses are bundled here.

## Data and benchmark rights

No permission is granted here for third-party benchmark instances, TSPLIB
materials, Paquete materials, solver binaries, or other external datasets.
Before publishing or redistributing any such material, verify provenance,
redistribution rights, attribution requirements, and the terms of the exact
source from which it was obtained.

## Repository license boundary

The repository's LICENSE is deliberately conservative and all-rights-reserved.
It does not convert third-party components to that status, nor does it satisfy
their license obligations. Conversely, a permissive license on an external
dependency does not license this repository.
