# Research reference environment

`reference.json` and `reference-linux-x86_64-py312.lock` define a **new prospective
reference**, not the unknown runtime that produced historical research results.
The current reference uses CPython **3.12.14**, Linux x86_64 with glibc >=2.28,
25 pinned wheels, and single-thread BLAS/OpenMP settings. GitHub's numerical job
uses Ubuntu 24.04. Hardware, kernel, native-library versions/hashes and CPU
dispatch architecture remain recorded observations; numerical acceptance uses
explicit tolerances rather than a cross-machine bitwise claim.

## Reconstruct and verify

Use the exact Python patch version in `reference.json`. Start with a fresh venv
so unrelated packages cannot silently enter the reference:

```bash
python3.12 -m venv .venv-reference
mkdir -p outputs/reference-bootstrap
.venv-reference/bin/python -m pip install \
  --require-hashes --only-binary=:all: --force-reinstall --no-deps \
  --report outputs/reference-bootstrap/install.json \
  -r environment/reference-linux-x86_64-py312.lock
.venv-reference/bin/python -m pip install --no-deps --no-build-isolation -e .
.venv-reference/bin/python -m pip check

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0 MPLBACKEND=Agg
export EEG_MA_INSTALL_REPORT="$PWD/outputs/reference-bootstrap/install.json"
.venv-reference/bin/python scripts/18_environment.py verify \
  --out outputs/reference-bootstrap/environment.json
```

The interpreter name `python3.12` alone is not proof of its patch version; the
verification command checks the actual `3.12.14`. `--force-reinstall` makes the
pip receipt include **every** locked package, including tooling already present
in a fresh venv. `--no-deps` is safe here because every active dependency is
explicitly resolved in the lock and independently checked; `pip check` verifies
the installed dependency relationships. No build-isolation environment is
silently resolved for the local editable project. Use a new output directory
for each bootstrap; do not overwrite previous evidence.

The pip receipt may contain machine-local paths or index URLs. Keep the raw
receipt with private/local or CI artifacts as appropriate. Environment evidence
stores only normalized package names, versions, wheel basenames and SHA-256s,
not source credentials or absolute download paths.

For capture without a reference claim:

```bash
python scripts/18_environment.py capture --out outputs/observed-environment.json
```

This succeeds for a compatibility runtime while recording `mismatched` or
`unavailable` reference status. Merely having a lock in the checkout cannot make
an arbitrary installed environment pass strict verification.

## What is checked and preserved

The reference definition binds each wheel archive SHA-256, package METADATA
SHA-256, resolved active requirements, and an inventory fingerprint for **8,683
original package files** in the current wheel set. Live reference verification
hashes actual installed bytes and compares this independent fingerprint; a
matching version string or copied pip receipt alone is insufficient. It ignores
generated `.pyc`, installation markers, generated command-entry scripts and
relocated `.data` files when constructing this content fingerprint. Complete
wheel-archive hashes still cover the original distributions. These exclusions
avoid machine-specific generated files and are not a claim to authenticate an
entire operating system.

Observed packages include transitive packages and tooling. Unexpected third-party
distributions, missing dependencies, duplicated metadata, modified installed
content, a missing/incomplete wheel receipt, a Python/platform mismatch, or
different thread settings prevent reference status. Native BLAS/OpenMP libraries
are loaded consistently before snapshots so fitting a model does not by itself
change the environment identity. Actual native-library hashes, versions, thread
counts and architecture are recorded without absolute library paths.

The run identity uses this evidence automatically through
`provenance.environment_identity()`. `environment_id` hashes the whole observed
payload. `lock_id` hashes the exact reference-definition and requirements-file
digests. Any relevant environment change therefore changes new run identity and
prevents stale resume. Existing historical manifests and `legacy_unverified` /
`historical_environment: unknown` inventories remain unchanged.

Model training manifests bind the exact environment artifact and `lock_id`;
the protocol gate validates the archived evidence without comparing it to the
machine reading the lock. Capturing a verified environment does not claim that
the future #11 models have already been trained in it.

## Controlled upgrades and independent archives

An upgrade requires a new versioned reference definition, complete wheel hashes
and dependency closure, followed by a same-code/same-fixture numerical comparison
between the old and proposed environments. Changes in numerical outputs must be
reported and reviewed before adopting the new reference. Broad Python 3.10/3.12
compatibility testing continues independently and does not replace this gate.

`EEG_MA_REFERENCE_SPEC` or `--reference` chooses an explicitly supplied candidate
definition. Its lock file must be a same-directory basename; directives,
includes, URLs, ranges and unpinned requirements are rejected before pip. The
stdlib-only `load_reference()` API permits that preflight before dependencies
are installed. `validate_environment_evidence()` instead checks embedded
definition/lock text and their observed evidence; it does not consult local
package versions, so an intact archived study lock can be verified elsewhere.

The evidence is a local reproducibility/audit contract, not a digital signature
or proof of human behavior. It cannot reconstruct unknown private raw EEG or
retroactively establish the environment of old results.

## Dependency provenance

The 25 versioned wheels were downloaded and hashed; their actual ZIP METADATA
and RECORD contents were used to check active dependency closure and installed
content fingerprints. Dependencies were selected to retain the already tested
numeric stack used for prospective planning, not from a claim about the legacy
research runtime. No wheel binaries are committed to the repository.

Primary tool documentation:

- [pip secure installs](https://pip.pypa.io/en/stable/topics/secure-installs/)
- [pip installation reports](https://pip.pypa.io/en/stable/reference/installation-report/)
- [pip build isolation](https://pip.pypa.io/en/stable/cli/pip_install/#cmdoption-no-build-isolation)
