# Prospective protocol gate and amendment ledger

This gate implements the draft → locked → outcome-accessed boundary for the
[confirmatory protocol](confirmatory_protocol.md). The repository's
`configs/confirmatory_protocol.json` is deliberately a **blocked draft**. The
planning simulation and analysis specification are available; actual device,
reference, ground, units, acquisition verification, responsible-person approval,
resource commitment and fitted model/training-environment evidence still
require real study evidence. Issue #10 supplies the reference environment and
its verification mechanism, described in [the reproducibility guide](reproducibility.md).
Issue #11 supplies [development fitting, JSON export and external inference](fixed_external.md),
including a prediction-bound outcome-access event before labels are opened.
Passing automated tests does not provide real model-training
evidence, enroll anyone, or register a study externally.

## Commands and their meaning

Run commands from the repository root with the package installed, or set
`PYTHONPATH=src`. The following paths illustrate operator-owned inputs and new
output locations; they are not provided live-study artifacts.

```bash
python scripts/16_protocol.py validate --protocol configs/confirmatory_protocol.json
python scripts/16_protocol.py lock --protocol outputs/study/protocol.json --out-dir outputs/study/lock-v1
python scripts/16_protocol.py verify --lock-dir outputs/study/lock-v1
python scripts/16_protocol.py launch --lock-dir outputs/study/lock-v1 --cohort outputs/study/cohort.json
python scripts/17_audit_enrollment.py --lock outputs/study/lock-v1 --enrollment outputs/study/enrollment.json --out-dir outputs/study/enrollment-audit-1
```

For fixed-model execution, follow the [fit → predict → score commands](fixed_external.md).
`score-external` registers outcome access automatically after validating the
immutable predictions and before opening labels. The legacy manual
`record-outcome-access` command remains available for archive workflows, but
its unbound event cannot authorize new fixed-model scoring.

`validate` lists missing or contradictory decisions. Exit code 0 means the
decision fields are complete enough to attempt a lock; it does **not** verify
artifact bytes or authorize recruitment. Incomplete drafts and failed gates exit
with code 2 and actionable errors. `lock` checks real evidence, source identities
and consistency, then publishes a self-contained directory atomically. Failure
does not publish a partial lock. Existing outputs are never overwritten, and
canonical `results/`, feature tables and raw data cannot be output destinations.

`verify` checks the entire bundle and its event chain. `launch` requires a valid
bundle and an outcome-blind **complete cumulative roster**, with no development
or previously exposed participant, using literal stable participant identifiers.
It rejects extra fields such as outcome scores. A changed number formatting
convention does not define a new person: the operator must preserve the same ID
across cohorts and sessions. Software preserves distinct strings such as `001`
and `1`; it does not infer that they belong to the same human.

## What a lock freezes

| Evidence | Required consistency |
| --- | --- |
| Protocol snapshot | Versioned decisions, acquisition verification, adult population, A/B identity and ordering, eligibility, budgets, replacement rules, estimands, thresholds, multiplicity, participant inference, stopping and amendments |
| Source code | Actual clean Git commit, source-file SHA-256 inventory and aggregate fingerprint, captured automatically at lock time |
| Development dataset | JSON inventory plus copied bytes of each named source and any legacy mapping; no dangling dependency on the original source directory |
| Development manifest | Exact literal development IDs, development inventory hash and feature-schema hash |
| Primary and reference models | Separate fitted JSON files and training manifests; executable numerical state, feature/class order, source/data/environment identities and development-only grouped splits are verified without pickle |
| Analysis configuration | Sampling, channels, epoch length, marker and numerical QA settings consistent with the protocol; bounded-count preflight contract |
| Feature schema | Exact ordered 21 alpha-PLV and 168 BP+COH candidate features, learner identities, class mapping and acquisition identity |
| Precision report | Completed design simulation with embedded configuration and engine identity; reconstructed coverage/width gates, all scenarios and actual selected N/cap |
| Environment | Verified reference evidence with embedded definition/requirements, wheel receipt and installed-content checks, exact Python/dependencies, native libraries, thread settings, environment identity and lock identity |
| Acquisition verification | Actual operator, timezone-aware verification time, positive verification and digest of the declared acquisition facts |

All frozen artifact bytes are copied into `artifacts/`. Development source bytes
are copied into `data_sources/`; the original development inventory bytes and
hash remain unchanged. Its original source paths are historical locators. Bundle
verification uses the copied source inventory, allowing the original files to be
moved or removed without invalidating an intact lock.

The gate rechecks source bytes and source-code identity before publication.
Drafts must be locked from a clean source checkout. Do not fill a manually guessed
`code_commit` into the configuration: source capture is automatic and avoids a
self-referential committed-configuration hash. A reproducible deployment still
needs the corresponding Git revision and recorded environment. The bundle is an
evidence archive, not a Python installation or model execution container.

Precision evidence cannot be replaced by `{"passed": true}`. The gate validates
the actual report and binds its N, recruitment cap, confidence, bootstrap
resamples, seed, target half-width, coverage/width-probability floors, retention
assumption and recruitment-completion assurance to the protocol. The simulation
must span the frozen minimum and maximum per-class epoch counts. Floating-point
representations are compared within `1e-12`; this does not loosen the stated
scientific thresholds. Coverage and width-probability floors remain at least
0.90. The current design selects 60 evaluable participants and a cap of 94, not 60
epochs or sessions.

### Required model training manifest

Each model's JSON manifest uses `schema_version: eeg-training-manifest-1` and
contains all of these fields:

| Field | Meaning |
| --- | --- |
| `model_role`, `learner_id` | `primary` / `alpha_plv_sfs_lda` or `reference` / `bp_coh_sfs_lda` |
| `model_sha256` | Exact fitted model bytes |
| `development_manifest_sha256` | Exact development manifest bytes |
| `training_data_sha256` | Exact development **dataset inventory** bytes, whose source hashes are also verified |
| `analysis_config_sha256`, `feature_schema_sha256` | Exact frozen analysis and feature schema files |
| `environment_sha256` | Exact frozen environment evidence bytes, binding both models to their declared training environment |
| `environment_id` | Canonical identity of the complete observed environment evidence |
| `environment_lock_id` | The reference lock identity actually verified by that environment evidence |
| `training_subject_ids` | Unique literal IDs equal to the declared development set |
| `eligibility_rule_sha256` | Canonical SHA-256 of the protocol's entire `eligibility` object |
| `fitted_on_development_only` | Boolean `true` |
| `target_fit_performed` | Boolean `false` |

Changing a model binary without regenerating consistent provenance fails.
Changing eligibility while preserving an old model manifest also fails. Both
models must share the same development data, eligibility rule and frozen training
environment. Replacing the environment file with an unrelated, separately hashed
environment cannot satisfy either model's training provenance. A versions-only
record, a legacy unknown environment, or a lock file merely present on disk does
not meet `freeze.environment_policy: verified_reference_environment`. Verification
of an archived lock uses its embedded evidence, not the verifier's current host;
new execution must separately verify the live environment.

The new `freeze.model_artifact_policy: executable_fixed_lda_v1` additionally
requires `model_artifact_schema`, `model_id`, `execution_scope`,
`protocol_training_contract_sha256`, `source_protocol_sha256` and `code` in each
training manifest. These bind actual validated model JSON to its clean fitting
revision and exact scientific contract. The contract excludes administrative
approval/registration/status and artifact locators so fitting can precede final
freeze; acquisition and other scientific decisions remain bound. The model also
retains the exact original protocol/config/schema bytes and hashes. See the
[artifact contract](fixed_external.md) for the fitted numerical and split fields.

Archived opaque evidence remains readable under the historical policy, without
claiming that opaque bytes independently prove fitting. Executable markers in a
model or training manifest force the new semantic checks even if an outer policy
field is removed. New external inference always requires executable models and
rejects archived opaque or `software_validation` artifacts.

### Acquisition and registration

Acquisition verification JSON uses `schema_version:
eeg-acquisition-verification-1`, `verified: true`, `acquisition_sha256`,
`verified_by` and `verified_at`. The digest is the canonical SHA-256 of the full
protocol `acquisition` object. Required facts cannot be null, blank, placeholder
text or absent. Setting a boolean to true without the matching evidence and
actual operator record is not a verified acquisition.

External registration is optional and is never implied by local locking. Use
`study.registration.status: not_registered` without a URL/time if no registration
exists. `registered` requires its actual HTTPS URL and timezone-aware timestamp.
Approval, acquisition verification and any claimed registration must precede the
lock; recruitment must follow it. Timestamps provide consistency checks, not
independent proof of external events.

## Cohort and outcome-access boundary

The separate cohort JSON contains exactly:

```json
{
  "schema_version": "eeg-confirmatory-cohort-1",
  "cohort_id": "confirmatory-a-v1",
  "protocol_id": "eeg-ma-A-capped-v1",
  "subject_ids": ["new-001", "new-002"],
  "outcomes_accessed": false
}
```

The two IDs here only demonstrate syntax. Every launch check refers to the
complete cumulative roster, never a per-session or incremental batch. The
separate enrollment file follows the schema documented in
[the confirmatory protocol](confirmatory_protocol.md). It records population
eligibility/consent attestations, recruitment and disposition timestamps,
prespecified session attempts, task counts and shared-QA report hashes. It must
not include model scores or predictions.

Before first outcome access, the gate requires all of the following:

1. The cohort and enrollment roster contain exactly the same unique people.
2. No person appears in current or historical development or an exposed ancestor
   cohort; repeated sessions do not increase independent N.
3. The full roster passes the score-blind session, replacement, epoch-budget and
   recruitment chronology audit.
4. Recruitment has stopped because the evaluable-N target or unique-person cap
   was reached, and every already-recruited participant has a final disposition.
   Already-enrolled people finish before outcome access; all eligible people are
   included, including any prespecified in-flight overshoot of target N.
5. Recruitment follows the lock and all dispositions and sessions finish before
   the outcome-access event. No target-based enrollment occurs after the
   stopping time.

At the cap, incomplete precision is a reportable result. It is not permission to
recruit after looking at accuracy. A primary endpoint can remain evaluable when
Type2 is incomplete, while any structural QA failure rejects the entire
recording. The shared raw `qa.preflight_sessions` check remains mandatory before
feature extraction: an internally consistent metadata roster does not replace
actual raw QA.

`record-outcome-access` stores the exact cohort and enrollment bytes, their
hashes, participant IDs, N counts and timestamp in an immutable hash-linked event.
The new `score-external` path additionally binds the exact `prediction_id`,
`protocol_lock_id`, cohort SHA-256, complete subject roster and prediction
creation time. Predictions must already be complete and validated. It reuses an
event only for those exact predictions and identical cohort/enrollment bytes;
labels are opened only after the event commits. A failed label check or score
write preserves exposure and permits only an auditable retry at a new path.
Once this event commits, **all further launch checks on that lock fail**, even
for new IDs. This closes the possibility of using the command to peek and then
extend recruitment. It does not prevent someone from bypassing the software and
looking at data elsewhere; the operator is responsible for recording access
before inspecting any held-out model outcome.

## Event integrity and interrupted writes

The immutable `protocol_lock.json` identifies the frozen snapshot. `events/`
contains digest-named events; `STATE.json` is an atomically replaced pointer to
the committed chain. Writers use the existing OS-level lock. Verification rejects
modified events, broken links, missing committed events, reordered sequences,
unexpected files, symlinks and orphan events.

Prediction preparation/publication holds the same OS lock through the atomic
directory rename, so a cooperating outcome writer cannot commit between the last
authorization check and publication. New prediction also requires a clean live
source fingerprint equal to the frozen code and a verified live reference lock.

If an event file is written but updating the state pointer fails, the event is
not discarded. Verification, launch and amendments fail closed until the intact
append is recovered:

```bash
python scripts/16_protocol.py recover-event --lock-dir outputs/study/lock-v1
```

Recovery commits exactly one valid event that extends the existing pointer and
revalidates its complete outcome evidence. It never resets exposure or deletes
history. Multiple, corrupt or otherwise ambiguous orphan records require human
investigation, preserving the original files. Do not manually edit `STATE.json`
or delete ledger records to regain launch eligibility.

This is a **local audit contract**, not tamper-proof access control, an external
timestamping service or authenticated identity registry. An administrator able to
replace an entire bundle and recompute every digest can fabricate a new history.
Keep immutable copies and independent repository or registration records when
stronger provenance is needed. The software makes ordinary accidental or partial
mutation detectable and does not claim to authenticate human attestations.

## Amendments and fresh holdouts

An amendment always writes a new output directory, advances the protocol version,
records a reason and links the parent lock and parent-state digests. Parent
snapshot/model/data bytes remain unchanged. The parent is reverified before the
new bundle publishes. IDs from **every** development generation and exposed
ancestor remain excluded, even if a new development manifest removes them.

```bash
python scripts/16_protocol.py amend --lock-dir outputs/study/lock-v1 --protocol outputs/study/protocol-v2.json --out-dir outputs/study/lock-v2 --reason "Prospective documented change" --holdout outputs/study/new-holdout.json
```

Before outcome access, a prospective change can be made through a new lock.
Changing the development dataset or manifest additionally requires a distinct,
untouched holdout, so expanding development never silently consumes the test
boundary. After outcome access, every amendment requires a new cohort definition
and holdout roster disjoint from all current/historical development and exposed
subjects. Model, QC, threshold or sample-size changes cannot revalidate a cohort
whose outcomes were already seen.

A switch to `pilot_estimation` must occur before outcome access through a linked
new lock. Changing the mode alone does not authorize fewer people or reuse the
60-person precision evidence for an arbitrary smaller N. A resource-limited plan
must prospectively choose its actual width target and budget, regenerate a
matching validated precision report and bind that report in the amended lock.
Pilot interpretation does not make confirmatory support claims. The original
confirmatory plan remains preserved, and a retrospective downgrade after
unblinding is rejected.

Protocol B stays explicitly inactive and has zero weight in A's primary result.
Its planned ordering, balanced sequence and repeated-person treatment remain
visible. This v1 executable gate cannot activate B; B requires a separate study
lock and validated acquisition/analysis implementation before collection.

## Verification scope

Synthetic tests exercise complete locking, self-contained source copies,
literal-ID separation, precision/model binding, artifact modification, early or
future outcome access, recruitment closure, failed pointer writes and recovery,
pre/post-outcome amendments, historical development exclusions and CLI failures.
Synthetic fitted files and acquisition approvals are clearly labeled and never
become the repository's real protocol lock. The production draft remains blocked
until the missing real evidence is supplied.
