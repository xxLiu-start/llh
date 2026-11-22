# Feasibility Reflector

## Inspected architecture and insertion point

`cli._run_command` constructs `HarnessConfig` and the existing role adapters.
`manager.run` supplies the crash boundary; `manager._run_impl` owns the MEA loop.
`build_role_manager_prompt` → `_run_role_episode(manager_agent)` →
`extract_role_task_contract` / `extract_role_task_state` / `parse_role_manager_next_step`
→ **optional Reflector** → `build_role_executor_prompt` → original GUI/CLI adapter
→ `build_role_auditor_prompt` → original independent Auditor + format repair
→ `ManagedRound` → `_record_round` → next Manager invocation.

The insertion is after a valid GUI/CLI route and before Executor prompt construction.
The original contract is text, not a new JSON contract. A rejected contract becomes
ordinary harness feedback for the next invocation of the same Manager, within the
existing round budget. No second MEA loop or new Manager prompt is introduced.

Task state is Manager-maintained text. Only historical Auditor reports are authority
for verified facts. The adapter explicitly distinguishes these two sources.
`role_orchestration/rounds/round_NNN/` stores contracts, plans, state, outputs,
Auditor reports and raw trajectories; `rounds.jsonl` is the append-only ledger.
The Environment protocol exposes exec/upload/download/screenshot, not a portable
read-only resource inspection API. The Reflector never calls any of them. Unknown
files, permissions, dependency availability and opaque CLI/MCP tool catalogs remain
unknown unless evidence is supplied through its typed snapshot interface.

Existing tools are registered in individual CLI/plugin backends. The normal runtime
supplies route/backend descriptions, explicitly marked as an incomplete tool catalog.
No benchmark-specific tool assumptions or prompts are added.

Historical training outputs do not provide a reliable held-out split. The offline
pipeline therefore requires an explicit task-level split manifest. Test records
are excluded before opening their trajectory files.

Old Skill routing, three-role prompt injection, online evolution and replay training
are removed from the active package and retained under `archive/legacy_skills/`.
Historical banks and result directories are not migrated into feasibility skills.
Original benchmark evaluation code, prompts, datasets and scoring remain untouched.

## Runtime configuration

### Observe the original run first

```bash
cd /data0/liuxule/project/LHH/LongHorizon-Harness
./training/run_feasibility_shadow.sh \
  --output output/feasibility-shadow-first \
  --task "Your complete original task description"
```

Forward your already-working original `--agent`, `--model`, `--workspace` and
role options as needed. `--task @/absolute/path/task.txt` or the convenience
`--task-file /absolute/path/task.txt` reads an existing task. The task argument
is task text, not a Terminal-Bench task ID; this script launches the main package's
original MEA, not the independent Harbor benchmark evaluator.

Interpreter selection: explicit `LH_HARNESS_PYTHON`, then an explicitly set
`CONDA_ENV`, then this repository's `.venv/bin/python`, then current `python`.
For the previously used environment, prefix the command with
`CONDA_ENV=terminal-bench-2-1`. The original native role backends must already be
installed/configured. The wrapper does not install tools or change role configuration.

The script fixes `mode=shadow`, disables Skill learning and contract revisions,
and never applies a Reflector REVISE/BLOCK verdict to dispatch. Each output directory
must be fresh to avoid mixing runs. No changes to benchmark evaluation are made.
`--dry-run` writes/validates configuration without launching the Harness.

Whenever the Reflector is enabled, both shadow and gate assess current evidence
even if the bank is empty, retrieval finds no applicable Active Skill, or retrieval
fails. Skill guidance is optional. Evidence-only judgments record
`retrieved_skills=[]` and `trace.used_skill_guidance=false`. Lack of a Skill alone
never implies UNKNOWN. The old `--skip-without-skills` switch has been removed.
Old JSON files containing `assess_without_skills` remain readable, but that obsolete
field no longer controls whether evidence is assessed (including when set false).
Missing model API access still yields recorded UNKNOWN/error and original execution
continues. Native CLI login alone may not supply credentials for the separate
JSON-only observer API.

Output layout:

```text
<output>/
  console_logs/run.log                 # realtime combined stdout/stderr
  reflector_config.json               # actual fixed shadow configuration
  manager_contracts.jsonl              # every Manager round's original contract + verdict reference
  manager_contracts.csv                # concise round-by-round table
  shadow_summary.json                 # verdict counts, errors and separate Reflector token cost
  runtime/main/lh_harness/
    report.json                       # untouched original task report
    role_orchestration/rounds/round_NNN/
      manager_plan.txt
      task_contract.txt
      feasibility.json                # executable-round assessment and later Auditor link
      auditor_report.txt
```

Judgments are made for rounds dispatching a GUI/CLI Executor. Done/blocked/ask and
unfinished rounds remain in the contract table with `assessed=false`; no verdict is
invented. Ctrl+C/SIGTERM is forwarded to the original process group; after it exits,
the wrapper collects partial records and preserves the interruption/exit status.
Only observable task artifacts and structured decisions are recorded. The wrapper
does not save the invocation's credentials or raw model chain-of-thought.

Default: disabled. No retrieval, model call, contract changes or new role instructions.
Start with a separate JSON config such as:

```json
{
  "enabled": true,
  "mode": "shadow",
  "skill_bank_path": "~/.lh-harness/feasibility-skills",
  "top_k": 5,
  "max_revision_rounds": 1,
  "unknown_policy": "pass_through",
  "runtime_error_policy": "fail_open",
  "revision_exhausted_policy": "pass_through",
  "enable_skill_learning": false
}
```

```bash
cd /data0/liuxule/project/LHH/LongHorizon-Harness
PYTHONPATH=src python -m lh_harness run --task "YOUR TASK" \
  --feasibility-reflector-config /absolute/path/to/reflector.json
```

Or use `[run.feasibility_reflector]` in `.lh-harness/config.toml` with the same fields.
The JSON CLI option replaces that table. The shipped
`training/feasibility_reflector.disabled.json` explicitly disables the feature.
`enabled=true` and `mode=shadow|gate` are both required for activation.

Shadow runs the checker but always dispatches the untouched original plan/contract.
Gate dispatches ACCEPT unchanged; UNKNOWN passes through by default. REVISE writes
structured non-audit harness feedback for the next normal Manager round. This uses
the original round budget, not a hidden extra Manager loop. Default revision limit
is one per consecutive rejected series, including resumed runs. Exhaustion passes
through by default, or blocks when configured. BLOCK stops dispatch and records
`abort_reason=feasibility_blocked`; original completion logic returns incomplete,
never a synthetic success or invented Auditor verdict.

The CLI's JSON-only adapter uses the Manager's resolved provider/model, API key and
base URL (or existing provider environment variables). Claude's OpenAI bridge flag
is honored. It implements the existing AgentAdapter interface, but does not run a
command agent or expose its CLI/MCP tools. CLI login-only credentials cannot be
reused as API credentials: missing credentials produce UNKNOWN/fail-open. No
credentials are written in Reflector logs. Programmatic callers can pass an
AgentAdapter-backed `FeasibilityReflector` to `manager.run` explicitly. The adapter
receives `NoToolEnvironment`; production `JsonModelAgent` supplies no model tools.

An enabled Reflector assesses current evidence with or without Skills. Candidate
Skills are excluded from guidance. Retrieval failures are recorded separately and
fall back to evidence assessment. Oversized optional Skill context is omitted before
retrying the input-size check. If the original snapshot itself exceeds the configured
bound (default 48,000 characters), it yields UNKNOWN instead of dropping user constraints.
Insufficient evidence or model/parsing failures may also yield UNKNOWN. Model failures
default to fail-open. Each eligible subtask normally uses one checker call. Invalid
JSON/schema/evidence can trigger one bounded repair, configurable through
`max_validation_repairs=0|1` (default 1). Both calls' provider costs are counted and
share the same timeout. The evidence validators remain in force after repair. A disabled
Reflector performs no retrieval or model call.

The model input deduplicates exact copies of contract/state already embedded in
the Manager plan, retaining their complete texts in dedicated snapshot fields.
Original task text, the current contract, and prior Auditor reports are not
truncated. Original role inputs and recorded task state are untouched. Both launcher
scripts accept `--reflector-max-input-chars` (2,000–100,000; default 48,000).
`trace.raw_input_chars`, `input_chars`, `input_char_limit` and `deduplicated_fields`
make input omissions/limits observable.

UNKNOWN diagnostics distinguish `input_budget`, `model_call`, `json_parsing` and
`assessment_validation`. Parser/schema failures record a safe `validation_error`
and per-attempt response size, finish reason and usage in `model_attempts` without
storing private reasoning or raw provider errors. Evidence matching ignores
whitespace wrapping only, never paraphrases, identifiers, numeric values or negation.
The live Harbor console now prints uncertainty, error stage and validation error.

## Input/output schemas

`Snapshot` preserves the original contract and plan as text, plus original task,
Manager-maintained state, historical Auditor reports with references/status metadata,
available tool descriptions/schemas (explicitly incomplete by default), optional
already-available observations, route and a hash of the prior audit evidence.
It never reads future outputs or executes environment inspection commands.

`Assessment` returns decision (`ACCEPT|REVISE|BLOCK|UNKNOWN`), feasibility,
issues (`type`, severity, description, exact evidence quotes, affected requirement,
suggested correction), passed checks with `check_evidence`, selected Skill IDs,
confidence and uncertainty. Confidence is not a gate score. Contradictory fields,
ungrounded quotes and malformed JSON become runtime UNKNOWN. Missing-resource,
dependency-not-ready and unavailable-tool claims additionally require an actual
observation or prior audit; a Manager assertion cannot prove resource absence.
Acceptance requires evidenced passed checks; BLOCK requires a supported high-severity
issue. The seven assessment dimensions are defined in `feasibility/schema.py`.

## Skill storage and offline learning

```text
feasibility-skills/
  index.json
  candidates/FS_<id>.json
  skills/FS_<id>.json
  history/FS_<id>/v<N>.json
  history.jsonl
  generation.json
  validation.json
```

A Skill has `skill_id`, name, description, applicability_conditions,
preconditions_to_check, failure_patterns, required_evidence, counterexamples,
suggested_corrections, source_trajectory_ids, support_count, confidence,
status (`candidate|active|deprecated`) and version. The pipeline also stores
source_task_ids, task fingerprints and provenance. File names derive from generated
IDs, never model-provided file paths. All required list fields must be nonempty.

The bank uses atomic file replacements and a process lock for read/merge/write.
Skill files are authoritative, so a stale index does not hide saved skills. Rebuild:

```bash
PYTHONPATH=src python -m lh_harness.feasibility.offline recover-index --bank /path/to/bank
```

History is retained before updates. A candidate revision does not remove an existing
active version. Deprecation requires an explanation. Legacy banks are incompatible
and are never silently converted.

The manifest is a JSON array with explicit task-level splits. Paths are relative to
the manifest. Supply actual existing reports; do not point test trajectories at
training entries. Minimal format:

```json
[
  {"trajectory_id":"train_run_a", "task_id":"training_task_a", "split":"train",
   "path":"training_a/role_orchestration/report.json"},
  {"trajectory_id":"validation_run_b", "task_id":"held_out_task_b", "split":"validation",
   "path":"validation_b/role_orchestration/report.json",
   "labels":{"1":{"feasibility_defect":true,
                    "evidence":["An exact observable quote from that historical record"]}}},
  {"trajectory_id":"test_run_c", "task_id":"test_task_c", "split":"test",
   "path":"test_c/report.json"}
]
```

```bash
PYTHONPATH=src python -m lh_harness.feasibility.offline generate \
  --manifest /path/to/splits.json --bank /path/to/bank --model YOUR_MODEL
PYTHONPATH=src python -m lh_harness.feasibility.offline validate \
  --manifest /path/to/splits.json --bank /path/to/bank --model YOUR_MODEL
```

Use `--protocol anthropic` and/or `--base-url` for the existing provider. Set the
corresponding API key in the terminal; it is not a CLI parameter or manifest field.

Generation loads only training trajectories. It reconstructs pre-execution snapshots,
uses recorded Executor/Auditor/tool traces separately for offline attribution, and
requires quoted evidence and confidence >= 0.8 before generating planning/dependency/
tool-feasibility candidates. EXECUTOR_FAILURE, ENVIRONMENT_FAILURE, AUDITOR_FAILURE and
UNCERTAIN do not create skills. Positive experience adds grounded counterexamples to
candidate checks; it does not mandate extra execution steps. Exact mechanism duplicates
merge; optional model-assisted semantic equivalence requires complete structured
condition/check/correction/failure-pattern alignment. Complementary checks remain
separate. Potentially conflicting corrections under identical applicability are
flagged and cannot be automatically promoted. No active strategy is overwritten.

Validation loads only validation trajectories and sends only pre-execution snapshots
to the checker. Labels/future results stay in metric computation. Generation sources
are excluded by trajectory ID, task ID and original-task fingerprint. Test paths are
not opened. Task/path overlap across declared splits is rejected. Correct split
assignment remains the experimenter's responsibility; relabeling hidden duplicates
or an artifact's split cannot be detected perfectly by this software.

Evidence-only judgments are logged but never credited to an unselected candidate
Skill's precision/recall, independent support or promotion. Candidate improvement
requires actual candidate use and the validation evidence described below.

Default promotion requires >=3 independent supporting validation tasks, >=2 retrieved
independent tasks, precision >=0.8, recall >=0.5, observed false rejection <=0.05,
no unresolved conflicts, and measured downstream improvement with no paired regression.
No feasible negative validation examples means false rejection is unmeasured and
promotion is refused. `skill_validation.min_support` can raise the support threshold;
disabling promotion never activates unvalidated candidates. False rejection and
cross-task checks remain mandatory safety floors even if optional reporting flags
are false. Runtime `enable_skill_learning=true` is explicitly rejected: all learning
is offline, separate from the original evaluator.

Offline verdicts alone do not establish downstream success. To enable promotion,
provide paired validation outcome artifacts in an entry's `validation_artifact_paths`
and its label's `paired_downstream.evidence_paths` (exactly two paths). Each outcome
has `split=validation`, matching `task_id`, `condition=baseline|reflector`, boolean
`success`, and identical nonempty `controls` for model/tools/budget/environment.
The reflector outcome also names `skill_id`. The pipeline reads assigned validation
artifacts only, verifies control equality, and counts each task once. These must be
observations from actual controlled validation runs. The pipeline does **not** launch
paired benchmark jobs or manufacture these outcomes.

## Logs, tests, and limitations

`round_NNN/feasibility.json` contains the original contract, pre-execution evidence
reference, selected/rejected retrieval reasons, structured decision, feedback,
dispatch disposition, contract version, actual provider token counts and later
Auditor reference/report. `feasibility_summary.json` aggregates only Reflector cost;
missing provider usage stays unknown. Original task reports, rewards and evaluation
CSV files are untouched. Console progress prints checking/decision transitions.
No private chain-of-thought or raw model reasoning is persisted by this component.

Tests in `tests/test_feasibility.py` exercise disabled byte/behavior compatibility,
all decisions, bounded revisions, read-only isolation, grounded evidence, retrieval,
offline attribution, split isolation, candidate persistence/merging, held-out promotion,
regression rejection, history retention and write failures. Model outputs and paired
validation artifacts in these tests are explicitly controlled fixtures, not real
benchmark/model results.

The original Terminal-Bench evaluation uses a separate legacy orchestration package
under `eval/TB-harness/Harness/src`, which remains unchanged. Existing original-eval
commands continue to evaluate Original LHH; they do not automatically enable this
Reflector. Task-ID shadow runs now have a separate optional launcher,
`evaluation/run_feasibility_shadow_terminal_bench_pro.sh`, documented in
`evaluation/README.feasibility-shadow.zh-CN.md`. Its Harbor subclass delegates to
the original Agent's run method and wraps only the Manager AgentAdapter, returning
its original result object unchanged. It reuses the original evaluator's config,
execution and result helpers; no MEA or scoring implementation is duplicated.

Semantic applicability/attribution still depends on the configured model; exact
quotes ensure traceability but do not prove semantic correctness. Retrieval is
lexical and cross-language recall is limited. No embeddings/router model or vector
database is introduced. Opaque native tool catalogs and unobserved environment facts
are deliberately unknown. No real paid-model/benchmark run can be claimed from unit
tests; runtime verification requires the user's configured model and environment.

## Delivered changes and verification (2026-11-11)

New production files under `src/lh_harness/feasibility/`:

| File | Responsibility |
| --- | --- |
| `schema.py` | Typed config, original-contract snapshot and evidence-validated result |
| `bank.py` | Candidate/active/deprecated skills, version history, locks, atomic writes and merges |
| `retrieval.py` | Active-only lexical applicability retrieval, counterexamples, dedup and conflicts |
| `model.py` | JSON-only AgentAdapter with no tools and bounded asynchronous HTTP |
| `reflector.py` | Seven-dimension assessment, evidence validation and fail-open boundary |
| `integration.py` | Read-only snapshot, policy decisions and normal Manager feedback |
| `offline.py` | Split-aware loading, attribution, candidate generation, held-out validation and promotion |
| `reporting.py` | Sidecar decision/cost summary |
| `__init__.py` | Public interfaces |

Changed production files:

| File | Necessary change |
| --- | --- |
| `manager.py` | Remove old SkillSession hooks; add optional pre-dispatch hook and sidecar audit linkage |
| `types.py` | Remove old Skill options; add optional Reflector configuration |
| `role_prompts.py` | Remove unused old Skill argument/slots; preserve rendered original prompt bytes |
| `config.py` | Accept and validate optional project TOML section |
| `cli.py` | Replace old Skill flags with optional config, reuse Manager provider, print checker progress |

New tests: `tests/test_feasibility.py`. Existing `test_verifier_uv.py` retains the
original evaluator's uv checks after old Skill drivers were archived;
`test_offline_claude_setup.py` checks the same prerequisite against the surviving
OriginalHarness adapter. Old Skill-only tests and launchers are archived. The two
training READMEs now point to this replacement and mark old instructions historical.

Final targeted suite: **130 passed**, including **31 Reflector tests**. It includes
the original role prompts, Manager hardening, resume, final response, CLI, original
Harbor evaluation, uv and offline Claude setup regression tests. Localhost proxy
bypass and local socket permission were required for the existing Dashboard HTTP
test. No Dashboard implementation was changed.

Artifacts in `output/feasibility-implementation-validation-20261111/`:

- `combined-tests.xml`: actual final JUnit results.
- `verification.json`: counts and explicit real-model/benchmark limitations.
- `protected_integrity.json`: 1,001 protected pre-change hashes checked; only the
  intentional removal of inert main-role Skill slots differs. All benchmark source,
  evaluation prompts, datasets, scoring and original-eval drivers match before.
- `file_inventory.json`: complete created/modified/archived file list.
- `controlled_integration_fixture/FIXTURE.json`: labels deterministic test doubles.
  Its `logs/role_orchestration/rounds/round_001/feasibility.json` comes from an actual
  invocation of the original `manager.run` loop with the real retrieval/checker
  modules and controlled model/role replies: `FS_dependencies` → ACCEPT → one
  Executor dispatch → one Auditor call → linked audit. Environment I/O is a memory
  test double; remote trace mirrors are not part of this fixture. The fixture's
  token counts are synthetic and must not be reported as provider costs.

No real model API credentials were present, so no real paid-model task, learned
active bank, downstream benchmark improvement or end-to-end real training success
is claimed. The offline pipeline evaluates externally supplied controlled validation
outcomes; it does not itself schedule paired benchmark replays. Existing Original
Terminal-Bench runs remain entirely on their original evaluation path. Only the
explicit optional task-ID shadow launcher adds the observational Manager wrapper.
