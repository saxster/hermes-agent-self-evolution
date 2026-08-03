# Graph Report - hermes-agent-self-evolution  (2026-06-10)

## Corpus Check
- 48 files · ~44,070 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1003 nodes · 2312 edges · 55 communities (47 shown, 8 thin omitted)
- Extraction: 79% EXTRACTED · 21% INFERRED · 0% AMBIGUOUS · INFERRED: 486 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `1d673b8a`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 40|Community 40]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 45|Community 45]]
- [[_COMMUNITY_Community 46|Community 46]]
- [[_COMMUNITY_Community 47|Community 47]]
- [[_COMMUNITY_Community 48|Community 48]]
- [[_COMMUNITY_Community 49|Community 49]]
- [[_COMMUNITY_Community 50|Community 50]]

## God Nodes (most connected - your core abstractions)
1. `EvolutionConfig` - 83 edges
2. `EvalDataset` - 65 edges
3. `DiagnosisAgent` - 60 edges
4. `TraceWriter` - 60 edges
5. `ConstraintValidator` - 51 edges
6. `EvalExample` - 49 edges
7. `GoldenDatasetLoader` - 47 edges
8. `TestSecretDetection` - 39 edges
9. `_contains_secret()` - 38 edges
10. `SyntheticDatasetBuilder` - 32 edges

## Surprising Connections (you probably didn't know these)
- `TestGrowthConstraints` --uses--> `EvolutionConfig`  [INFERRED]
  tests/core/test_constraints.py → evolution/core/config.py
- `TestNonEmpty` --uses--> `EvolutionConfig`  [INFERRED]
  tests/core/test_constraints.py → evolution/core/config.py
- `TestSizeConstraints` --uses--> `EvolutionConfig`  [INFERRED]
  tests/core/test_constraints.py → evolution/core/config.py
- `TestSkillStructure` --uses--> `EvolutionConfig`  [INFERRED]
  tests/core/test_constraints.py → evolution/core/config.py
- `TestValidateAll` --uses--> `EvolutionConfig`  [INFERRED]
  tests/core/test_constraints.py → evolution/core/config.py

## Import Cycles
- None detected.

## Communities (55 total, 8 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.06
Nodes (67): Any, bool, float, int, Path, str, float, Path (+59 more)

### Community 1 - "Community 1"
Cohesion: 0.06
Nodes (32): compute_weighted_fitness(), _conciseness_score(), _correctness_score(), _parse_score(), Proxy for "follows expected procedure" — how structured is the output?      Heur, Proxy for "appropriately concise" — penalize outputs drastically longer     than, Keyword overlap proxy — how many expected_behavior words appear in the output., Compute a weighted fitness score from agent output + rubric + weights.      Shar (+24 more)

### Community 2 - "Community 2"
Cohesion: 0.07
Nodes (49): bool, int, Path, str, bool, float, int, compare() (+41 more)

### Community 3 - "Community 3"
Cohesion: 0.08
Nodes (34): CodeOrganism, Manages code variants as git branches for safe mutation + validation.  A CodeOrg, Run pytest on a variant branch and collect pass rate.          Args:, Remove all evolution branches created during this run.          Returns:, Check that the code parses as valid Python., Run a git command in the hermes-agent repo., Get the name of the current git branch., Extract pass rate from pytest output.          Parses lines like '5 passed, 2 fa (+26 more)

### Community 4 - "Community 4"
Cohesion: 0.05
Nodes (37): 1. Full Test Suite, 2. Character/Token Limits, 3. Prompt Caching Compatibility, 4. Semantic Preservation, 5. Deployment via PR (Never Direct Commit), Architecture, Benchmarks as Fitness Signals, Constraints & Guardrails (+29 more)

### Community 5 - "Community 5"
Cohesion: 0.13
Nodes (33): FitnessScore, LLMJudge, Multi-dimensional fitness score., LLM-as-judge scorer with rubric-based evaluation.      Scores agent outputs on m, get_signal_enhanced_fitness_weight(), Get signal-informed fitness weights for a skill.      When real signals show tha, bool, EvalDataset (+25 more)

### Community 6 - "Community 6"
Cohesion: 0.11
Nodes (4): _contains_secret(), Check if text contains potential API keys or tokens., Verify that known secret formats are caught and normal text is not., TestSecretDetection

### Community 7 - "Community 7"
Cohesion: 0.10
Nodes (25): bool, EvolutionQueue, int, str, main(), parse_interval(), post_to_mission_control(), process_one() (+17 more)

### Community 8 - "Community 8"
Cohesion: 0.09
Nodes (21): EvolutionConfig, get_hermes_agent_path(), Configuration and hermes-agent repo discovery., Configuration for a self-evolution optimization run., Discover the hermes-agent repo path.      Priority:     1. HERMES_AGENT_REPO env, _extract_json_array(), GenerateTestCases, Evaluation dataset generation for hermes-agent-self-evolution.  Sources: A) Synt (+13 more)

### Community 9 - "Community 9"
Cohesion: 0.14
Nodes (25): Path, Meta-harness extensions for hermes-agent-self-evolution.  Implements the filesys, get_active_lessons(), load_lessons_from_path(), Per-task trace writer for GEPA optimization runs.  The TraceWriter persists ever, Install the lessons text that DSPy modules will read in forward()., Return the current active lessons text, or empty string if none., Read a lessons.md file and install it as active lessons.      Returns the loaded (+17 more)

### Community 10 - "Community 10"
Cohesion: 0.12
Nodes (25): bool, Path, str, _build_paren_string(), _escape_for_python(), _eval_concatenated_strings(), _extract_string_constant(), load_prompt_sections() (+17 more)

### Community 11 - "Community 11"
Cohesion: 0.15
Nodes (23): bool, EvolutionQueue, float, int, str, Path, check_skill_quality(), main() (+15 more)

### Community 12 - "Community 12"
Cohesion: 0.11
Nodes (14): Path, Prediction, str, find_skill(), load_skill(), Wraps a SKILL.md file as a DSPy module for optimization.  The key abstraction: a, Read the current skill body text from the predictor's signature.          After, Reassemble a skill file from frontmatter and evolved body.      Preserves the or (+6 more)

### Community 13 - "Community 13"
Cohesion: 0.13
Nodes (24): ConstraintResult, bool, bool, EvolutionConfig, int, str, Return True if tracing is opted in via the env var., tracing_enabled() (+16 more)

### Community 14 - "Community 14"
Cohesion: 0.08
Nodes (7): Tests for constraint validators., TestGrowthConstraints, TestNonEmpty, TestSizeConstraints, TestSkillStructure, TestValidateAll, validator()

### Community 15 - "Community 15"
Cohesion: 0.13
Nodes (16): Any, float, int, Path, str, _atomic_write_text(), _get(), _hash() (+8 more)

### Community 16 - "Community 16"
Cohesion: 0.16
Nodes (4): Validate and normalize fields before creating an EvalExample.      Returns:, _validate_eval_example(), Verify _validate_eval_example normalizes, rejects, and caps fields., TestValidateEvalExample

### Community 17 - "Community 17"
Cohesion: 0.17
Nodes (20): Path, str, Path, current_scope(), Path-scoped file-tool helpers for the meta-harness diagnosis agent.  The diagnos, Context manager that sets HERMES_READ_SAFE_ROOT for the block.      Yields the r, Return the currently-active scope path, or None if unset., scoped_reads() (+12 more)

### Community 18 - "Community 18"
Cohesion: 0.17
Nodes (20): Path, str, Install (or clear) the active writer.      Process-global by design — DSPy's Par, set_active_writer(), PromptSectionModule, A DSPy module that wraps one system prompt section for optimization.      **Desi, _fake_predictor(), _FakePredictor (+12 more)

### Community 19 - "Community 19"
Cohesion: 0.26
Nodes (22): Path, make_tracing_metric(), Wrap a DSPy-compatible fitness metric so each call logs a trace.      The wrappe, Persist per-task evaluation traces to an archive directory.      Usage:, TraceWriter, _make_example(), _make_prediction(), Tests for evolution.meta_harness.trace_writer. (+14 more)

### Community 20 - "Community 20"
Cohesion: 0.15
Nodes (13): bool, int, str, EvolutionQueue, Check if a target is already pending or in progress., Mark an in-progress item as completed or failed.          Returns True if the it, Remove completed/failed items older than max_age_hours (default 7 days)., Check if a target was completed recently (within cooldown window). (+5 more)

### Community 21 - "Community 21"
Cohesion: 0.15
Nodes (19): bool, Path, str, _escape_for_replacement(), _extract_description_from_register(), _extract_description_from_schema_var(), _find_schema_var_for_tool(), find_tool_file() (+11 more)

### Community 22 - "Community 22"
Cohesion: 0.19
Nodes (11): ConstraintResult, ConstraintValidator, Constraint validators for evolved artifacts.  Every candidate variant must pass, Check that a skill file has valid YAML frontmatter and markdown body., Result of constraint validation., Validates evolved artifacts against hard constraints., Run all applicable constraints. Returns list of results., Run the full hermes-agent test suite. Must pass 100%. (+3 more)

### Community 23 - "Community 23"
Cohesion: 0.12
Nodes (10): ClaudeCodeImporter, HermesSessionImporter, Import user prompts from Claude Code history.jsonl.      Claude Code stores a fl, Read user messages from Claude Code history.          Args:             limit: M, Import conversations from Hermes Agent session files.      Hermes stores session, Read user/assistant pairs from Hermes session files.          Args:, Verify output matches GoldenDatasetLoader expected format., TestClaudeCodeImporter (+2 more)

### Community 24 - "Community 24"
Cohesion: 0.14
Nodes (11): build_dataset_from_external(), Extract messages from external tools, filter for relevance, and save.      This, Test the main orchestration function., Verify end-to-end: import -> filter -> split -> save., An unrecognized source name is silently skipped., Verify the full pipeline: fake files -> import -> filter -> save -> reload., Write fake Claude Code history, run pipeline, load with GoldenDatasetLoader., Create real Copilot events, import, filter (mocked), save, reload. (+3 more)

### Community 25 - "Community 25"
Cohesion: 0.16
Nodes (11): bool, float, int, str, Score all skills that have been used in the given window.          Returns a dic, Query state.db for sessions where skill was active.          Returns {session_id, Get names of all skills used since cutoff timestamp., Detect if the user corrected the agent in this session. (+3 more)

### Community 26 - "Community 26"
Cohesion: 0.20
Nodes (14): Generate evaluation datasets using a strong LLM.      Reads the target artifact, SyntheticDatasetBuilder, EvolutionConfig, bool, int, str, evolve(), main() (+6 more)

### Community 27 - "Community 27"
Cohesion: 0.19
Nodes (7): _parse_scoring_json(), Extract a JSON object from LLM scoring output.      Strategy:       1. Try direc, Verify _parse_scoring_json handles various LLM output formats., Clean JSON should be parsed directly without regex fallback., A JSON array or string should return None (we need a dict)., JSON with braces inside string values must parse correctly., TestScoringJsonParser

### Community 28 - "Community 28"
Cohesion: 0.18
Nodes (8): _parse_copilot_events(), Read user/assistant message pairs from Copilot sessions.          Args:, Extract cwd from a Copilot workspace.yaml file., Parse a single Copilot events.jsonl into user/assistant pairs., _read_copilot_workspace(), Outer try-catch: file doesn't exist -> returns empty list, doesn't crash., Outer try-catch: unreadable file -> returns empty list, doesn't crash., TestCopilotHelpers

### Community 29 - "Community 29"
Cohesion: 0.19
Nodes (7): _is_relevant_to_skill(), Quick heuristic check if a message might be relevant to a skill.      Uses keywo, Verify cheap pre-filter catches obvious matches and rejects non-matches., Verify that short skill names match via exact full-name check., TestRelevanceHeuristics, TestSkillNameMatching, bool

### Community 30 - "Community 30"
Cohesion: 0.20
Nodes (10): EvalExample, A single evaluation example., Import session data from external AI tools into golden eval datasets.  Bridges t, Use LLM-as-judge to determine which messages are relevant to a skill.      Two-s, Score whether a user message is relevant to a specific agent skill.          Ret, Filter messages by relevance and generate eval examples.          Args:, RelevanceFilter, ScoreRelevance (+2 more)

### Community 31 - "Community 31"
Cohesion: 0.26
Nodes (7): _load_skill_text(), main(), Load skill text from the installed Hermes skills directory.      This is used by, Import external session data into golden eval datasets for self-evolution., Tests for external session importers.  Tests cover:   - Secret detection and fil, TestLoadSkillText, Path

### Community 32 - "Community 32"
Cohesion: 0.21
Nodes (11): _get_hermes_db(), get_skill_performance_summary(), get_skill_signals(), get_weakest_skills(), Import implicit signal data from Hermes state.db for skill evolution.  Bridges t, Get a SessionDB instance from the hermes-agent codebase., Retrieve implicit signals for a specific skill.      Returns list of signal dict, Get performance summary for all skills from implicit signals.      Returns dict (+3 more)

### Community 33 - "Community 33"
Cohesion: 0.24
Nodes (6): EvalDataset, Train/val/holdout split of evaluation examples., Save dataset splits to JSONL files., Load dataset splits from JSONL files., Load a golden dataset. If no splits exist, auto-split the single file., Path

### Community 34 - "Community 34"
Cohesion: 0.18
Nodes (7): CopilotImporter, Import conversations from GitHub Copilot session events.      Copilot stores ses, Verify MIN_DATASET_SIZE warning in build_dataset_from_external., Even with < MIN_DATASET_SIZE examples, a dataset is still returned., Test the Click CLI entry point using CliRunner., TestCLI, TestMinDatasetSizeWarning

### Community 35 - "Community 35"
Cohesion: 0.18
Nodes (6): Verify validation is wired correctly into RelevanceFilter., LLM returns relevant=True but empty expected_behavior -> example dropped., LLM returns invalid difficulty -> normalized to medium., Messages missing 'source' key are dropped before scoring., Messages missing 'task_input' are dropped before scoring., TestValidationIntegration

### Community 36 - "Community 36"
Cohesion: 0.22
Nodes (3): A user message with no following assistant response is dropped., Multiple assistant.message events concatenate into one response., TestCopilotImporter

### Community 37 - "Community 37"
Cohesion: 0.22
Nodes (3): Test RelevanceFilter with mocked LLM calls., Mock dspy.LM and dspy.context to avoid real LLM calls., TestRelevanceFilter

### Community 38 - "Community 38"
Cohesion: 0.22
Nodes (8): Engines, Full Plan, Guardrails, 🧬 Hermes Agent Self-Evolution, How It Works, License, Quick Start, What It Optimizes

### Community 40 - "Community 40"
Cohesion: 0.38
Nodes (7): GoldenDatasetLoader, Load hand-curated evaluation datasets from JSONL files., Example, float, Prediction, prompt_section_fitness(), DSPy-compatible metric for prompt section optimization.      Composes three sign

### Community 41 - "Community 41"
Cohesion: 0.29
Nodes (3): float, Path, Load queue state from disk.

### Community 42 - "Community 42"
Cohesion: 0.33
Nodes (4): Prediction, Prediction, get_active_writer(), Return the currently-active writer, or None.

### Community 43 - "Community 43"
Cohesion: 0.50
Nodes (3): build_report(), str, Generate the Phase 1 validation report as PDF.

## Knowledge Gaps
- **50 isolated node(s):** `Path`, `Any`, `str`, `bool`, `bool` (+45 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **8 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `EvolutionConfig` connect `Community 8` to `Community 33`, `Community 1`, `Community 3`, `Community 5`, `Community 40`, `Community 13`, `Community 14`, `Community 22`, `Community 26`, `Community 30`?**
  _High betweenness centrality (0.218) - this node is a cross-community bridge._
- **Why does `DiagnosisAgent` connect `Community 0` to `Community 40`, `Community 9`, `Community 5`, `Community 13`?**
  _High betweenness centrality (0.163) - this node is a cross-community bridge._
- **Why does `EvalDataset` connect `Community 33` to `Community 5`, `Community 6`, `Community 8`, `Community 13`, `Community 16`, `Community 23`, `Community 24`, `Community 26`, `Community 27`, `Community 28`, `Community 29`, `Community 30`, `Community 31`, `Community 34`, `Community 35`, `Community 36`, `Community 37`, `Community 39`, `Community 40`?**
  _High betweenness centrality (0.123) - this node is a cross-community bridge._
- **Are the 65 inferred relationships involving `EvolutionConfig` (e.g. with `CodeOrganism` and `VariantResult`) actually correct?**
  _`EvolutionConfig` has 65 INFERRED edges - model-reasoned connections that need verification._
- **Are the 51 inferred relationships involving `EvalDataset` (e.g. with `ConstraintResult` and `EvolutionConfig`) actually correct?**
  _`EvalDataset` has 51 INFERRED edges - model-reasoned connections that need verification._
- **Are the 23 inferred relationships involving `DiagnosisAgent` (e.g. with `ConstraintResult` and `float`) actually correct?**
  _`DiagnosisAgent` has 23 INFERRED edges - model-reasoned connections that need verification._
- **Are the 24 inferred relationships involving `TraceWriter` (e.g. with `ConstraintResult` and `Path`) actually correct?**
  _`TraceWriter` has 24 INFERRED edges - model-reasoned connections that need verification._