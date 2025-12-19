---
document_type: decisions
project_id: SPEC-2025-12-16-001
---

# SDD-Bench - Architecture Decision Records

## ADR-001: Extend Existing Package vs New Package

**Date**: 2025-12-16
**Status**: Accepted
**Deciders**: zircote, planning session

### Context

The existing `claude_spec_benchmark` package provides SWE-bench evaluation infrastructure. The project brief originally specified a separate `sdd_bench` package with different structure.

### Decision

Extend the existing `claude_spec_benchmark` package with new subpackages rather than creating a separate package.

### Consequences

**Positive:**
- Reuse existing infrastructure (Docker, evaluation, metrics)
- Single import namespace for users
- Shared data models avoid duplication
- Existing tests cover foundation

**Negative:**
- Package name doesn't match "SDD-Bench" marketing name
- Must maintain backward compatibility with existing APIs
- pyproject.toml already configured for current structure

**Neutral:**
- CLI command remains `claude-spec-benchmark`
- "SDD-Bench" becomes product/marketing name only

### Alternatives Considered

1. **Create separate `sdd_bench` package**: Rejected - would duplicate infrastructure and create integration complexity
2. **Rename to `sdd_bench`**: Rejected - user prefers keeping existing name
3. **`sdd_bench` as wrapper package**: Rejected - unnecessary indirection

---

## ADR-002: Rules-First Oracle vs LLM-First Oracle

**Date**: 2025-12-16
**Status**: Accepted
**Deciders**: zircote, planning session

### Context

The Elicitation Oracle can generate responses using either rule-based logic (deterministic) or LLM inference (more natural but variable).

### Decision

Implement rules-first oracle as the default and primary mode. LLM mode becomes a v1.x enhancement.

### Consequences

**Positive:**
- Reproducible benchmarks (same question → same score)
- No API costs for evaluation
- Faster execution (no LLM latency)
- Easier testing and debugging

**Negative:**
- Less realistic stakeholder simulation
- May not capture nuanced question quality
- Harder to handle novel question phrasings

**Neutral:**
- LLM mode can be added later without breaking interface

### Alternatives Considered

1. **LLM-first**: Rejected - reproducibility is critical for benchmarking
2. **Hybrid default**: Rejected - adds complexity without clear benefit
3. **Both equally prioritized**: Rejected - scope creep, rules sufficient for v1

---

## ADR-003: Subpackage Organization

**Date**: 2025-12-16
**Status**: Accepted
**Deciders**: zircote, planning session

### Context

New SDD modules (degradation, elicitation, frameworks, reporting) need to be organized within the package.

### Decision

Use subpackage structure: `degradation/`, `elicitation/`, `frameworks/`, `reporting/` directories with `__init__.py` files.

### Consequences

**Positive:**
- Clear separation of concerns
- Easier navigation and discovery
- Supports independent testing
- Matches project brief structure

**Negative:**
- More files to manage
- Deeper import paths
- Must update `__init__.py` exports

**Neutral:**
- Existing flat files (`runner.py`, `evaluator.py`) remain at package root

### Alternatives Considered

1. **Flat structure**: Rejected - would create many files at root level
2. **Single `sdd.py` module**: Rejected - doesn't scale, poor organization
3. **Match brief exactly**: Current choice largely matches brief

---

## ADR-004: Local Docker Only for v1

**Date**: 2025-12-16
**Status**: Accepted
**Deciders**: zircote, planning session

### Context

SWE-bench evaluation requires Docker containers (120GB+ disk for images). Modal cloud offers alternative but adds complexity.

### Decision

Support only local Docker for v1. Modal cloud integration deferred to v1.x.

### Consequences

**Positive:**
- Simpler architecture
- No cloud account requirements
- Full offline capability
- Easier debugging

**Negative:**
- Requires beefy local machine (120GB+ disk, 16GB+ RAM)
- Cannot scale to large parallel runs
- Users without Docker cannot evaluate

**Neutral:**
- SWE-bench harness handles Docker complexity
- Existing `DockerManager` class sufficient

### Alternatives Considered

1. **Both required**: Rejected - doubles implementation scope
2. **Modal-first**: Rejected - adds external dependency for basic use
3. **Defer Docker entirely**: Not viable - SWE-bench requires Docker

---

## ADR-005: CLI Framework Migration (Click → Typer)

**Date**: 2025-12-16
**Status**: Proposed
**Deciders**: Architecture review

### Context

The existing codebase uses Click for CLI. The project brief specifies Typer. Both are popular, with Typer being newer and type-hint based.

### Decision

Migrate from Click to Typer for new SDD commands. Maintain backward compatibility for existing commands during transition.

### Consequences

**Positive:**
- Type hints provide better IDE support
- Auto-completion out of the box
- Modern, actively maintained
- Brief compliance

**Negative:**
- Migration effort for existing commands
- Two CLI frameworks during transition
- Team must learn Typer patterns

**Neutral:**
- Both libraries by same author (tiangolo)
- Click commands can coexist with Typer

### Alternatives Considered

1. **Keep Click**: Simpler but doesn't match brief
2. **Full immediate migration**: Risky, could break existing users
3. **New CLI binary**: Creates user confusion

---

## ADR-006: Elicitation Question Scoring Algorithm

**Date**: 2025-12-16
**Status**: Proposed
**Deciders**: Architecture review

### Context

The oracle must score question relevance (0.0-1.0) to determine what information to reveal. Multiple scoring approaches are possible.

### Decision

Use keyword-based TF-IDF scoring with category bonuses. Implementation:

1. Extract keywords from question (stopword removal, stemming)
2. Match against requirement keyword index
3. Compute TF-IDF similarity score
4. Apply category bonus (clarification: +0.1, meta: -0.1)
5. Apply personality modifier

### Consequences

**Positive:**
- Deterministic and explainable
- No external dependencies beyond NLTK/sklearn
- Fast execution
- Testable with known inputs

**Negative:**
- May miss semantic similarity (synonyms)
- Keyword extraction quality varies
- Category classification is heuristic

**Neutral:**
- Can be enhanced with embeddings in v1.x
- Calibration needed with real data

### Alternatives Considered

1. **Embedding similarity**: More accurate but requires model, slower
2. **LLM-as-judge**: Non-deterministic, API costs
3. **Simple keyword overlap**: Too simplistic, poor discrimination

---

## ADR-007: Checkpoint/Resume Strategy

**Date**: 2025-12-16
**Status**: Proposed
**Deciders**: Architecture review

### Context

Full SDD evaluation on SWE-bench Lite (300 instances) takes 2-8 hours. Failures should not lose all progress.

### Decision

Checkpoint at phase granularity per instance. Store checkpoints as JSON files in output directory.

```
results/
├── checkpoints/
│   ├── django__django-12345.json    # Phase results so far
│   └── ...
└── final/
    └── results.jsonl                 # Complete results
```

### Consequences

**Positive:**
- Resume from last completed phase on failure
- Partial results available during run
- Simple file-based storage

**Negative:**
- Disk I/O per phase completion
- Must handle checkpoint schema evolution
- Orphan checkpoints need cleanup

**Neutral:**
- SQLite alternative considered but adds complexity
- JSON is human-readable for debugging

### Alternatives Considered

1. **Per-instance only**: Coarser, loses more progress on failure
2. **In-memory only**: No resume capability
3. **SQLite database**: More robust but harder to debug/inspect

---

## ADR-008: Test Generation Evaluation Strategy

**Date**: 2025-12-16
**Status**: Proposed
**Deciders**: Architecture review

### Context

Need to evaluate generated tests using fail-to-pass methodology (SWT-bench). Options: integrate SWT-bench harness directly, or build custom evaluation.

### Decision

Integrate with SWT-bench harness for test evaluation. Use their Docker-based methodology and metrics.

### Consequences

**Positive:**
- Standardized metrics (Applicability, Success Rate, F→P)
- Proven evaluation methodology
- Academic credibility (NeurIPS 2024)

**Negative:**
- External dependency
- Must match SWT-bench output format
- Less control over evaluation details

**Neutral:**
- SWT-bench is open source, can fork if needed
- Metrics well-documented

### Alternatives Considered

1. **Custom evaluation**: Full control but duplicates work, less credible
2. **Skip test evaluation**: Incomplete pipeline
3. **Mock/stub evaluation**: Only for development, not production

---

## ADR-009: Framework Protocol Design Philosophy

**Date**: 2025-12-16
**Status**: Accepted
**Deciders**: Architecture review

### Context

The `SDDFrameworkProtocol` defines what frameworks must implement. Trade-off between minimal interface (easy adoption) and comprehensive interface (better benchmarking).

### Decision

Minimal protocol with 7 methods covering the essential SDD phases. Methods can return empty/default values if phase not supported.

```python
class SDDFrameworkProtocol(Protocol):
    # Elicitation (3 methods)
    def start_elicitation(self, initial_context: str) -> str: ...
    def process_response(self, stakeholder_response: str) -> str | None: ...
    def get_elicited_spec(self) -> str: ...

    # Specification (1 method)
    def parse_spec(self, spec: str, repo_path: Path) -> list[str]: ...

    # TDD (3 methods)
    def generate_tests(self, spec: str, requirements: list[str], repo_path: Path) -> str: ...
    def implement(self, spec: str, requirements: list[str], tests: str, repo_path: Path) -> str: ...
    def refine(self, implementation: str, test_failures: list[str], repo_path: Path) -> str: ...
```

### Consequences

**Positive:**
- Low barrier to entry for framework authors
- Clear phase separation
- Type hints enable static checking
- Passthrough implementation trivial

**Negative:**
- May not capture all framework capabilities
- Some frameworks may not fit model perfectly
- No async methods (could limit some implementations)

**Neutral:**
- Protocol can be extended in v1.x
- Optional methods via default implementations

### Alternatives Considered

1. **Larger protocol (15+ methods)**: Higher barrier, better coverage
2. **Single `evaluate()` method**: Too opaque, no phase metrics
3. **Abstract base class**: Less flexible than Protocol

---

## ADR-010: Python Version Requirement (3.14+)

**Date**: 2025-12-16
**Status**: Inherited
**Deciders**: Original project setup

### Context

The existing pyproject.toml specifies `requires-python = ">=3.14"`. This is a very recent Python version.

### Decision

Maintain Python 3.14+ requirement per existing configuration.

### Consequences

**Positive:**
- Access to latest language features
- Pattern matching, improved typing
- Aligns with forward-looking project

**Negative:**
- Limited user base (3.14 just released)
- Some dependencies may not support yet
- CI/CD tooling may need updates

**Neutral:**
- Can revisit if adoption issues arise
- Most features work on 3.11+ anyway

### Alternatives Considered

1. **Lower to 3.11+**: Wider compatibility but loses modern features
2. **Dual support**: Complexity not justified
