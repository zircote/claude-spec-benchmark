# SDD-Bench: Spec-Driven Development Benchmarking Suite

## Project Brief

**Version:** 1.0.0-draft  
**Author:** Robert Allen <zircote@gmail.com>   
**Date:** December 15, 2025  
**Status:** Planning

---

## Executive Summary

SDD-Bench is a comprehensive benchmarking and evaluation framework for testing spec-driven development (SDD) tools, AI coding agents, and LLM-powered development frameworks. It extends SWE-bench's methodology to evaluate the complete software development lifecycle—from ambiguous requirements through elicitation, specification, test generation, and implementation.

Unlike SWE-bench which provides pre-written issue descriptions, SDD-Bench tests an agent's ability to:
1. **Elicit** requirements through Socratic dialogue
2. **Specify** testable criteria from discovered requirements
3. **Generate** tests that validate the specification (TDD "red" phase)
4. **Implement** code that satisfies those tests (TDD "green" phase)
5. **Refine** through iteration until gold tests pass

---

## Problem Statement

### Current Limitations

**SWE-bench Gap:** Existing benchmarks provide complete, well-specified issues. Real-world development begins with vague, incomplete, or contradictory requirements that must be discovered through stakeholder dialogue.

```
What SWE-bench Tests:
┌─────────────────────────┐     ┌────────────────┐
│ Complete Specification  │ ──► │ Implementation │
└─────────────────────────┘     └────────────────┘

What Real Development Requires:
┌────────────┐     ┌─────────────┐     ┌──────────┐     ┌─────────┐     ┌────────────┐
│ Vague Need │ ──► │ Elicitation │ ──► │ Spec     │ ──► │ Tests   │ ──► │ Code       │
└────────────┘     └─────────────┘     └──────────┘     └─────────┘     └────────────┘
```

**No Elicitation Evaluation:** No standard benchmark tests an agent's ability to ask clarifying questions, probe assumptions, or iteratively refine understanding.

**No TDD Pipeline Metrics:** Current benchmarks measure final output (patch resolves issue), not intermediate quality (test validity, spec completeness, elicitation efficiency).

### Target Users

- **AI Agent Developers:** Teams building Claude Code extensions, coding assistants, or autonomous development agents
- **Research Teams:** Academics studying LLM capabilities in software engineering
- **Enterprise Tool Teams:** Organizations evaluating SDD/TDD tooling for internal use
- **Framework Authors:** Developers of spec-driven development frameworks and methodologies

---

## Product Vision

### One-Liner

> The SWE-bench for spec-driven development—evaluate AI agents on the complete journey from "it's broken" to working code.

### Core Principles

1. **Full Pipeline Evaluation:** Test every phase from elicitation to implementation
2. **Degraded Specification Simulation:** Progressively remove information to test resilience
3. **Socratic Oracle System:** Simulate stakeholders who reveal information only when asked correctly
4. **Composable Metrics:** Independent scores for each phase that aggregate into end-to-end success
5. **SWE-bench Compatible:** Output predictions in standard format for cross-comparison

---

## Feature Specification

### F1: Specification Degradation Engine

**Purpose:** Create progressively vaguer versions of SWE-bench issues to simulate incomplete requirements.

**Degradation Levels:**

| Level | Name | Description | Example Transform |
|-------|------|-------------|-------------------|
| 0 | FULL | Original SWE-bench issue | No change |
| 1 | PARTIAL | Remove code snippets, stack traces, file paths | `File: django/db/models.py` → `[file]` |
| 2 | VAGUE | Keep only problem description | Multi-paragraph → first paragraph only |
| 3 | MINIMAL | Single sentence summary | Full issue → "The export function crashes" |
| 4 | AMBIGUOUS | Intentionally unclear/contradictory | "slow" → "not performing optimally" |

**API:**
```python
class SpecDegradationLevel(Enum):
    FULL = 0
    PARTIAL = 1
    VAGUE = 2
    MINIMAL = 3
    AMBIGUOUS = 4

def degrade_specification(
    full_issue: str,
    level: SpecDegradationLevel,
    seed: int | None = None,  # For reproducibility
) -> DegradedSpec:
    """Returns degraded text and list of hidden details."""
```

**Acceptance Criteria:**
- [ ] Deterministic output given same seed
- [ ] Hidden details tracked for scoring
- [ ] Preserves enough signal for elicitation to succeed
- [ ] Configurable degradation patterns per domain

---

### F2: Elicitation Oracle System

**Purpose:** Simulate a stakeholder who knows the full requirements but only reveals information when asked appropriate questions.

**Socratic Question Categories:**

| Category | Example | Effectiveness |
|----------|---------|---------------|
| Clarification | "What do you mean by 'slow'?" | High for ambiguous terms |
| Assumption | "Are you assuming single-threaded execution?" | High for implicit constraints |
| Evidence | "Can you show an example of the failure?" | High for reproduction |
| Viewpoint | "How would the API consumer see this?" | Medium for UX requirements |
| Implication | "What happens if we change X?" | High for edge cases |
| Meta | "Why is this the priority now?" | Low for technical details |

**Oracle Behavior:**
- High-relevance questions (>0.7): Reveal specific hidden requirements
- Medium-relevance questions (0.4-0.7): Provide hints, request clarification
- Low-relevance questions (<0.4): Vague responses, request rephrasing

**API:**
```python
class ElicitationOracle:
    def __init__(
        self,
        full_spec: str,
        hidden_requirements: list[str],
        response_model: str = "rules",  # or "llm" for dynamic responses
    ): ...
    
    def ask(self, question: str) -> OracleResponse:
        """Answer based on question quality and relevance."""
    
    def get_elicitation_score(self) -> ElicitationMetrics:
        """Return discovery rate, efficiency, question distribution."""
```

**Acceptance Criteria:**
- [ ] Deterministic scoring for question relevance
- [ ] Tracks all revealed vs hidden requirements
- [ ] Supports both rule-based and LLM-based response generation
- [ ] Configurable "stakeholder personality" (helpful, terse, confused)

---

### F3: Requirements Extractor

**Purpose:** Parse SWE-bench issues into discrete, testable requirements that become the oracle's hidden knowledge.

**Extraction Strategy:**
1. Use LLM to identify atomic requirements
2. Classify each as functional/non-functional
3. Tag with keywords for relevance matching
4. Validate extractability (can this be discovered via questions?)

**Output Format:**
```python
@dataclass
class Requirement:
    id: str
    text: str
    category: Literal["functional", "non-functional", "constraint"]
    keywords: list[str]
    discoverable: bool  # Can be elicited through questions
    source_span: tuple[int, int]  # Character range in original
```

**Acceptance Criteria:**
- [ ] Extracts 5-20 requirements per typical SWE-bench issue
- [ ] Requirements are atomic and testable
- [ ] Keywords enable relevance scoring
- [ ] Supports manual validation/override

---

### F4: Test Generation Evaluator (SWT-Bench Integration)

**Purpose:** Evaluate generated tests using fail-to-pass methodology.

**Key Metrics:**
- **Applicability (W):** % of tests that execute without error
- **Success Rate (S):** % of tests that fail before patch, pass after
- **F→P Rate:** Fail-to-pass transitions (the TDD goal)
- **Coverage Delta (Δ):** % of gold patch lines covered by tests

**Integration:**
- Use SWT-bench harness for Docker-based evaluation
- Support both generated tests and reproduction scripts
- Track coverage at statement and branch level

**API:**
```python
def evaluate_test_generation(
    generated_tests: str,
    task: SWEBenchTask,
    mode: Literal["unit_test", "reproduction_script"] = "unit_test",
) -> TestGenerationMetrics:
    """Run SWT-bench evaluation on generated tests."""
```

**Acceptance Criteria:**
- [ ] Compatible with SWT-bench harness
- [ ] Reports all standard SWT-bench metrics
- [ ] Supports incremental evaluation (single test at a time)
- [ ] Handles test generation failures gracefully

---

### F5: Implementation Evaluator (SWE-Bench Integration)

**Purpose:** Validate generated patches against gold tests.

**Integration:**
- Output predictions in SWE-bench JSONL format
- Support local Docker evaluation and Modal cloud
- Track resolution rate across degradation levels

**Output Format:**
```jsonl
{"instance_id": "django__django-16527", "model_name_or_path": "sdd-bench-v1", "model_patch": "diff --git a/..."}
```

**Acceptance Criteria:**
- [ ] 100% compatible with `swebench.harness.run_evaluation`
- [ ] Supports SWE-bench Lite, Verified, and full datasets
- [ ] Tracks metrics per degradation level
- [ ] Generates comparison reports across configurations

---

### F6: Full Pipeline Orchestrator

**Purpose:** Run complete SDD evaluation from degraded spec to validated implementation.

**Pipeline Phases:**
```
Phase 0: Degrade specification
    └── Input: Full SWE-bench issue
    └── Output: Degraded spec + hidden requirements

Phase 1: Elicitation dialogue
    └── Input: Degraded spec
    └── Output: Elicited spec, elicitation metrics

Phase 2: Spec parsing
    └── Input: Elicited spec
    └── Output: Structured requirements

Phase 3: Test generation (TDD Red)
    └── Input: Requirements
    └── Output: Tests, test validity metrics

Phase 4: Implementation (TDD Green)
    └── Input: Requirements + Tests
    └── Output: Patch, implementation metrics

Phase 5: Refinement (TDD Refactor)
    └── Input: Failing tests + Patch
    └── Output: Refined patch

Phase 6: Validation
    └── Input: Final patch
    └── Output: SWE-bench resolve status
```

**API:**
```python
class SDDBenchRunner:
    def __init__(
        self,
        framework: SDDFrameworkProtocol,
        dataset: str = "princeton-nlp/SWE-bench_Lite",
        degradation_level: SpecDegradationLevel = SpecDegradationLevel.VAGUE,
    ): ...
    
    async def run(
        self,
        limit: int | None = None,
        parallel: int = 1,
    ) -> SDDBenchResults:
        """Run full evaluation pipeline."""
    
    def generate_report(self) -> Path:
        """Generate HTML/Markdown report with visualizations."""
```

**Acceptance Criteria:**
- [ ] Supports async/parallel evaluation
- [ ] Checkpoint/resume for long runs
- [ ] Configurable phase skipping (e.g., skip elicitation)
- [ ] Structured logging for debugging

---

### F7: Framework Protocol Definition

**Purpose:** Define the interface that SDD frameworks must implement for evaluation.

**Protocol:**
```python
from typing import Protocol

class SDDFrameworkProtocol(Protocol):
    """Interface for spec-driven development frameworks."""
    
    # Elicitation phase
    def start_elicitation(self, initial_context: str) -> str:
        """Begin elicitation, return first question."""
        ...
    
    def process_response(self, stakeholder_response: str) -> str | None:
        """Process response, return next question or None if done."""
        ...
    
    def get_elicited_spec(self) -> str:
        """Return specification derived from elicitation."""
        ...
    
    # Specification phase
    def parse_spec(self, spec: str, repo_path: Path) -> list[str]:
        """Extract structured requirements from specification."""
        ...
    
    # Test generation phase (TDD Red)
    def generate_tests(
        self,
        spec: str,
        requirements: list[str],
        repo_path: Path,
    ) -> str:
        """Generate tests that encode the requirements."""
        ...
    
    # Implementation phase (TDD Green)
    def implement(
        self,
        spec: str,
        requirements: list[str],
        tests: str,
        repo_path: Path,
    ) -> str:
        """Generate implementation that passes the tests."""
        ...
    
    # Refinement phase (TDD Refactor)
    def refine(
        self,
        implementation: str,
        test_failures: list[str],
        repo_path: Path,
    ) -> str:
        """Refine implementation based on test failures."""
        ...
```

**Reference Implementations:**
- `sdd_bench.frameworks.passthrough` - Passes full spec directly (baseline)
- `sdd_bench.frameworks.claude_code` - Claude Code SDK integration
- `sdd_bench.frameworks.openai_agent` - OpenAI agent integration

**Acceptance Criteria:**
- [ ] Protocol is minimal but complete
- [ ] At least one reference implementation included
- [ ] Clear documentation with examples
- [ ] Type hints for IDE support

---

### F8: Metrics Dashboard and Reporting

**Purpose:** Visualize and compare evaluation results.

**Core Metrics:**

| Phase | Metric | Description | Target |
|-------|--------|-------------|--------|
| Elicitation | Discovery Rate | % of requirements discovered | >80% |
| Elicitation | Question Efficiency | Discoveries per question | >0.3 |
| Elicitation | Socratic Balance | Distribution across question types | Varied |
| Spec | Completeness | % of original spec captured | >85% |
| Spec | Accuracy | Semantic similarity to gold | >0.8 |
| Tests | Fail-to-Pass | % of valid TDD tests | >90% |
| Tests | Coverage Delta | % of gold patch covered | >70% |
| Implementation | Resolve Rate | Passes gold tests | Track |
| E2E | Success Rate | Vague input → working code | Track |

**Reports:**
- Per-instance detailed breakdown
- Aggregate by degradation level
- Aggregate by repository
- Comparison across framework configurations
- Trend analysis over time

**Output Formats:**
- Markdown report
- HTML interactive dashboard
- JSON for programmatic access
- CSV for spreadsheet analysis

**Acceptance Criteria:**
- [ ] All metrics computed automatically
- [ ] Interactive HTML report with filtering
- [ ] Export to multiple formats
- [ ] Comparison mode for A/B testing

---

## Technical Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────────────┐
│                          SDD-Bench CLI                              │
│  sdd-bench run --framework claude --dataset lite --degradation vague│
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Pipeline Orchestrator                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │ Degrader │→│ Oracle   │→│ Spec     │→│ Test     │→│ Impl     │  │
│  │          │ │          │ │ Parser   │ │ Evaluator│ │ Evaluator│  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
            ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
            │ SWE-bench   │ │ SWT-bench   │ │ Docker      │
            │ Harness     │ │ Harness     │ │ Runtime     │
            └─────────────┘ └─────────────┘ └─────────────┘
                                    │
                                    ▼
                          ┌─────────────────┐
                          │ Results Store   │
                          │ (SQLite/JSON)   │
                          └─────────────────┘
                                    │
                                    ▼
                          ┌─────────────────┐
                          │ Report Generator│
                          │ (HTML/MD/JSON)  │
                          └─────────────────┘
```

### Technology Stack

| Component | Technology | Rationale |
|-----------|------------|-----------|
| Language | Python 3.11+ | SWE-bench compatibility, ecosystem |
| Package Manager | uv | Fast, modern, your preference |
| CLI | Typer | Type hints, auto-completion |
| Data Validation | Pydantic | Schema enforcement |
| Database | SQLite | Simple, portable, no server |
| Async | asyncio | Parallel evaluation |
| Testing | pytest | Standard, fixtures |
| Linting | ruff | Fast, comprehensive |
| Type Checking | mypy | Strict mode |
| Docs | MkDocs | Markdown-based |

### Directory Structure

```
sdd-bench/
├── pyproject.toml
├── README.md
├── LICENSE
├── CLAUDE.md                    # Claude Code context
│
├── src/
│   └── sdd_bench/
│       ├── __init__.py
│       ├── cli.py               # Typer CLI entry point
│       ├── runner.py            # Pipeline orchestrator
│       │
│       ├── degradation/
│       │   ├── __init__.py
│       │   ├── engine.py        # Specification degradation
│       │   └── patterns.py      # Degradation patterns
│       │
│       ├── elicitation/
│       │   ├── __init__.py
│       │   ├── oracle.py        # Stakeholder simulation
│       │   ├── scoring.py       # Question relevance scoring
│       │   └── extraction.py    # Requirement extraction
│       │
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── tests.py         # SWT-bench integration
│       │   ├── implementation.py # SWE-bench integration
│       │   └── metrics.py       # Metrics computation
│       │
│       ├── frameworks/
│       │   ├── __init__.py
│       │   ├── protocol.py      # SDDFrameworkProtocol
│       │   ├── passthrough.py   # Baseline implementation
│       │   └── claude_code.py   # Claude Code integration
│       │
│       ├── reporting/
│       │   ├── __init__.py
│       │   ├── dashboard.py     # HTML report generation
│       │   └── export.py        # JSON/CSV/MD export
│       │
│       └── data/
│           ├── __init__.py
│           ├── store.py         # Results persistence
│           └── models.py        # Pydantic models
│
├── tests/
│   ├── conftest.py
│   ├── test_degradation.py
│   ├── test_elicitation.py
│   ├── test_evaluation.py
│   └── fixtures/
│       └── sample_issues.json
│
├── docs/
│   ├── index.md
│   ├── getting-started.md
│   ├── configuration.md
│   ├── framework-guide.md
│   └── metrics-reference.md
│
└── examples/
    ├── basic_evaluation.py
    ├── custom_framework.py
    └── batch_comparison.py
```

---

## Implementation Roadmap

### Phase 1: Foundation (Week 1-2)

**Goal:** Core infrastructure and baseline capability

- [ ] Project scaffolding with uv
- [ ] Pydantic models for all data structures
- [ ] Specification degradation engine (F1)
- [ ] Basic CLI structure
- [ ] Unit tests for degradation

**Deliverable:** Can degrade SWE-bench issues at all levels

### Phase 2: Elicitation (Week 3-4)

**Goal:** Complete elicitation simulation system

- [ ] Requirements extraction from issues (F3)
- [ ] Elicitation oracle with rule-based scoring (F2)
- [ ] Question classification (Socratic categories)
- [ ] Elicitation metrics computation
- [ ] Integration tests with sample dialogues

**Deliverable:** Can simulate elicitation dialogue and score quality

### Phase 3: Evaluation Integration (Week 5-6)

**Goal:** Connect to SWE-bench and SWT-bench harnesses

- [ ] SWT-bench test evaluation integration (F4)
- [ ] SWE-bench implementation evaluation integration (F5)
- [ ] Docker environment management
- [ ] Prediction output in standard format

**Deliverable:** Can evaluate tests and implementations against gold standard

### Phase 4: Pipeline & Frameworks (Week 7-8)

**Goal:** Complete end-to-end pipeline

- [ ] Full pipeline orchestrator (F6)
- [ ] Framework protocol definition (F7)
- [ ] Passthrough baseline framework
- [ ] Claude Code SDK framework integration
- [ ] Async/parallel evaluation support

**Deliverable:** Can run complete evaluation on any framework

### Phase 5: Reporting & Polish (Week 9-10)

**Goal:** Production-ready release

- [ ] Metrics dashboard (F8)
- [ ] HTML report generation
- [ ] Export to all formats
- [ ] Documentation site
- [ ] Example notebooks
- [ ] Performance optimization

**Deliverable:** v1.0.0 release

---

## Success Criteria

### MVP (Phase 3 Complete)

- [ ] Evaluate 10 SWE-bench Lite instances end-to-end
- [ ] Degradation works at all 5 levels
- [ ] Elicitation metrics computed correctly
- [ ] Output compatible with SWE-bench harness

### V1.0 (Phase 5 Complete)

- [ ] Full SWE-bench Lite evaluation in <4 hours
- [ ] At least 2 framework integrations
- [ ] Interactive HTML dashboard
- [ ] Published documentation
- [ ] >90% test coverage

### V1.x Goals

- [ ] LLM-based oracle responses (more realistic)
- [ ] Custom dataset support (beyond SWE-bench)
- [ ] Modal cloud evaluation integration
- [ ] Leaderboard submission support
- [ ] VS Code extension for local evaluation

---

## Risks and Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| SWE-bench harness changes | High | Medium | Pin versions, integration tests |
| Elicitation scoring subjectivity | Medium | High | Multiple scoring strategies, calibration |
| Docker resource requirements | Medium | Medium | Cloud evaluation option (Modal) |
| Framework integration complexity | Medium | Medium | Minimal protocol, good docs |
| LLM API costs for oracle | Low | Low | Rule-based default, LLM optional |

---

## Dependencies

### External Projects

| Dependency | Version | Purpose |
|------------|---------|---------|
| SWE-bench | latest | Base evaluation harness |
| SWT-bench | latest | Test generation evaluation |
| datasets | >=2.0 | HuggingFace dataset loading |
| docker | >=6.0 | Container management |
| anthropic | >=0.30 | Claude API (for LLM oracle) |

### System Requirements

- Python 3.11+
- Docker 24+
- 120GB+ disk space (for SWE-bench images)
- 16GB+ RAM
- 8+ CPU cores recommended

---

## Open Questions

1. **Oracle Response Strategy:** Should the default oracle be purely rule-based, or should we use an LLM to generate more natural responses? Trade-off: reproducibility vs realism.

2. **Degradation Reproducibility:** How do we ensure degradation is consistent across runs while still being realistic? Consider seeded randomization.

3. **Multi-language Support:** SWE-bench is Python-only. Should we plan for multi-language expansion, or focus purely on Python?

4. **Elicitation Budget:** What's the right max number of questions before evaluation times out? Need empirical data.

5. **Intermediate Checkpointing:** How granular should checkpoints be for resuming long evaluations?

---

## Appendix A: Example Evaluation Run

```bash
# Install
uv pip install sdd-bench

# Run evaluation on SWE-bench Lite with VAGUE degradation
sdd-bench run \
    --framework claude-code \
    --dataset princeton-nlp/SWE-bench_Lite \
    --degradation vague \
    --limit 50 \
    --parallel 4 \
    --output results/

# Generate report
sdd-bench report results/ --format html --output report.html

# Compare two configurations
sdd-bench compare results/baseline results/with-planning --output comparison.html
```

---

## Appendix B: Sample Metrics Output

```json
{
  "run_id": "sdd-bench-2025-12-15-001",
  "framework": "claude-code-v1",
  "dataset": "princeton-nlp/SWE-bench_Lite",
  "degradation_level": "VAGUE",
  "instances_evaluated": 50,
  
  "aggregate_metrics": {
    "elicitation": {
      "discovery_rate": 0.82,
      "question_efficiency": 0.34,
      "avg_questions_per_instance": 8.2,
      "question_distribution": {
        "clarification": 0.35,
        "assumption": 0.25,
        "evidence": 0.20,
        "implication": 0.15,
        "viewpoint": 0.05
      }
    },
    "specification": {
      "completeness": 0.87,
      "accuracy": 0.83
    },
    "test_generation": {
      "applicability": 0.91,
      "fail_to_pass_rate": 0.72,
      "coverage_delta": 0.68
    },
    "implementation": {
      "resolve_rate": 0.42,
      "passes_generated_tests": 0.78
    },
    "end_to_end": {
      "success_rate": 0.38
    }
  },
  
  "by_degradation_level": {
    "FULL": { "resolve_rate": 0.52 },
    "PARTIAL": { "resolve_rate": 0.48 },
    "VAGUE": { "resolve_rate": 0.42 },
    "MINIMAL": { "resolve_rate": 0.31 },
    "AMBIGUOUS": { "resolve_rate": 0.22 }
  }
}
```

---

## Appendix C: Related Work

- **SWE-bench:** Base benchmark for code generation from issues
- **SWT-bench:** Test generation benchmark (NeurIPS 2024)
- **TDD-Bench Verified:** High-quality TDD benchmark with coverage metrics
- **Elicitron:** LLM-based requirements elicitation simulation
- **LLMREI:** Automated requirements elicitation interviews

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0-draft | 2025-12-15 | Bob | Initial product brief |