---
document_type: requirements
project_id: SPEC-2025-12-16-001
version: 0.1.0
last_updated: 2025-12-16T00:00:00Z
status: draft
---

# SDD-Bench: Spec-Driven Development Benchmarking Suite

## Product Requirements Document

### Executive Summary

SDD-Bench is a comprehensive benchmarking and evaluation framework for testing spec-driven development (SDD) tools, AI coding agents, and LLM-powered development frameworks. It extends SWE-bench's methodology to evaluate the complete software development lifecycle—from ambiguous requirements through elicitation, specification, test generation, and implementation.

Unlike SWE-bench which provides pre-written issue descriptions, SDD-Bench tests an agent's ability to:
1. **Elicit** requirements through Socratic dialogue
2. **Specify** testable criteria from discovered requirements
3. **Generate** tests that validate the specification (TDD "red" phase)
4. **Implement** code that satisfies those tests (TDD "green" phase)
5. **Refine** through iteration until gold tests pass

### Problem Statement

#### The Problem

Existing benchmarks (SWE-bench, HumanEval) provide complete, well-specified issues. Real-world development begins with vague, incomplete, or contradictory requirements that must be discovered through stakeholder dialogue.

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

#### Impact

- **AI Agent Developers**: No way to measure elicitation capability
- **Research Teams**: Cannot study requirements discovery in LLMs
- **Enterprise Teams**: Cannot compare SDD tooling objectively
- **Framework Authors**: No standard benchmark for TDD pipeline quality

#### Current State

Users must manually evaluate agent elicitation capability, which is subjective, time-consuming, and non-reproducible.

### Goals and Success Criteria

#### Primary Goal

Enable quantitative, reproducible evaluation of AI agents on the complete spec-driven development pipeline.

#### Success Metrics

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| Feature Completeness | 8/8 features (F1-F8) | Implementation coverage |
| SWE-bench Compatibility | 100% | Output passes `swebench.harness.run_evaluation` |
| Evaluation Reproducibility | Deterministic | Same inputs → same scores |
| Evaluation Performance | <5min per instance | Average on SWE-bench Lite |
| Test Coverage | >80% | pytest-cov |

#### Non-Goals (Explicit Exclusions)

- Multi-language support (Python only, per SWE-bench)
- Cloud evaluation (Modal integration deferred to v1.x)
- LLM-based oracle responses (rules-first for reproducibility)
- Real-time web dashboard (static reports only)

### User Analysis

#### Primary Users

**AI Agent Developers**
- **Needs**: Quantitative metrics to compare agent versions, identify elicitation weaknesses
- **Context**: CI/CD pipelines, A/B testing, regression testing

**Research Teams**
- **Needs**: Reproducible benchmarks for academic papers, ablation studies
- **Context**: Academic computing environments, HPC clusters

**Enterprise Tool Teams**
- **Needs**: Objective comparison of SDD tooling options
- **Context**: Procurement decisions, internal tool evaluation

#### User Stories

1. As an **AI agent developer**, I want to run SDD-Bench on my agent so that I can measure its elicitation capability quantitatively.

2. As a **researcher**, I want deterministic degradation of issues so that I can study the relationship between information completeness and resolution rate.

3. As an **enterprise evaluator**, I want to compare multiple frameworks on the same benchmark so that I can make data-driven tooling decisions.

4. As a **framework author**, I want to implement the SDDFrameworkProtocol so that my framework can be evaluated against others.

---

## Functional Requirements

### F1: Specification Degradation Engine (P0)

**Purpose**: Create progressively vaguer versions of SWE-bench issues to simulate incomplete requirements.

| ID | Requirement | Rationale | Acceptance Criteria |
|----|-------------|-----------|---------------------|
| FR-001 | Support 5 degradation levels (FULL, PARTIAL, VAGUE, MINIMAL, AMBIGUOUS) | Match spec levels from brief | All levels produce distinct outputs |
| FR-002 | Deterministic output given same seed | Reproducible benchmarks | `degrade(issue, level, seed=42) == degrade(issue, level, seed=42)` |
| FR-003 | Track hidden details for scoring | Enable discovery metrics | Hidden details list returned with degraded spec |
| FR-004 | Configurable degradation patterns | Different repos may need different patterns | Pattern config per repository/domain |

**API Signature** (from brief):
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
    seed: int | None = None,
) -> DegradedSpec:
    """Returns degraded text and list of hidden details."""
```

### F2: Elicitation Oracle System (P0)

**Purpose**: Simulate a stakeholder who knows full requirements but reveals information only when asked appropriate questions.

| ID | Requirement | Rationale | Acceptance Criteria |
|----|-------------|-----------|---------------------|
| FR-101 | Rule-based response generation (default) | Reproducibility | Same question → same relevance score |
| FR-102 | Question relevance scoring (0.0-1.0) | Quantify question quality | Score correlates with requirement discovery |
| FR-103 | Support 6 Socratic question categories | Measure question diversity | Categories: clarification, assumption, evidence, viewpoint, implication, meta |
| FR-104 | Track revealed vs hidden requirements | Enable discovery metrics | `oracle.get_elicitation_score()` returns complete breakdown |
| FR-105 | Configurable stakeholder personality | Test agent robustness | Personalities: helpful, terse, confused |

**Oracle Behavior** (from brief):
- High-relevance (>0.7): Reveal specific hidden requirements
- Medium-relevance (0.4-0.7): Provide hints, request clarification
- Low-relevance (<0.4): Vague responses, request rephrasing

### F3: Requirements Extractor (P0)

**Purpose**: Parse SWE-bench issues into discrete, testable requirements.

| ID | Requirement | Rationale | Acceptance Criteria |
|----|-------------|-----------|---------------------|
| FR-201 | Extract 5-20 requirements per issue | Appropriate granularity | Mean ~10 requirements/issue on SWE-bench Lite |
| FR-202 | Classify as functional/non-functional/constraint | Standard categorization | Each requirement tagged |
| FR-203 | Tag with keywords for relevance matching | Enable oracle question matching | Keywords enable >0.7 relevance on direct questions |
| FR-204 | Support manual validation/override | Quality assurance | JSON schema for manual edits |

### F4: Test Generation Evaluator (P0)

**Purpose**: Evaluate generated tests using fail-to-pass methodology.

| ID | Requirement | Rationale | Acceptance Criteria |
|----|-------------|-----------|---------------------|
| FR-301 | Compute Applicability (W) metric | % tests execute without error | SWT-bench compatible |
| FR-302 | Compute Success Rate (S) metric | % tests fail→pass correctly | SWT-bench compatible |
| FR-303 | Compute F→P Rate | Core TDD metric | Fail before patch, pass after |
| FR-304 | Compute Coverage Delta | % gold patch lines covered | Δ measurable per test suite |
| FR-305 | Integrate with SWT-bench harness | Standard evaluation | Use Docker-based evaluation |

### F5: Implementation Evaluator (P0)

**Purpose**: Validate generated patches against gold tests.

| ID | Requirement | Rationale | Acceptance Criteria |
|----|-------------|-----------|---------------------|
| FR-401 | Output predictions in SWE-bench JSONL format | Standard compatibility | `{"instance_id": ..., "model_patch": ...}` |
| FR-402 | Support SWE-bench Lite, Verified, full datasets | Flexibility | Configurable dataset parameter |
| FR-403 | Track metrics per degradation level | Correlation analysis | Metrics grouped by level |
| FR-404 | 100% compatible with `swebench.harness.run_evaluation` | Direct integration | Passes SWE-bench validation |

### F6: Full Pipeline Orchestrator (P0)

**Purpose**: Run complete SDD evaluation from degraded spec to validated implementation.

| ID | Requirement | Rationale | Acceptance Criteria |
|----|-------------|-----------|---------------------|
| FR-501 | Execute 7-phase pipeline (degrade→elicit→parse→test→implement→refine→validate) | Full lifecycle coverage | All phases execute in order |
| FR-502 | Async/parallel evaluation support | Performance | Multiple instances evaluated concurrently |
| FR-503 | Checkpoint/resume for long runs | Reliability | Resume from last completed phase |
| FR-504 | Configurable phase skipping | Flexibility | Skip elicitation to test downstream phases |
| FR-505 | Structured logging for debugging | Observability | JSON-formatted logs per phase |

### F7: Framework Protocol Definition (P0)

**Purpose**: Define interface that SDD frameworks must implement for evaluation.

| ID | Requirement | Rationale | Acceptance Criteria |
|----|-------------|-----------|---------------------|
| FR-601 | Define SDDFrameworkProtocol with type hints | IDE support | Protocol fully typed |
| FR-602 | Implement passthrough baseline | Baseline comparison | Passes full spec directly |
| FR-603 | Implement at least one real framework | Demonstrate value | Claude Code or similar |
| FR-604 | Protocol is minimal but complete | Low barrier | <10 methods required |

### F8: Metrics Dashboard and Reporting (P0)

**Purpose**: Visualize and compare evaluation results.

| ID | Requirement | Rationale | Acceptance Criteria |
|----|-------------|-----------|---------------------|
| FR-701 | Compute all phase-specific metrics | Full observability | Elicitation, spec, tests, implementation metrics |
| FR-702 | Generate Markdown report | Simple output | Valid markdown, renders correctly |
| FR-703 | Generate HTML interactive report | Rich visualization | Static HTML with filtering |
| FR-704 | Export to JSON/CSV | Programmatic access | Valid JSON/CSV schemas |
| FR-705 | Comparison mode for A/B testing | Framework comparison | Side-by-side metrics |

---

## Non-Functional Requirements

### Performance

- Single instance evaluation: <5 minutes average (excluding Docker image pull)
- Batch evaluation (SWE-bench Lite, 300 instances): <4 hours with parallel=8
- Memory usage: <4GB for orchestrator process (Docker containers separate)

### Security

- No secrets in logs or reports
- Docker containers run with minimal privileges
- No network access from evaluation containers unless required

### Reliability

- Graceful handling of Docker failures
- Checkpoint/resume prevents lost work
- Timeout handling for hung processes

### Maintainability

- >80% test coverage
- Type hints throughout (mypy strict)
- Docstrings on all public APIs

---

## Technical Constraints

- Python 3.14+ (per existing pyproject.toml)
- Package name: `claude_spec_benchmark` (existing)
- CLI command: `claude-spec-benchmark`
- Subpackage structure: `degradation/`, `elicitation/`, `frameworks/`, `reporting/`
- Docker 24+ required for evaluation

---

## Dependencies

### Internal Dependencies

- Existing `runner.py`: Reuse for TDD implementation phase
- Existing `evaluator.py`: Reuse for patch evaluation
- Existing `models.py`: Extend with new data classes
- Existing `docker_manager.py`: Reuse for container management

### External Dependencies

(To be finalized after dependency analysis agent completes)

- `swebench` - Official harness integration
- `datasets` - HuggingFace dataset loading (existing)
- `pydantic` - Data validation (existing)
- `typer` - CLI framework (replacing click per brief)

---

## Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| SWE-bench harness API changes | Medium | High | Pin version, integration tests |
| Elicitation scoring subjectivity | High | Medium | Multiple scoring strategies, calibration dataset |
| Docker resource requirements (120GB+) | Medium | Medium | Lazy image pulling, cleanup utilities |
| Framework integration complexity | Medium | Medium | Minimal protocol, comprehensive docs |
| Rules-based oracle too rigid | Medium | Low | LLM mode as v1.x enhancement |

---

## Open Questions

- [ ] What's the appropriate elicitation question budget (max questions before timeout)?
- [ ] How granular should checkpoints be (per-phase vs per-instance)?
- [ ] Should degradation patterns be repository-specific or universal?
- [ ] What calibration data exists for elicitation scoring?

---

## Appendix

### Glossary

| Term | Definition |
|------|------------|
| Degradation | Process of removing information from a specification |
| Elicitation | Discovery of requirements through dialogue |
| Oracle | System that simulates a stakeholder with hidden knowledge |
| Fail-to-Pass (F→P) | Tests that fail before patch and pass after |
| TDD Red Phase | Writing failing tests before implementation |
| TDD Green Phase | Writing implementation to make tests pass |

### References

- [SWE-bench](https://github.com/princeton-nlp/SWE-bench) - Base benchmark
- [SWT-bench](https://arxiv.org/abs/2406.14259) - Test generation benchmark
- [LLMREI](https://arxiv.org/abs/2401.17394) - LLM-based requirements elicitation
- Project Brief: `docs/project-brief.md`
