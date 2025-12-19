---
document_type: implementation_plan
project_id: SPEC-2025-12-16-001
version: 0.1.0
last_updated: 2025-12-16T00:00:00Z
status: draft
---

# SDD-Bench - Implementation Plan

## Overview

This plan implements the 8 features (F1-F8) from the project brief by extending the existing `claude_spec_benchmark` package with new subpackages for degradation, elicitation, frameworks, and reporting.

## Phase Summary

| Phase | Focus | Key Deliverables |
|-------|-------|------------------|
| Phase 1 | Foundation | Degradation engine, data models, CLI skeleton |
| Phase 2 | Elicitation | Oracle system, question scoring, requirements extraction |
| Phase 3 | Integration | SWE/SWT-bench harness integration, test evaluation |
| Phase 4 | Pipeline | Full orchestrator, framework protocol, reference implementations |
| Phase 5 | Polish | Reporting dashboard, documentation, performance optimization |

---

## Phase 1: Foundation

**Goal**: Core infrastructure for specification degradation and extended data models.

**Prerequisites**: Existing codebase understanding, development environment setup

### Tasks

#### Task 1.1: Create Subpackage Structure

- **Description**: Create directory structure for new subpackages
- **Dependencies**: None
- **Acceptance Criteria**:
  - [ ] `src/claude_spec_benchmark/degradation/` exists with `__init__.py`
  - [ ] `src/claude_spec_benchmark/elicitation/` exists with `__init__.py`
  - [ ] `src/claude_spec_benchmark/frameworks/` exists with `__init__.py`
  - [ ] `src/claude_spec_benchmark/reporting/` exists with `__init__.py`

#### Task 1.2: Extend Data Models

- **Description**: Add SDD-specific Pydantic models to `models.py`
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - [ ] `SpecDegradationLevel` enum added
  - [ ] `DegradedSpec` model added (frozen)
  - [ ] `QuestionCategory` enum added
  - [ ] `Requirement` model added (frozen)
  - [ ] `OracleResponse` model added (frozen)
  - [ ] `ElicitationMetrics` model added (frozen)
  - [ ] `SDDPhaseResult` model added (frozen)
  - [ ] `SDDBenchResult` model added (frozen)
  - [ ] All models have docstrings
  - [ ] Unit tests pass

#### Task 1.3: Implement DegradationPatterns

- **Description**: Create configurable patterns for degradation transforms
- **Dependencies**: Task 1.2
- **Acceptance Criteria**:
  - [ ] `degradation/patterns.py` created
  - [ ] `DegradationPatterns` dataclass with regex patterns
  - [ ] Default patterns for Python repos
  - [ ] `for_repo()` class method for repo-specific patterns
  - [ ] Unit tests for pattern matching

#### Task 1.4: Implement DegradationEngine

- **Description**: Core degradation logic for all 5 levels
- **Dependencies**: Task 1.3
- **Acceptance Criteria**:
  - [ ] `degradation/engine.py` created
  - [ ] `degrade()` method implements all 5 levels
  - [ ] Deterministic output with seeded random
  - [ ] Hidden details tracked correctly
  - [ ] Unit tests for each degradation level
  - [ ] Integration test with sample SWE-bench issues

#### Task 1.5: Add Degradation CLI Command

- **Description**: CLI command to test degradation standalone
- **Dependencies**: Task 1.4
- **Acceptance Criteria**:
  - [ ] `claude-spec-benchmark sdd degrade` command works
  - [ ] Accepts issue file, level, seed parameters
  - [ ] Outputs degraded spec to stdout or file
  - [ ] Help text complete

#### Task 1.6: Update Package Exports

- **Description**: Export new classes from `__init__.py`
- **Dependencies**: Task 1.4
- **Acceptance Criteria**:
  - [ ] All new models exported
  - [ ] `DegradationEngine` exported
  - [ ] No import errors

### Phase 1 Deliverables

- [ ] Degradation engine working for all 5 levels
- [ ] Extended data models with tests
- [ ] CLI command for standalone degradation
- [ ] Documentation strings on all new code

### Phase 1 Exit Criteria

- [ ] `pytest tests/test_degradation.py` passes
- [ ] `ruff check src/` passes
- [ ] `mypy src/` passes

---

## Phase 2: Elicitation

**Goal**: Complete elicitation simulation system with oracle and scoring.

**Prerequisites**: Phase 1 complete

### Tasks

#### Task 2.1: Implement QuestionScorer

- **Description**: Score question relevance to hidden requirements
- **Dependencies**: Phase 1 complete
- **Acceptance Criteria**:
  - [ ] `elicitation/scoring.py` created
  - [ ] `QuestionScorer` class with `score()` method
  - [ ] Keyword extraction from questions
  - [ ] TF-IDF or simpler keyword matching
  - [ ] `classify_question()` for Socratic categories
  - [ ] Unit tests with known question/requirement pairs

#### Task 2.2: Implement RequirementsExtractor

- **Description**: Extract atomic requirements from issue text
- **Dependencies**: Task 2.1 (shares keyword logic)
- **Acceptance Criteria**:
  - [ ] `elicitation/extraction.py` created
  - [ ] `RequirementsExtractor` class with `extract()` method
  - [ ] Produces 5-20 requirements per typical issue
  - [ ] Categories assigned (functional/non-functional/constraint)
  - [ ] Keywords extracted per requirement
  - [ ] Unit tests with sample issues

#### Task 2.3: Implement ElicitationOracle

- **Description**: Simulated stakeholder with hidden knowledge
- **Dependencies**: Tasks 2.1, 2.2
- **Acceptance Criteria**:
  - [ ] `elicitation/oracle.py` created
  - [ ] `ElicitationOracle` class with `ask()` method
  - [ ] Response behavior based on relevance thresholds
  - [ ] Requirement revelation tracking
  - [ ] `get_metrics()` returns `ElicitationMetrics`
  - [ ] Personality modes (helpful, terse, confused)
  - [ ] Unit tests for revelation logic
  - [ ] Integration test simulating dialogue

#### Task 2.4: Add Extraction CLI Command

- **Description**: CLI command to test extraction standalone
- **Dependencies**: Task 2.2
- **Acceptance Criteria**:
  - [ ] `claude-spec-benchmark sdd extract` command works
  - [ ] Accepts issue file
  - [ ] Outputs requirements as JSON

#### Task 2.5: Update Package Exports

- **Description**: Export elicitation classes
- **Dependencies**: Task 2.3
- **Acceptance Criteria**:
  - [ ] `ElicitationOracle` exported
  - [ ] `QuestionScorer` exported
  - [ ] `RequirementsExtractor` exported

### Phase 2 Deliverables

- [ ] Elicitation oracle with rule-based scoring
- [ ] Requirements extraction from issues
- [ ] Question classification system
- [ ] CLI commands for testing

### Phase 2 Exit Criteria

- [ ] `pytest tests/test_elicitation.py` passes
- [ ] Oracle produces consistent scores for same inputs
- [ ] Extraction produces reasonable requirements on sample issues

---

## Phase 3: Integration

**Goal**: Connect to SWE-bench and SWT-bench harnesses for evaluation.

**Prerequisites**: Phase 2 complete

### Tasks

#### Task 3.1: SWE-bench Harness Integration

- **Description**: Ensure output compatibility with SWE-bench evaluation
- **Dependencies**: Phase 2 complete
- **Acceptance Criteria**:
  - [ ] Output predictions in JSONL format
  - [ ] Format: `{"instance_id": ..., "model_name_or_path": ..., "model_patch": ...}`
  - [ ] Integration test with `swebench.harness.run_evaluation`
  - [ ] Results match expected schema

#### Task 3.2: SWT-bench Test Evaluation

- **Description**: Integrate test generation evaluation metrics
- **Dependencies**: Task 3.1
- **Acceptance Criteria**:
  - [ ] Compute Applicability (W) metric
  - [ ] Compute Success Rate (S) metric
  - [ ] Compute F→P Rate
  - [ ] Coverage Delta computation
  - [ ] Results stored in `EvaluationMetrics`

#### Task 3.3: Extend MetricsCollector

- **Description**: Add SDD-specific metrics to collector
- **Dependencies**: Task 3.2
- **Acceptance Criteria**:
  - [ ] Elicitation metrics aggregation
  - [ ] Per-degradation-level breakdowns
  - [ ] Phase timing metrics
  - [ ] JSON export includes new metrics

#### Task 3.4: Docker Image Management

- **Description**: Ensure Docker image caching and cleanup utilities
- **Dependencies**: Task 3.1
- **Acceptance Criteria**:
  - [ ] Image pull status reporting
  - [ ] Cleanup command for orphan containers
  - [ ] Disk usage reporting

### Phase 3 Deliverables

- [ ] SWE-bench compatible output
- [ ] SWT-bench test evaluation metrics
- [ ] Extended metrics collection

### Phase 3 Exit Criteria

- [ ] Can run evaluation on sample task and get valid metrics
- [ ] Output passes SWE-bench validation
- [ ] Docker lifecycle managed correctly

---

## Phase 4: Pipeline

**Goal**: Complete end-to-end orchestrator and framework protocol.

**Prerequisites**: Phase 3 complete

### Tasks

#### Task 4.1: Define SDDFrameworkProtocol

- **Description**: Create Protocol class defining framework interface
- **Dependencies**: Phase 3 complete
- **Acceptance Criteria**:
  - [ ] `frameworks/protocol.py` created
  - [ ] 7 method signatures defined
  - [ ] Full type hints
  - [ ] Docstrings with examples

#### Task 4.2: Implement PassthroughFramework

- **Description**: Baseline framework that passes spec unchanged
- **Dependencies**: Task 4.1
- **Acceptance Criteria**:
  - [ ] `frameworks/passthrough.py` created
  - [ ] All protocol methods implemented
  - [ ] Returns defaults/empty for elicitation
  - [ ] Uses existing runner for implementation

#### Task 4.3: Implement ClaudeCodeFramework

- **Description**: Framework using Claude Code for SDD phases
- **Dependencies**: Task 4.2
- **Acceptance Criteria**:
  - [ ] `frameworks/claude_code.py` created
  - [ ] Elicitation via Claude prompts
  - [ ] Test generation via Claude
  - [ ] Implementation via existing `ClaudeCodeRunner`
  - [ ] Configurable model selection

#### Task 4.4: Implement SDDBenchRunner

- **Description**: Full pipeline orchestrator
- **Dependencies**: Tasks 4.1-4.3
- **Acceptance Criteria**:
  - [ ] `sdd_runner.py` created
  - [ ] 7-phase pipeline execution
  - [ ] Async/parallel support
  - [ ] Checkpoint/resume functionality
  - [ ] Phase skipping option
  - [ ] Structured logging

#### Task 4.5: Add SDD Run CLI Command

- **Description**: Main CLI command for running SDD evaluation
- **Dependencies**: Task 4.4
- **Acceptance Criteria**:
  - [ ] `claude-spec-benchmark sdd run` command works
  - [ ] All options from spec implemented
  - [ ] Progress output to console
  - [ ] Results saved to output directory

#### Task 4.6: Protocol Compliance Tests

- **Description**: Tests to verify framework implementations
- **Dependencies**: Tasks 4.2, 4.3
- **Acceptance Criteria**:
  - [ ] Tests for PassthroughFramework
  - [ ] Tests for ClaudeCodeFramework
  - [ ] Protocol type checking tests

### Phase 4 Deliverables

- [ ] Framework protocol definition
- [ ] Two reference implementations
- [ ] Full pipeline orchestrator
- [ ] CLI for running evaluations

### Phase 4 Exit Criteria

- [ ] Can run full pipeline on single task
- [ ] Passthrough framework produces valid results
- [ ] Checkpoint/resume works correctly

---

## Phase 5: Polish

**Goal**: Production-ready release with reporting and documentation.

**Prerequisites**: Phase 4 complete

### Tasks

#### Task 5.1: Implement SDDReportGenerator

- **Description**: Generate reports in multiple formats
- **Dependencies**: Phase 4 complete
- **Acceptance Criteria**:
  - [ ] `reporting/dashboard.py` created
  - [ ] Markdown report generation
  - [ ] HTML report generation
  - [ ] JSON/CSV export
  - [ ] Comparison mode for A/B testing

#### Task 5.2: HTML Report Templates

- **Description**: Create interactive HTML dashboard
- **Dependencies**: Task 5.1
- **Acceptance Criteria**:
  - [ ] `reporting/templates/report.html.j2` created
  - [ ] Filtering by degradation level
  - [ ] Filtering by phase
  - [ ] Charts for metrics visualization
  - [ ] Responsive design

#### Task 5.3: Add Report CLI Command

- **Description**: CLI command for generating reports
- **Dependencies**: Task 5.1
- **Acceptance Criteria**:
  - [ ] `claude-spec-benchmark sdd report` command works
  - [ ] Format selection (html/markdown/json/csv)
  - [ ] Comparison mode with baseline

#### Task 5.4: Performance Optimization

- **Description**: Profile and optimize hot paths
- **Dependencies**: Tasks 5.1-5.3
- **Acceptance Criteria**:
  - [ ] Profile full pipeline run
  - [ ] Identify bottlenecks
  - [ ] Optimize keyword scoring if needed
  - [ ] Docker image caching effective

#### Task 5.5: Documentation

- **Description**: User and developer documentation
- **Dependencies**: All previous tasks
- **Acceptance Criteria**:
  - [ ] README updated with SDD features
  - [ ] Framework integration guide
  - [ ] API reference generated
  - [ ] Example notebooks

#### Task 5.6: Final Testing

- **Description**: End-to-end validation on full dataset
- **Dependencies**: All previous tasks
- **Acceptance Criteria**:
  - [ ] Run on 10 SWE-bench Lite instances
  - [ ] All degradation levels tested
  - [ ] Both frameworks tested
  - [ ] Report generation verified
  - [ ] >80% test coverage

### Phase 5 Deliverables

- [ ] Interactive HTML dashboard
- [ ] Complete documentation
- [ ] Performance validated
- [ ] v1.0.0 ready

### Phase 5 Exit Criteria

- [ ] Full pipeline runs without errors
- [ ] Documentation complete
- [ ] Test coverage >80%
- [ ] Performance targets met

---

## Dependency Graph

```
Phase 1: Foundation
  Task 1.1 (Structure)
       │
       ├──► Task 1.2 (Models)
       │         │
       │         ├──► Task 1.3 (Patterns)
       │         │         │
       │         │         └──► Task 1.4 (Engine)
       │         │                   │
       │         │                   └──► Task 1.5 (CLI)
       │         │
       │         └──► Task 1.6 (Exports)
       │
Phase 2: Elicitation
       │
       ├──► Task 2.1 (Scorer) ───────┐
       │         │                   │
       │         └──► Task 2.2 (Extractor)
       │                   │         │
       │                   │    Task 2.4 (CLI)
       │                   │
       │                   └──► Task 2.3 (Oracle)
       │                             │
       │                             └──► Task 2.5 (Exports)
       │
Phase 3: Integration
       │
       ├──► Task 3.1 (SWE-bench)
       │         │
       │         ├──► Task 3.2 (SWT-bench)
       │         │         │
       │         │         └──► Task 3.3 (Metrics)
       │         │
       │         └──► Task 3.4 (Docker)
       │
Phase 4: Pipeline
       │
       ├──► Task 4.1 (Protocol)
       │         │
       │         ├──► Task 4.2 (Passthrough)
       │         │
       │         └──► Task 4.3 (Claude Code)
       │                   │
       │                   └──► Task 4.4 (Runner)
       │                             │
       │                             ├──► Task 4.5 (CLI)
       │                             │
       │                             └──► Task 4.6 (Tests)
       │
Phase 5: Polish
       │
       ├──► Task 5.1 (Report Gen)
       │         │
       │         ├──► Task 5.2 (Templates)
       │         │
       │         └──► Task 5.3 (CLI)
       │                   │
       │                   └──► Task 5.4 (Perf)
       │                             │
       │                             └──► Task 5.5 (Docs)
       │                                       │
       │                                       └──► Task 5.6 (Final Test)
```

---

## Risk Mitigation Tasks

| Risk | Mitigation Task | Phase |
|------|-----------------|-------|
| SWE-bench harness changes | Pin version, add integration tests | Phase 3 |
| Elicitation scoring subjectivity | Create calibration test suite | Phase 2 |
| Docker resource requirements | Add cleanup utilities, progress reporting | Phase 3 |
| Framework integration complexity | Start with passthrough, minimal protocol | Phase 4 |

---

## Testing Checklist

- [ ] Unit tests for DegradationEngine
- [ ] Unit tests for QuestionScorer
- [ ] Unit tests for RequirementsExtractor
- [ ] Unit tests for ElicitationOracle
- [ ] Unit tests for SDDBenchRunner
- [ ] Integration tests for full pipeline
- [ ] Protocol compliance tests for frameworks
- [ ] Performance tests for batch evaluation

---

## Documentation Tasks

- [ ] Update README with SDD features
- [ ] Create framework integration guide
- [ ] Document CLI commands
- [ ] Add API reference (autodoc)
- [ ] Create example notebooks

---

## Launch Checklist

- [ ] All tests passing
- [ ] Documentation complete
- [ ] CLI help text reviewed
- [ ] Performance targets met
- [ ] Example evaluation run completed
- [ ] CHANGELOG updated
- [ ] Version bumped to 1.0.0
