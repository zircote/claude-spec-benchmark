---
document_type: architecture
project_id: SPEC-2025-12-16-001
version: 0.1.0
last_updated: 2025-12-16T00:00:00Z
status: draft
---

# SDD-Bench: Technical Architecture

## System Overview

SDD-Bench extends the existing `claude_spec_benchmark` package with modules for specification degradation, elicitation simulation, and full pipeline orchestration. The architecture preserves existing patterns (async/await, plugin architecture, frozen models) while adding new capabilities.

### Architecture Diagram

```
                              ┌─────────────────────────────────────────────────────────────────────────┐
                              │                    SDD-Bench CLI                                        │
                              │    claude-spec-benchmark run --framework claude --degradation vague     │
                              └─────────────────────────────────────────────────────────────────────────┘
                                                          │
                                                          ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    SDDBenchRunner (Pipeline Orchestrator)                             │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐        │
│  │ Phase 0       │  │ Phase 1       │  │ Phase 2       │  │ Phase 3-4     │  │ Phase 5-6     │        │
│  │ Degradation   │─▶│ Elicitation   │─▶│ Spec Parse    │─▶│ TDD Red/Green │─▶│ Validation    │        │
│  │               │  │               │  │               │  │               │  │               │        │
│  │ DegradedSpec  │  │ ElicitedSpec  │  │ Requirements  │  │ Tests + Code  │  │ Results       │        │
│  └───────────────┘  └───────────────┘  └───────────────┘  └───────────────┘  └───────────────┘        │
└───────────────────────────────────────────────────────────────────────────────────────────────────────┘
        │                     │                     │                     │                     │
        ▼                     ▼                     ▼                     ▼                     ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│ degradation/  │     │ elicitation/  │     │ elicitation/  │     │ frameworks/   │     │ evaluation/   │
│               │     │               │     │               │     │               │     │               │
│ engine.py     │     │ oracle.py     │     │ extraction.py │     │ protocol.py   │     │ [existing]    │
│ patterns.py   │     │ scoring.py    │     │               │     │ passthrough   │     │ evaluator.py  │
│               │     │               │     │               │     │ claude_code   │     │ runner.py     │
└───────────────┘     └───────────────┘     └───────────────┘     └───────────────┘     └───────────────┘
                                                                          │                     │
                                                                          ▼                     ▼
                                                                  ┌───────────────┐     ┌───────────────┐
                                                                  │ DockerManager │     │ SWE-bench     │
                                                                  │ [existing]    │     │ Harness       │
                                                                  └───────────────┘     └───────────────┘
```

### Key Design Decisions

1. **Extend, don't replace**: New modules plug into existing infrastructure
2. **Subpackage organization**: Each feature area gets its own subpackage
3. **Protocol-based framework abstraction**: Frameworks implement `SDDFrameworkProtocol`
4. **Rules-first oracle**: Reproducibility over realism for benchmarking

---

## Component Design

### Existing Components (Reused)

#### DockerManager (`docker_manager.py`)
- **Purpose**: Container lifecycle management
- **Reuse**: Test execution, repo checkout, patch application
- **No changes required**

#### ClaudeCodeRunner (`runner.py`)
- **Purpose**: Execute Claude Code CLI on tasks
- **Reuse**: Implementation phase (TDD Green)
- **Note**: Will be wrapped by framework implementations

#### Evaluator (`evaluator.py`)
- **Purpose**: Compute diff metrics, run tests
- **Reuse**: Validation phase
- **Extension**: New metric plugins for SDD-specific metrics

#### TaskLoader (`task_loader.py`)
- **Purpose**: Load SWE-bench dataset
- **Reuse**: Task sourcing
- **No changes required**

#### MetricsCollector (`metrics.py`)
- **Purpose**: Aggregate evaluation results
- **Reuse**: Final reporting
- **Extension**: Add SDD phase metrics

---

### New Components

#### F1: degradation/ Subpackage

##### DegradationEngine (`degradation/engine.py`)

```python
class SpecDegradationLevel(Enum):
    FULL = 0      # Original SWE-bench issue
    PARTIAL = 1   # Remove code snippets, stack traces, file paths
    VAGUE = 2     # Keep only problem description
    MINIMAL = 3   # Single sentence summary
    AMBIGUOUS = 4 # Intentionally unclear/contradictory

@dataclass(frozen=True)
class DegradedSpec:
    degraded_text: str
    hidden_details: list[str]
    original_text: str
    level: SpecDegradationLevel
    seed: int

class DegradationEngine:
    """Deterministic specification degradation."""

    def __init__(self, patterns: DegradationPatterns | None = None):
        self._patterns = patterns or DegradationPatterns.default()

    def degrade(
        self,
        full_issue: str,
        level: SpecDegradationLevel,
        seed: int | None = None,
    ) -> DegradedSpec:
        """Apply degradation transforms to create vaguer specification."""
```

**Responsibilities:**
- Apply level-appropriate transforms
- Track hidden details for scoring
- Ensure deterministic output given seed

**Dependencies:**
- `re` - Pattern matching for transform rules
- `random` - Seeded randomization

##### DegradationPatterns (`degradation/patterns.py`)

```python
@dataclass
class DegradationPatterns:
    """Configurable patterns for degradation transforms."""

    code_block_pattern: str = r"```[\s\S]*?```"
    file_path_pattern: str = r"[\w/]+\.(py|js|ts|java|go)"
    stack_trace_pattern: str = r"Traceback.*?(?=\n\n|\Z)"

    @classmethod
    def default(cls) -> "DegradationPatterns":
        """Return default patterns for Python-focused repos."""

    @classmethod
    def for_repo(cls, repo: str) -> "DegradationPatterns":
        """Return repo-specific patterns if available."""
```

---

#### F2-F3: elicitation/ Subpackage

##### ElicitationOracle (`elicitation/oracle.py`)

```python
@dataclass(frozen=True)
class OracleResponse:
    answer: str
    relevance_score: float  # 0.0-1.0
    revealed_requirements: list[str]
    question_category: QuestionCategory

class QuestionCategory(Enum):
    CLARIFICATION = "clarification"
    ASSUMPTION = "assumption"
    EVIDENCE = "evidence"
    VIEWPOINT = "viewpoint"
    IMPLICATION = "implication"
    META = "meta"
    UNKNOWN = "unknown"

@dataclass(frozen=True)
class ElicitationMetrics:
    discovery_rate: float           # % requirements discovered
    question_efficiency: float      # discoveries per question
    total_questions: int
    question_distribution: dict[QuestionCategory, int]
    revealed_requirements: list[str]
    hidden_requirements: list[str]

class ElicitationOracle:
    """Simulates a stakeholder with hidden knowledge."""

    def __init__(
        self,
        full_spec: str,
        requirements: list[Requirement],
        personality: Literal["helpful", "terse", "confused"] = "helpful",
    ):
        self._full_spec = full_spec
        self._requirements = requirements
        self._revealed: set[str] = set()
        self._questions: list[tuple[str, OracleResponse]] = []
        self._personality = personality

    def ask(self, question: str) -> OracleResponse:
        """Answer question based on relevance to hidden requirements."""

    def get_metrics(self) -> ElicitationMetrics:
        """Return elicitation performance metrics."""
```

**Behavior Logic:**
- High relevance (>0.7): Reveal specific requirement
- Medium relevance (0.4-0.7): Provide hints
- Low relevance (<0.4): Request clarification

##### QuestionScorer (`elicitation/scoring.py`)

```python
class QuestionScorer:
    """Score question relevance to hidden requirements."""

    def __init__(self, requirements: list[Requirement]):
        self._requirements = requirements
        self._keyword_index = self._build_index()

    def score(self, question: str) -> tuple[float, list[str]]:
        """
        Returns (relevance_score, matching_requirement_ids).

        Scoring algorithm:
        1. Extract keywords from question
        2. Match against requirement keywords
        3. Weight by category effectiveness
        4. Return highest-scoring matches
        """

    def classify_question(self, question: str) -> QuestionCategory:
        """Classify question into Socratic category."""
```

**Scoring Strategy:**
- Keyword overlap (TF-IDF or simpler)
- Question category bonus (clarification > meta)
- Personality modifier (terse oracle needs more precise questions)

##### RequirementsExtractor (`elicitation/extraction.py`)

```python
@dataclass(frozen=True)
class Requirement:
    id: str
    text: str
    category: Literal["functional", "non-functional", "constraint"]
    keywords: list[str]
    discoverable: bool
    source_span: tuple[int, int]

class RequirementsExtractor:
    """Extract atomic requirements from issue text."""

    def extract(self, issue_text: str) -> list[Requirement]:
        """
        Extract requirements using rule-based patterns.

        Strategy:
        1. Split into sentences/paragraphs
        2. Identify requirement indicators ("should", "must", "when")
        3. Classify each as functional/non-functional/constraint
        4. Extract keywords for relevance matching
        5. Mark as discoverable/implicit
        """
```

---

#### F6: Pipeline Orchestrator

##### SDDBenchRunner (`runner.py` - new or extension)

```python
@dataclass(frozen=True)
class SDDPhaseResult:
    phase: str
    success: bool
    duration_seconds: float
    artifacts: dict[str, Any]
    error: str | None = None

@dataclass(frozen=True)
class SDDBenchResult:
    instance_id: str
    degradation_level: SpecDegradationLevel
    phase_results: list[SDDPhaseResult]
    final_status: EvaluationResult
    elicitation_metrics: ElicitationMetrics | None
    test_metrics: EvaluationMetrics | None
    total_duration_seconds: float

class SDDBenchRunner:
    """Full pipeline orchestrator for SDD evaluation."""

    def __init__(
        self,
        framework: SDDFrameworkProtocol,
        dataset: str = "princeton-nlp/SWE-bench_Lite",
        degradation_level: SpecDegradationLevel = SpecDegradationLevel.VAGUE,
        docker_manager: DockerManager | None = None,
    ):
        self._framework = framework
        self._task_loader = TaskLoader(dataset)
        self._degradation = DegradationEngine()
        self._level = degradation_level
        self._docker = docker_manager or DockerManager()

    async def run(
        self,
        limit: int | None = None,
        parallel: int = 1,
        skip_phases: list[str] | None = None,
    ) -> list[SDDBenchResult]:
        """Run full pipeline on tasks."""

    async def run_single(
        self,
        task: SWEBenchTask,
        checkpoint: Path | None = None,
    ) -> SDDBenchResult:
        """Run pipeline on single task with optional checkpointing."""
```

**Pipeline Phases:**
1. **Degrade** (Phase 0): DegradationEngine
2. **Elicit** (Phase 1): Framework + Oracle dialogue
3. **Parse** (Phase 2): Framework spec parsing
4. **Test** (Phase 3): Framework test generation + SWT-bench evaluation
5. **Implement** (Phase 4): Framework implementation + patch generation
6. **Refine** (Phase 5): Framework refinement on failures
7. **Validate** (Phase 6): Evaluator + SWE-bench harness

---

#### F7: frameworks/ Subpackage

##### SDDFrameworkProtocol (`frameworks/protocol.py`)

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

##### PassthroughFramework (`frameworks/passthrough.py`)

```python
class PassthroughFramework:
    """Baseline that passes full spec directly (no elicitation)."""

    def start_elicitation(self, initial_context: str) -> str:
        return ""  # No questions

    def process_response(self, stakeholder_response: str) -> str | None:
        return None  # Done immediately

    def get_elicited_spec(self) -> str:
        return self._initial_context  # Return unchanged

    # ... other methods pass through or return defaults
```

##### ClaudeCodeFramework (`frameworks/claude_code.py`)

```python
class ClaudeCodeFramework:
    """Claude Code SDK integration for SDD evaluation."""

    def __init__(
        self,
        runner: ClaudeCodeRunner,
        model: str = "sonnet",
        max_elicitation_rounds: int = 10,
    ):
        self._runner = runner
        self._model = model
        self._max_rounds = max_elicitation_rounds
        self._elicitation_history: list[tuple[str, str]] = []

    # Implementation uses ClaudeCodeRunner for each phase
```

---

#### F8: reporting/ Subpackage

##### SDDReportGenerator (`reporting/dashboard.py`)

```python
class SDDReportGenerator:
    """Generate SDD-specific reports and dashboards."""

    def generate_html_report(
        self,
        results: list[SDDBenchResult],
        output_path: Path,
    ) -> None:
        """Generate interactive HTML dashboard."""

    def generate_markdown_report(
        self,
        results: list[SDDBenchResult],
    ) -> str:
        """Generate Markdown summary."""

    def generate_comparison_report(
        self,
        baseline_results: list[SDDBenchResult],
        experiment_results: list[SDDBenchResult],
    ) -> str:
        """Compare two framework configurations."""
```

---

## Data Design

### New Data Models

Add to `models.py`:

```python
# Degradation models
class SpecDegradationLevel(str, Enum):
    FULL = "full"
    PARTIAL = "partial"
    VAGUE = "vague"
    MINIMAL = "minimal"
    AMBIGUOUS = "ambiguous"

class DegradedSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    degraded_text: str
    hidden_details: list[str]
    original_text: str
    level: SpecDegradationLevel
    seed: int | None

# Elicitation models
class QuestionCategory(str, Enum):
    CLARIFICATION = "clarification"
    ASSUMPTION = "assumption"
    EVIDENCE = "evidence"
    VIEWPOINT = "viewpoint"
    IMPLICATION = "implication"
    META = "meta"
    UNKNOWN = "unknown"

class Requirement(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    text: str
    category: Literal["functional", "non-functional", "constraint"]
    keywords: list[str]
    discoverable: bool = True
    source_span: tuple[int, int] | None = None

class OracleResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    answer: str
    relevance_score: float
    revealed_requirements: list[str]
    question_category: QuestionCategory

class ElicitationMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    discovery_rate: float
    question_efficiency: float
    total_questions: int
    question_distribution: dict[str, int]
    revealed_requirements: list[str]
    hidden_requirements: list[str]

# Pipeline models
class SDDPhaseResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    phase: str
    success: bool
    duration_seconds: float
    artifacts: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None

class SDDBenchResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    instance_id: str
    degradation_level: SpecDegradationLevel
    phase_results: list[SDDPhaseResult]
    final_status: EvaluationResult
    elicitation_metrics: ElicitationMetrics | None = None
    test_metrics: EvaluationMetrics | None = None
    total_duration_seconds: float
```

### Data Flow

```
SWEBenchTask.problem_statement
        │
        ▼
DegradationEngine.degrade()
        │
        ▼
DegradedSpec { degraded_text, hidden_details }
        │
        ├──────────────────────────────────────────┐
        ▼                                          ▼
RequirementsExtractor.extract()          ElicitationOracle(hidden_details)
        │                                          │
        ▼                                          ▼
list[Requirement]                        Framework.start_elicitation()
        │                                          │
        │                                    ┌─────┴─────┐
        │                                    ▼           ▼
        │                             Framework.process_response()
        │                                    │
        │                              (loop until None)
        │                                    │
        │                                    ▼
        │                           Framework.get_elicited_spec()
        │                                    │
        └────────────────────────────────────┴──────────────┐
                                                            ▼
                                              Framework.generate_tests()
                                                            │
                                                            ▼
                                              Framework.implement()
                                                            │
                                                            ▼
                                              Evaluator.evaluate()
                                                            │
                                                            ▼
                                              SDDBenchResult
```

---

## API Design

### CLI Extensions

```bash
# Existing commands (unchanged)
claude-spec-benchmark info
claude-spec-benchmark list [--repo REPO]
claude-spec-benchmark run [--tasks IDS] [--limit N]
claude-spec-benchmark report RESULTS_DIR

# New SDD commands
claude-spec-benchmark sdd run \
    --framework {passthrough|claude-code} \
    --degradation {full|partial|vague|minimal|ambiguous} \
    --dataset {lite|verified|full} \
    --limit N \
    --parallel N \
    --skip-phases {elicitation,tests,implementation} \
    --output DIR \
    --checkpoint DIR

claude-spec-benchmark sdd report RESULTS_DIR \
    --format {html|markdown|json|csv} \
    --compare BASELINE_DIR

claude-spec-benchmark sdd degrade ISSUE_FILE \
    --level {full|partial|vague|minimal|ambiguous} \
    --seed N \
    --output FILE

claude-spec-benchmark sdd extract ISSUE_FILE \
    --output FILE
```

### Programmatic API

```python
from claude_spec_benchmark import SDDBenchRunner, SpecDegradationLevel
from claude_spec_benchmark.frameworks import PassthroughFramework, ClaudeCodeFramework

# Run evaluation
runner = SDDBenchRunner(
    framework=ClaudeCodeFramework(),
    degradation_level=SpecDegradationLevel.VAGUE,
)

results = await runner.run(limit=50, parallel=4)

# Generate report
from claude_spec_benchmark.reporting import SDDReportGenerator

reporter = SDDReportGenerator()
reporter.generate_html_report(results, Path("report.html"))
```

---

## Integration Points

### Internal Integrations

| Component | Integration Type | Purpose |
|-----------|-----------------|---------|
| TaskLoader | Direct import | Load SWE-bench tasks |
| DockerManager | Dependency injection | Container management |
| Evaluator | Composition | Patch evaluation |
| MetricsCollector | Extension | Add SDD metrics |
| ClaudeCodeRunner | Wrapping | Framework implementation |

### External Integrations

| Service | Integration Type | Purpose |
|---------|-----------------|---------|
| SWE-bench harness | Process invocation | Gold test evaluation |
| HuggingFace datasets | Library | Dataset loading |
| Docker daemon | SDK | Container orchestration |

---

## Security Design

### Authentication
- No authentication required (CLI tool)
- Optional API key for LLM oracle mode (v1.x)

### Authorization
- Docker runs with minimal privileges
- Network disabled by default in containers

### Data Protection
- No PII in evaluation data
- Results stored locally only

### Security Considerations

| Threat | Mitigation |
|--------|------------|
| Command injection | Array args (no shell interpolation) |
| Container escape | Resource limits, network isolation |
| Malicious patches | Sandboxed container execution |

---

## Performance Considerations

### Expected Load
- Typical evaluation: 300 instances (SWE-bench Lite)
- Per-instance: 5-30 minutes depending on phases
- Full run: 2-8 hours with parallel=4

### Performance Targets

| Metric | Target | Rationale |
|--------|--------|-----------|
| Instance latency | <5 min avg | Responsive iteration |
| Memory (orchestrator) | <4GB | Desktop-friendly |
| Parallelism | 4-8 instances | Balance throughput vs resources |
| Docker image cache | 100% hit rate | Avoid repeated pulls |

### Optimization Strategies

1. **Checkpoint/resume**: Resume from last completed phase
2. **Parallel execution**: Semaphore-controlled concurrency
3. **Lazy loading**: Task dataset loaded on demand
4. **Docker layer caching**: Reuse base images

---

## Testing Strategy

### Unit Testing

| Component | Coverage Target | Focus |
|-----------|-----------------|-------|
| DegradationEngine | 95% | Determinism, all levels |
| ElicitationOracle | 90% | Scoring, reveal logic |
| RequirementsExtractor | 85% | Edge cases |
| SDDBenchRunner | 80% | Phase orchestration |

### Integration Testing

- End-to-end pipeline on 3-5 sample tasks
- Framework protocol compliance tests
- Docker container lifecycle tests

### Performance Testing

- Benchmark degradation engine on 1000 issues
- Measure oracle response time per question
- Profile memory usage during full runs

---

## Deployment Considerations

### Environment Requirements
- Python 3.14+
- Docker 24+
- 120GB+ disk (SWE-bench images)
- 16GB+ RAM
- 8+ CPU cores recommended

### Configuration Management
- Environment variables for paths
- TOML/YAML config file support
- CLI args override config

### Rollout Strategy
- Alpha: Internal testing with passthrough framework
- Beta: Claude Code framework integration
- GA: Published to PyPI

---

## Directory Structure (Updated)

```
src/claude_spec_benchmark/
├── __init__.py               # Package exports
├── cli.py                    # Typer CLI (replacing Click)
├── models.py                 # Extended with SDD models
├── runner.py                 # Existing (unchanged)
├── evaluator.py              # Existing (unchanged)
├── docker_manager.py         # Existing (unchanged)
├── task_loader.py            # Existing (unchanged)
├── metrics.py                # Extended with SDD metrics
├── sdd_runner.py             # NEW: SDDBenchRunner
│
├── degradation/              # NEW: F1
│   ├── __init__.py
│   ├── engine.py             # DegradationEngine
│   └── patterns.py           # DegradationPatterns
│
├── elicitation/              # NEW: F2, F3
│   ├── __init__.py
│   ├── oracle.py             # ElicitationOracle
│   ├── scoring.py            # QuestionScorer
│   └── extraction.py         # RequirementsExtractor
│
├── frameworks/               # NEW: F7
│   ├── __init__.py
│   ├── protocol.py           # SDDFrameworkProtocol
│   ├── passthrough.py        # PassthroughFramework
│   └── claude_code.py        # ClaudeCodeFramework
│
└── reporting/                # NEW: F8
    ├── __init__.py
    ├── dashboard.py          # SDDReportGenerator
    └── templates/            # HTML report templates
        └── report.html.j2
```

---

## Future Considerations

### v1.x Enhancements
- LLM-based oracle responses
- Modal cloud evaluation
- Custom dataset support
- VS Code extension
- Leaderboard integration

### Scaling
- Distributed evaluation across machines
- Results aggregation service
- CI/CD integration examples
