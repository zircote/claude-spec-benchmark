# GitHub Copilot Instructions

## Project Overview

This is a SWE-bench test harness for evaluating AI code generation capabilities. The project benchmarks how well AI models can solve real-world software engineering tasks from GitHub issues and pull requests.

## Technology Stack

- **Python**: 3.14+ (use modern type hints, no `from __future__ import annotations`)
- **Package Manager**: Astral UV (`uv sync`, `uv run`, `uv build`)
- **Type Checking**: mypy with strict mode
- **Linting/Formatting**: ruff
- **Testing**: pytest with coverage
- **Security**: bandit

## Code Conventions

### Type Hints

Use Python 3.14+ type hint syntax:

```python
# Good - modern syntax
def evaluate(task_id: str, patch: str) -> EvaluationResult:
    results: list[TaskResult] = []
    config: dict[str, Any] = {}

# Avoid - legacy syntax
from typing import List, Dict
def evaluate(task_id: str, patch: str) -> EvaluationResult:
    results: List[TaskResult] = []
```

### Dataclasses and Protocols

Prefer dataclasses for data containers and Protocols for structural typing:

```python
from dataclasses import dataclass
from typing import Protocol

@dataclass(frozen=True, slots=True)
class EvaluationResult:
    task_id: str
    passed: bool
    score: float
    details: dict[str, Any]

class Evaluator(Protocol):
    def evaluate(self, task: Task) -> EvaluationResult: ...
```

### Error Handling

Use specific exception types and context managers:

```python
class EvaluationError(Exception):
    """Base exception for evaluation errors."""
    pass

class TaskNotFoundError(EvaluationError):
    """Raised when a SWE-bench task cannot be found."""
    pass
```

### Path Handling

Use `pathlib.Path` for all filesystem operations:

```python
from pathlib import Path

def load_config(config_path: Path) -> Config:
    return Config.from_toml(config_path.read_text())
```

## UV Package Manager

All commands use UV:

```bash
# Install dependencies
uv sync --all-extras --dev

# Run tools
uv run pytest
uv run ruff check .
uv run mypy .

# Build package
uv build
```

## Testing Patterns

### Test Organization

```
tests/
  unit/           # Fast, isolated unit tests
  integration/    # Tests with external dependencies
  fixtures/       # Shared test data and fixtures
  conftest.py     # Pytest configuration and fixtures
```

### Fixture Patterns

```python
import pytest
from pathlib import Path

@pytest.fixture
def sample_task() -> Task:
    return Task(
        task_id="django__django-12345",
        repo="django/django",
        base_commit="abc123",
        problem_statement="Fix issue with...",
    )

@pytest.fixture
def temp_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace
```

## Documentation

- Use Google-style docstrings
- Include type information in docstrings for public APIs
- Document exceptions that may be raised

```python
def run_evaluation(
    task_ids: list[str],
    model: str,
    *,
    timeout: int = 300,
) -> list[EvaluationResult]:
    """Run evaluation on a set of SWE-bench tasks.

    Args:
        task_ids: List of SWE-bench task identifiers.
        model: The AI model identifier to evaluate.
        timeout: Maximum seconds per task evaluation.

    Returns:
        List of evaluation results, one per task.

    Raises:
        TaskNotFoundError: If any task_id is not in the dataset.
        EvaluationTimeoutError: If a task exceeds the timeout.
    """
```

## Common Patterns

### Configuration Loading

```python
from pathlib import Path
import tomllib

def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)
```

### Async Operations

Use `asyncio` for concurrent evaluations:

```python
import asyncio

async def evaluate_batch(tasks: list[Task]) -> list[EvaluationResult]:
    async with asyncio.TaskGroup() as tg:
        futures = [tg.create_task(evaluate_one(task)) for task in tasks]
    return [f.result() for f in futures]
```

## Files to Understand

When working on this codebase, these files provide key context:

- `pyproject.toml` - Project configuration and dependencies
- `src/*/evaluator.py` - Core evaluation logic
- `src/*/models.py` - Data models and types
- `tests/conftest.py` - Test fixtures and configuration
