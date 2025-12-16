# claude-spec-benchmark

SWE-bench test harness for evaluating Claude's software engineering capabilities.

## Features

- **SWE-bench Lite Support**: Load and evaluate against 300 curated tasks
- **Claude Code Integration**: Spawn Claude Code CLI subprocess for patch generation
- **Docker Isolation**: Each task runs in an isolated container for safety
- **Multi-Metric Evaluation**: Test-based pass/fail, diff similarity, and custom metrics
- **Rich Reporting**: Console tables and Markdown reports

## Prerequisites

- Python 3.14+
- Docker (for task isolation)
- Claude Code CLI (`claude`) installed and authenticated

## Installation

```bash
# Clone the repository
git clone https://github.com/zircote/claude-spec-benchmark.git
cd claude-spec-benchmark

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

## Quick Start

```bash
# List available tasks
claude-spec-benchmark list

# Run benchmark on specific tasks (dry-run first)
claude-spec-benchmark run --tasks "django__django-11099" --dry-run

# Run full benchmark
claude-spec-benchmark run --limit 10 --output ./results

# Generate report from results
claude-spec-benchmark report ./results
```

## CLI Commands

### `list` - List available tasks

```bash
claude-spec-benchmark list                    # List all tasks
claude-spec-benchmark list --repo django/django  # Filter by repo
claude-spec-benchmark list --limit 50         # Limit results
```

### `run` - Execute benchmark

```bash
claude-spec-benchmark run                     # Run all tasks
claude-spec-benchmark run --tasks "task1,task2"  # Specific tasks
claude-spec-benchmark run --repo django/django   # Tasks from repo
claude-spec-benchmark run --limit 10          # Limit task count
claude-spec-benchmark run --timeout 3600      # Custom timeout (seconds)
claude-spec-benchmark run --workers 8         # Parallel workers
claude-spec-benchmark run --model opus-4      # Model override
claude-spec-benchmark run --dry-run           # Preview without executing
```

### `report` - Generate reports

```bash
claude-spec-benchmark report ./results                  # Console table
claude-spec-benchmark report ./results --format markdown  # Markdown
claude-spec-benchmark report ./results --format json     # JSON
```

### `info` - Display harness information

```bash
claude-spec-benchmark info
```

## Python API

```python
from claude_spec_benchmark import (
    TaskLoader,
    ClaudeCodeRunner,
    DockerManager,
    Evaluator,
    MetricsCollector,
)

# Load tasks
loader = TaskLoader()
for task in loader.iter_tasks(repos=["django/django"]):
    print(f"Task: {task.instance_id}")

# Custom evaluation
docker = DockerManager()
evaluator = Evaluator(docker)
```

## Project Structure

```
claude-spec-benchmark/
├── src/claude_spec_benchmark/
│   ├── __init__.py          # Public API exports
│   ├── main.py              # CLI entry point
│   ├── models.py            # Pydantic data models
│   ├── task_loader.py       # SWE-bench dataset loading
│   ├── runner.py            # Claude Code subprocess execution
│   ├── docker_manager.py    # Docker container isolation
│   ├── evaluator.py         # Multi-metric evaluation engine
│   └── metrics.py           # Metrics collection and reporting
├── tests/
│   ├── conftest.py          # Pytest fixtures
│   ├── test_models.py       # Model tests
│   ├── test_evaluator.py    # Evaluator tests
│   └── test_metrics.py      # Metrics tests
├── .github/
│   ├── workflows/ci.yml     # CI/CD pipeline
│   └── ...                  # Issue templates, etc.
├── pyproject.toml           # Project configuration
└── Makefile                 # Development commands
```

## Development

```bash
# Install dev dependencies
uv sync

# Run tests
make test

# Run tests with coverage
make test-cov

# Run all quality checks
make quality

# Format code
make format

# Type check
make typecheck

# Security scan
make security
```

## Custom Metrics

Extend evaluation with custom metric plugins:

```python
from claude_spec_benchmark import MetricPlugin, Evaluator

class MyMetric(MetricPlugin):
    @property
    def name(self) -> str:
        return "my_metric"

    def compute(self, task, task_run, test_results):
        return {"score": 42}

evaluator = Evaluator(docker_manager)
evaluator.register_plugin(MyMetric())
```

## License

MIT
