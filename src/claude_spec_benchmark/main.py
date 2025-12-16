"""CLI entry point for claude-spec-benchmark.

Commands:
    run       - Execute benchmark on tasks
    list      - List available tasks
    info      - Display harness information
    report    - Generate reports from results
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


def setup_logging(verbose: bool) -> None:
    """Configure logging with Rich handler."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


@click.group()
@click.version_option()
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output")
def main(verbose: bool) -> None:
    """SWE-bench test harness for evaluating Claude's software engineering capabilities."""
    setup_logging(verbose)


@main.command()
def info() -> None:
    """Display information about the benchmark harness."""
    from claude_spec_benchmark import __version__

    console.print(f"[bold blue]claude-spec-benchmark[/bold blue] v{__version__}")
    console.print()
    console.print("[bold]Capabilities:[/bold]")
    console.print("  • Load SWE-bench Lite tasks (300 curated tasks)")
    console.print("  • Execute Claude Code CLI for patch generation")
    console.print("  • Docker container isolation for safe execution")
    console.print("  • Multi-metric evaluation (tests, diff, custom)")
    console.print()
    console.print("[bold]Usage:[/bold]")
    console.print("  claude-spec-benchmark list         # List tasks")
    console.print("  claude-spec-benchmark run          # Run benchmark")
    console.print("  claude-spec-benchmark report       # Generate report")


@main.command("list")
@click.option("--repo", help="Filter by repository name")
@click.option("--limit", default=20, help="Maximum tasks to show")
def list_tasks(repo: str | None, limit: int) -> None:
    """List available SWE-bench tasks."""
    from rich.table import Table

    from claude_spec_benchmark.task_loader import TaskLoader

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Loading tasks...", total=None)
        loader = TaskLoader()
        stats = loader.stats()

    console.print(f"\n[bold]Dataset:[/bold] {stats['dataset']}")
    console.print(f"[bold]Total tasks:[/bold] {stats['total_tasks']}")
    console.print(f"[bold]Repositories:[/bold] {stats['unique_repos']}")
    console.print()

    table = Table(title="Tasks")
    table.add_column("Instance ID", style="cyan")
    table.add_column("Repository", style="green")
    table.add_column("Version")

    count = 0
    for task in loader.iter_tasks(repos=[repo] if repo else None):
        if count >= limit:
            console.print(f"\n[dim]...and {len(loader) - limit} more tasks[/dim]")
            break
        table.add_row(task.instance_id, task.repo, task.version or "-")
        count += 1

    console.print(table)


@main.command()
@click.option("--tasks", help="Comma-separated task IDs to run (default: all)")
@click.option("--repo", help="Run only tasks from this repository")
@click.option("--limit", default=0, help="Max tasks to run (0 = all)")
@click.option("--timeout", default=1800, help="Per-task timeout in seconds")
@click.option("--workers", default=4, help="Parallel workers")
@click.option("--output", default="./results", help="Output directory")
@click.option("--model", help="Claude model override (e.g., opus-4)")
@click.option("--dry-run", is_flag=True, help="Show what would run without executing")
def run(
    tasks: str | None,
    repo: str | None,
    limit: int,
    timeout: int,
    workers: int,
    output: str,
    model: str | None,
    dry_run: bool,
) -> None:
    """Execute benchmark on SWE-bench tasks."""
    from claude_spec_benchmark.models import BenchmarkConfig
    from claude_spec_benchmark.task_loader import TaskLoader

    # Parse task IDs
    task_ids = [t.strip() for t in tasks.split(",")] if tasks else None

    # Load tasks
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Loading tasks...", total=None)
        loader = TaskLoader()
        task_list = list(loader.iter_tasks(
            task_ids=task_ids,
            repos=[repo] if repo else None,
        ))

    if limit > 0:
        task_list = task_list[:limit]

    config = BenchmarkConfig(
        task_ids=[t.instance_id for t in task_list],
        timeout_seconds=timeout,
        max_workers=workers,
        output_dir=Path(output),
    )

    console.print(f"\n[bold]Benchmark Configuration:[/bold]")
    console.print(f"  Tasks: {len(task_list)}")
    console.print(f"  Timeout: {timeout}s per task")
    console.print(f"  Workers: {workers}")
    console.print(f"  Output: {output}")
    if model:
        console.print(f"  Model: {model}")

    if dry_run:
        console.print("\n[yellow]DRY RUN - no tasks executed[/yellow]")
        for task in task_list[:10]:
            console.print(f"  • {task.instance_id}")
        if len(task_list) > 10:
            console.print(f"  ...and {len(task_list) - 10} more")
        return

    # Run benchmark
    console.print("\n[bold green]Starting benchmark...[/bold green]")
    asyncio.run(_run_benchmark(task_list, config, model))


async def _run_benchmark(
    tasks: list,
    config,
    model: str | None,
) -> None:
    """Execute benchmark asynchronously."""
    from claude_spec_benchmark.docker_manager import DockerManager
    from claude_spec_benchmark.evaluator import Evaluator
    from claude_spec_benchmark.metrics import MetricsCollector, ReportGenerator
    from claude_spec_benchmark.runner import ClaudeCodeRunner

    collector = MetricsCollector()
    reporter = ReportGenerator(console)

    try:
        runner = ClaudeCodeRunner(
            timeout_seconds=config.timeout_seconds,
            model=model,
        )
    except Exception as e:
        console.print(f"[red]Failed to initialize runner: {e}[/red]")
        console.print("[dim]Ensure 'claude' CLI is installed and in PATH[/dim]")
        return

    docker = DockerManager()
    evaluator = Evaluator(docker)

    with Progress(console=console) as progress:
        task_progress = progress.add_task(
            "[cyan]Running tasks...",
            total=len(tasks),
        )

        for task in tasks:
            progress.update(task_progress, description=f"[cyan]{task.instance_id}")

            try:
                # Create container
                container_id = await docker.create_task_container(task)

                # Run Claude Code
                task_run = await runner.run_task(
                    task,
                    Path("/tmp") / task.instance_id.replace("/", "-"),
                )

                # Apply generated patch if present
                if task_run.generated_patch:
                    await docker.apply_patch(container_id, task_run.generated_patch)

                # Evaluate
                metrics = await evaluator.evaluate(task, task_run, container_id)
                collector.add_evaluation(metrics, task_run.duration_seconds)

                # Cleanup
                await docker.cleanup(container_id)

            except Exception as e:
                console.print(f"[red]Error on {task.instance_id}: {e}[/red]")

            progress.advance(task_progress)

    # Generate reports
    aggregate = collector.compute_aggregate()
    reporter.print_summary(aggregate)

    # Save results
    config.output_dir.mkdir(parents=True, exist_ok=True)
    collector.save_report(config.output_dir / "metrics.json")
    reporter.save_markdown_report(
        config.output_dir / "REPORT.md",
        aggregate,
        collector._evaluations,
    )

    console.print(f"\n[green]Results saved to {config.output_dir}[/green]")


@main.command()
@click.argument("results_dir", type=click.Path(exists=True))
@click.option("--format", "fmt", default="table", type=click.Choice(["table", "markdown", "json"]))
def report(results_dir: str, fmt: str) -> None:
    """Generate report from benchmark results."""
    import json

    from claude_spec_benchmark.metrics import ReportGenerator
    from claude_spec_benchmark.models import AggregateMetrics, EvaluationMetrics

    results_path = Path(results_dir) / "metrics.json"
    if not results_path.exists():
        console.print(f"[red]No metrics.json found in {results_dir}[/red]")
        return

    data = json.loads(results_path.read_text())
    aggregate = AggregateMetrics(**data["aggregate"])
    evaluations = [EvaluationMetrics(**e) for e in data["evaluations"]]

    reporter = ReportGenerator(console)

    if fmt == "table":
        reporter.print_summary(aggregate)
        reporter.print_task_results(evaluations, show_all=True)
    elif fmt == "markdown":
        md = reporter.generate_markdown_report(aggregate, evaluations)
        console.print(md)
    elif fmt == "json":
        console.print_json(data=data)


if __name__ == "__main__":
    main()
