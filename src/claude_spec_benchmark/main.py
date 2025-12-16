"""CLI entry point for claude-spec-benchmark.

Commands:
    run       - Execute benchmark on tasks
    list      - List available tasks
    info      - Display harness information
    report    - Generate reports from results
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn

from claude_spec_benchmark.models import (
    BenchmarkConfig,
    ElicitationMetrics,
    EvaluationMetrics,
    SDDBenchResult,
    SDDPhaseResult,
    SpecDegradationLevel,
    SWEBenchTask,
)

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
    console.print("  * Load SWE-bench Lite tasks (300 curated tasks)")
    console.print("  * Execute Claude Code CLI for patch generation")
    console.print("  * Docker container isolation for safe execution")
    console.print("  * Multi-metric evaluation (tests, diff, custom)")
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

    for count, task in enumerate(loader.iter_tasks(repos=[repo] if repo else None)):
        if count >= limit:
            console.print(f"\n[dim]...and {len(loader) - limit} more tasks[/dim]")
            break
        table.add_row(task.instance_id, task.repo, task.version or "-")

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
        task_list = list(
            loader.iter_tasks(
                task_ids=task_ids,
                repos=[repo] if repo else None,
            )
        )

    if limit > 0:
        task_list = task_list[:limit]

    config = BenchmarkConfig(
        task_ids=[t.instance_id for t in task_list],
        timeout_seconds=timeout,
        max_workers=workers,
        output_dir=Path(output),
    )

    console.print("\n[bold]Benchmark Configuration:[/bold]")
    console.print(f"  Tasks: {len(task_list)}")
    console.print(f"  Timeout: {timeout}s per task")
    console.print(f"  Workers: {workers}")
    console.print(f"  Output: {output}")
    if model:
        console.print(f"  Model: {model}")

    if dry_run:
        console.print("\n[yellow]DRY RUN - no tasks executed[/yellow]")
        for task in task_list[:10]:
            console.print(f"  * {task.instance_id}")
        if len(task_list) > 10:
            console.print(f"  ...and {len(task_list) - 10} more")
        return

    # Run benchmark
    console.print("\n[bold green]Starting benchmark...[/bold green]")
    asyncio.run(_run_benchmark(task_list, config, model))


async def _run_benchmark(
    tasks: list[SWEBenchTask],
    config: BenchmarkConfig,
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
                # Use config output_dir for workspace to avoid hardcoded temp paths
                task_workspace = config.output_dir / task.instance_id.replace("/", "-")
                task_run = await runner.run_task(task, task_workspace)

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


# =============================================================================
# SDD-Bench Commands
# =============================================================================


@main.group()
def sdd() -> None:
    """SDD-Bench: Spec-Driven Development benchmarking commands."""


@sdd.command("degrade")
@click.argument("issue_file", type=click.Path(exists=True))
@click.option(
    "--level",
    type=click.Choice(["full", "partial", "vague", "minimal", "ambiguous"]),
    default="vague",
    help="Degradation level to apply",
)
@click.option("--seed", type=int, default=None, help="Random seed for reproducibility")
@click.option("--repo", default=None, help="Repository name for repo-specific patterns")
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Output file (default: stdout)",
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format",
)
@click.option("--show-hidden", is_flag=True, help="Show hidden details in output")
def degrade_cmd(
    issue_file: str,
    level: str,
    seed: int | None,
    repo: str | None,
    output: str | None,
    fmt: str,
    show_hidden: bool,
) -> None:
    """Degrade a specification to simulate incomplete requirements.

    Reads an issue/spec from ISSUE_FILE and produces a degraded version
    with technical details removed based on the degradation level.

    Example:
        claude-spec-benchmark sdd degrade issue.txt --level vague --seed 42
    """
    from claude_spec_benchmark.degradation.engine import DegradationEngine

    # Read input
    issue_text = Path(issue_file).read_text()

    # Map level string to enum
    level_map = {
        "full": SpecDegradationLevel.FULL,
        "partial": SpecDegradationLevel.PARTIAL,
        "vague": SpecDegradationLevel.VAGUE,
        "minimal": SpecDegradationLevel.MINIMAL,
        "ambiguous": SpecDegradationLevel.AMBIGUOUS,
    }
    degradation_level = level_map[level]

    # Create engine and degrade
    engine = DegradationEngine.from_repo(repo) if repo else DegradationEngine()
    result = engine.degrade(issue_text, degradation_level, seed=seed, repo=repo)

    # Format output
    if fmt == "json":
        output_data: dict[str, Any] = {
            "degraded_text": result.degraded_text,
            "level": result.level.value,
            "seed": result.seed,
            "hidden_count": len(result.hidden_details),
        }
        if show_hidden:
            output_data["hidden_details"] = result.hidden_details
        output_text = json.dumps(output_data, indent=2)
    else:
        output_text = result.degraded_text
        if show_hidden:
            output_text += "\n\n--- Hidden Details ---\n"
            for detail in result.hidden_details:
                output_text += f"  * {detail}\n"

    # Write output
    if output:
        Path(output).write_text(output_text)
        console.print(f"[green]Degraded spec written to {output}[/green]")
        console.print(
            f"[dim]Level: {level}, Hidden details: {len(result.hidden_details)}[/dim]"
        )
    else:
        console.print(output_text)


@sdd.command("run")
@click.option(
    "--framework",
    type=click.Choice(["passthrough", "claude-code"]),
    default="passthrough",
    help="SDD framework to evaluate",
)
@click.option(
    "--degradation",
    type=click.Choice(["full", "partial", "vague", "minimal", "ambiguous"]),
    default="vague",
    help="Specification degradation level",
)
@click.option("--limit", default=0, help="Max tasks to run (0 = all)")
@click.option("--parallel", default=1, help="Parallel task executions")
@click.option("--tasks", help="Comma-separated task IDs to run")
@click.option("--output", default="./sdd-results", help="Output directory")
@click.option(
    "--skip-phases",
    help="Comma-separated phases to skip (degrade,elicit,parse,test,implement,refine,validate)",
)
@click.option("--dry-run", is_flag=True, help="Show config without executing")
def sdd_run_cmd(
    framework: str,
    degradation: str,
    limit: int,
    parallel: int,
    tasks: str | None,
    output: str,
    skip_phases: str | None,
    dry_run: bool,
) -> None:
    """Run SDD-Bench evaluation pipeline.

    Executes the full spec-driven development evaluation:
    1. Degrade specification to simulate incomplete requirements
    2. Elicit requirements through oracle dialogue
    3. Parse specification into structured requirements
    4. Generate tests (TDD Red)
    5. Implement solution (TDD Green)
    6. Refine implementation
    7. Validate against SWE-bench

    Example:
        claude-spec-benchmark sdd run --framework passthrough --limit 10
    """
    from claude_spec_benchmark.frameworks import (
        ClaudeCodeFramework,
        PassthroughFramework,
    )
    from claude_spec_benchmark.runner import ClaudeCodeRunner
    from claude_spec_benchmark.sdd_runner import SDDBenchRunner

    # Map strings to enums
    level_map = {
        "full": SpecDegradationLevel.FULL,
        "partial": SpecDegradationLevel.PARTIAL,
        "vague": SpecDegradationLevel.VAGUE,
        "minimal": SpecDegradationLevel.MINIMAL,
        "ambiguous": SpecDegradationLevel.AMBIGUOUS,
    }
    deg_level = level_map[degradation]

    # Create framework
    fw: PassthroughFramework | ClaudeCodeFramework
    if framework == "passthrough":
        fw = PassthroughFramework()
    else:  # claude-code
        try:
            runner = ClaudeCodeRunner()
            fw = ClaudeCodeFramework(runner)
        except Exception as e:
            console.print(f"[red]Failed to initialize Claude Code runner: {e}[/red]")
            console.print("[dim]Ensure 'claude' CLI is installed and in PATH[/dim]")
            return

    # Parse options
    task_ids = [t.strip() for t in tasks.split(",")] if tasks else None
    skip_set = skip_phases.split(",") if skip_phases else None
    task_limit = limit if limit > 0 else None

    # Display config
    console.print("\n[bold]SDD-Bench Configuration:[/bold]")
    console.print(f"  Framework: {framework}")
    console.print(f"  Degradation: {degradation}")
    console.print(f"  Limit: {task_limit or 'all'}")
    console.print(f"  Parallel: {parallel}")
    console.print(f"  Output: {output}")
    if skip_set:
        console.print(f"  Skip phases: {', '.join(skip_set)}")

    if dry_run:
        console.print("\n[yellow]DRY RUN - no tasks executed[/yellow]")
        return

    # Create runner
    sdd_runner = SDDBenchRunner(
        framework=fw,
        degradation_level=deg_level,
    )

    # Run
    console.print("\n[bold green]Starting SDD-Bench evaluation...[/bold green]")
    results = asyncio.run(
        sdd_runner.run(
            limit=task_limit,
            parallel=parallel,
            task_ids=task_ids,
            skip_phases=skip_set,
            output_dir=Path(output),
        )
    )

    # Summary
    passed = sum(1 for r in results if r.final_status.value == "pass")
    total = len(results)
    console.print(f"\n[bold]Results:[/bold] {passed}/{total} passed ({100*passed/total:.1f}%)")
    console.print(f"[green]Results saved to {output}[/green]")


@sdd.command("report")
@click.argument("results_dir", type=click.Path(exists=True))
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["html", "markdown", "json", "csv"]),
    default="html",
    help="Output format",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Output file (default: auto-generated in results_dir)",
)
@click.option(
    "--compare",
    type=click.Path(exists=True),
    default=None,
    help="Path to baseline results for A/B comparison",
)
@click.option(
    "--title",
    default="SDD-Bench Evaluation Report",
    help="Report title",
)
def sdd_report_cmd(
    results_dir: str,
    fmt: str,
    output: str | None,
    compare: str | None,
    title: str,
) -> None:
    """Generate report from SDD-Bench evaluation results.

    Reads SDD results from RESULTS_DIR and generates a report in the specified format.
    Supports comparison mode for A/B testing different frameworks.

    Examples:
        claude-spec-benchmark sdd report ./sdd-results --format html
        claude-spec-benchmark sdd report ./sdd-results --format markdown -o report.md
        claude-spec-benchmark sdd report ./experiment --compare ./baseline --format markdown
    """
    from claude_spec_benchmark.reporting import SDDReportGenerator

    # Find results file
    results_path = Path(results_dir)
    sdd_metrics_file = results_path / "sdd_metrics.json"
    metrics_file = results_path / "metrics.json"

    if sdd_metrics_file.exists():
        try:
            data = json.loads(sdd_metrics_file.read_text())
        except json.JSONDecodeError as e:
            console.print(f"[red]Failed to parse {sdd_metrics_file}: {e}[/red]")
            console.print("[dim]The file contains invalid JSON. Please check the file format.[/dim]")
            return
        results_key = "sdd_results"
    elif metrics_file.exists():
        try:
            data = json.loads(metrics_file.read_text())
        except json.JSONDecodeError as e:
            console.print(f"[red]Failed to parse {metrics_file}: {e}[/red]")
            console.print("[dim]The file contains invalid JSON. Please check the file format.[/dim]")
            return
        results_key = "results" if "results" in data else "sdd_results"
    else:
        console.print(f"[red]No metrics file found in {results_dir}[/red]")
        console.print("[dim]Expected sdd_metrics.json or metrics.json[/dim]")
        return

    # Parse results
    def parse_results(data_dict: dict[str, Any], key: str) -> list[SDDBenchResult]:
        results = []
        for r in data_dict.get(key, []):
            # Handle enum conversion with error handling
            try:
                deg_level = SpecDegradationLevel(r["degradation_level"])
            except (ValueError, KeyError) as e:
                console.print(
                    f"[yellow]Warning: Invalid degradation level for {r.get('instance_id', 'unknown')}: {e}[/yellow]"
                )
                console.print("[dim]Skipping this result. Valid levels: full, partial, vague, minimal, ambiguous[/dim]")
                continue
            # Parse phase results
            phases = [
                SDDPhaseResult(
                    phase=p["phase"],
                    success=p["success"],
                    duration_seconds=p.get("duration_seconds", 0.0),
                    error=p.get("error"),
                    artifacts=p.get("artifacts", {}),
                )
                for p in r.get("phase_results", [])
            ]
            # Parse elicitation metrics if present
            elicit = None
            if r.get("elicitation_metrics"):
                em = r["elicitation_metrics"]
                elicit = ElicitationMetrics(
                    discovery_rate=em.get("discovery_rate", 0.0),
                    question_efficiency=em.get("question_efficiency", 0.0),
                    total_questions=em.get("total_questions", 0),
                    question_distribution=em.get("question_distribution", {}),
                    revealed_requirements=em.get("revealed_requirements", []),
                    hidden_requirements=em.get("hidden_requirements", []),
                    avg_relevance_score=em.get("avg_relevance_score", 0.0),
                )
            # Parse evaluation metrics if present
            impl_metrics = None
            if r.get("implementation_metrics"):
                impl_metrics = EvaluationMetrics(**r["implementation_metrics"])

            results.append(
                SDDBenchResult(
                    instance_id=r["instance_id"],
                    degradation_level=deg_level,
                    phase_results=phases,
                    final_status=r["final_status"],
                    elicitation_metrics=elicit,
                    implementation_metrics=impl_metrics,
                    total_duration_seconds=r.get("total_duration_seconds", 0.0),
                )
            )
        return results

    results = parse_results(data, results_key)
    if not results:
        console.print("[yellow]No results found in file[/yellow]")
        return

    # Create generator
    generator = SDDReportGenerator()

    # Handle comparison mode
    if compare:
        compare_path = Path(compare)
        baseline_file = compare_path / "sdd_metrics.json"
        if not baseline_file.exists():
            baseline_file = compare_path / "metrics.json"
        if not baseline_file.exists():
            console.print(f"[red]No metrics file found in {compare}[/red]")
            return

        try:
            baseline_data = json.loads(baseline_file.read_text())
        except json.JSONDecodeError as e:
            console.print(f"[red]Failed to parse {baseline_file}: {e}[/red]")
            console.print("[dim]The baseline file contains invalid JSON. Please check the file format.[/dim]")
            return
        baseline_key = "sdd_results" if "sdd_results" in baseline_data else "results"
        baseline_results = parse_results(baseline_data, baseline_key)

        # Generate comparison report (markdown only)
        comparison = generator.generate_comparison_report(
            baseline_results,
            results,
            baseline_name=compare_path.name,
            experiment_name=results_path.name,
        )

        if output:
            Path(output).write_text(comparison)
            console.print(f"[green]Comparison report written to {output}[/green]")
        else:
            console.print(comparison)
        return

    # Determine output path
    if output:
        out_path = Path(output)
    else:
        ext_map = {"html": "html", "markdown": "md", "json": "json", "csv": "csv"}
        out_path = results_path / f"report.{ext_map[fmt]}"

    # Generate report
    if fmt == "html":
        generator.generate_html_report(results, out_path, title=title)
        console.print(f"[green]HTML report written to {out_path}[/green]")
        console.print(f"[dim]Open in browser: file://{out_path.absolute()}[/dim]")
    elif fmt == "markdown":
        md = generator.generate_markdown_report(results, title=title)
        out_path.write_text(md)
        console.print(f"[green]Markdown report written to {out_path}[/green]")
    elif fmt == "json":
        generator.export_json(results, out_path)
        console.print(f"[green]JSON export written to {out_path}[/green]")
    elif fmt == "csv":
        generator.export_csv(results, out_path)
        console.print(f"[green]CSV export written to {out_path}[/green]")


@sdd.command("extract")
@click.argument("issue_file", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Output file (default: stdout)",
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["json", "text"]),
    default="json",
    help="Output format",
)
@click.option(
    "--max-requirements",
    type=int,
    default=30,
    help="Maximum requirements to extract",
)
def extract_cmd(
    issue_file: str,
    output: str | None,
    fmt: str,
    max_requirements: int,
) -> None:
    """Extract requirements from an issue specification.

    Parses an issue/spec from ISSUE_FILE and extracts atomic requirements
    that can be used for elicitation simulation and scoring.

    Example:
        claude-spec-benchmark sdd extract issue.txt --format json
    """
    from claude_spec_benchmark.elicitation.extraction import RequirementsExtractor

    # Read input
    issue_text = Path(issue_file).read_text()

    # Extract requirements
    extractor = RequirementsExtractor(max_requirements=max_requirements)
    requirements = extractor.extract(issue_text)

    # Format output
    if fmt == "json":
        output_data = {
            "total": len(requirements),
            "summary": extractor.get_summary(requirements),
            "requirements": [
                {
                    "id": req.id,
                    "text": req.text,
                    "category": req.category,
                    "keywords": req.keywords,
                    "discoverable": req.discoverable,
                }
                for req in requirements
            ],
        }
        output_text = json.dumps(output_data, indent=2)
    else:
        lines = [f"Extracted {len(requirements)} requirements:", ""]
        summary = extractor.get_summary(requirements)
        lines.append(
            f"  Functional: {summary['functional']}, "
            f"Non-functional: {summary['non-functional']}, "
            f"Constraints: {summary['constraint']}"
        )
        lines.append("")
        for req in requirements:
            lines.append(f"[{req.id}] ({req.category})")
            lines.append(f"  {req.text}")
            if req.keywords:
                lines.append(f"  Keywords: {', '.join(req.keywords[:5])}")
            lines.append("")
        output_text = "\n".join(lines)

    # Write output
    if output:
        Path(output).write_text(output_text)
        console.print(f"[green]Requirements written to {output}[/green]")
        console.print(f"[dim]Extracted {len(requirements)} requirements[/dim]")
    else:
        console.print(output_text)


@main.command()
@click.argument("results_dir", type=click.Path(exists=True))
@click.option(
    "--format", "fmt", default="table", type=click.Choice(["table", "markdown", "json"])
)
def report(results_dir: str, fmt: str) -> None:
    """Generate report from benchmark results."""
    from claude_spec_benchmark.metrics import ReportGenerator
    from claude_spec_benchmark.models import AggregateMetrics

    results_path = Path(results_dir) / "metrics.json"
    if not results_path.exists():
        console.print(f"[red]No metrics.json found in {results_dir}[/red]")
        return

    try:
        data = json.loads(results_path.read_text())
    except json.JSONDecodeError as e:
        console.print(f"[red]Failed to parse {results_path}: {e}[/red]")
        console.print("[dim]The file contains invalid JSON. Please check the file format.[/dim]")
        return

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
