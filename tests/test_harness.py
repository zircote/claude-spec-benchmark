"""Tests for SWE-bench harness integration module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from claude_spec_benchmark.harness import (
    HarnessResult,
    HarnessRunSummary,
    PredictionWriter,
    SWEBenchPrediction,
    convert_batch_to_jsonl,
    create_predictions_for_harness,
    generate_harness_command,
    load_predictions,
    validate_predictions,
)
from claude_spec_benchmark.models import (
    EvaluationResult,
    SDDBenchResult,
    SDDPhaseResult,
    SpecDegradationLevel,
    SWEBenchTask,
    TaskRun,
    TaskStatus,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_patch() -> str:
    """Sample unified diff patch."""
    return """diff --git a/file.py b/file.py
index abc1234..def5678 100644
--- a/file.py
+++ b/file.py
@@ -1,3 +1,4 @@
+# Fixed the bug
 def foo():
     pass
"""


@pytest.fixture
def sample_task_run(sample_patch: str) -> TaskRun:
    """Sample completed TaskRun with patch."""
    return TaskRun(
        task_id="django__django-11099",
        run_id="20251216_120000",
        status=TaskStatus.COMPLETED,
        generated_patch=sample_patch,
        exit_code=0,
    )


@pytest.fixture
def sample_task_run_no_patch() -> TaskRun:
    """Sample TaskRun without a patch."""
    return TaskRun(
        task_id="django__django-11100",
        run_id="20251216_120001",
        status=TaskStatus.FAILED,
        exit_code=1,
    )


@pytest.fixture
def sample_swe_task() -> SWEBenchTask:
    """Sample SWE-bench task."""
    return SWEBenchTask(
        instance_id="django__django-11099",
        repo="django/django",
        base_commit="abc123",
        problem_statement="Fix the bug",
        patch="diff --git...",
    )


@pytest.fixture
def sample_sdd_result(sample_patch: str) -> SDDBenchResult:
    """Sample SDD pipeline result."""
    return SDDBenchResult(
        instance_id="django__django-11099",
        degradation_level=SpecDegradationLevel.VAGUE,
        phase_results=[
            SDDPhaseResult(
                phase="degradation",
                success=True,
                duration_seconds=0.1,
            ),
            SDDPhaseResult(
                phase="implementation",
                success=True,
                duration_seconds=5.0,
                artifacts={"patch": sample_patch},
            ),
        ],
        final_status=EvaluationResult.PASS,
        total_duration_seconds=5.1,
    )


# =============================================================================
# SWEBenchPrediction Tests
# =============================================================================


class TestSWEBenchPrediction:
    """Tests for SWEBenchPrediction dataclass."""

    def test_to_dict(self, sample_patch: str):
        """Test conversion to dictionary."""
        pred = SWEBenchPrediction(
            instance_id="django__django-11099",
            model_name_or_path="claude-opus-4",
            model_patch=sample_patch,
        )
        result = pred.to_dict()

        assert result["instance_id"] == "django__django-11099"
        assert result["model_name_or_path"] == "claude-opus-4"
        assert result["model_patch"] == sample_patch

    def test_to_jsonl(self, sample_patch: str):
        """Test JSONL serialization."""
        pred = SWEBenchPrediction(
            instance_id="django__django-11099",
            model_name_or_path="claude-opus-4",
            model_patch=sample_patch,
        )
        jsonl = pred.to_jsonl()

        # Should be valid JSON
        parsed = json.loads(jsonl)
        assert parsed["instance_id"] == "django__django-11099"

    def test_to_jsonl_no_trailing_newline(self, sample_patch: str):
        """Test that to_jsonl doesn't include trailing newline."""
        pred = SWEBenchPrediction(
            instance_id="test",
            model_name_or_path="model",
            model_patch=sample_patch,
        )
        jsonl = pred.to_jsonl()
        assert not jsonl.endswith("\n")

    def test_from_task_run(self, sample_task_run: TaskRun):
        """Test creation from TaskRun."""
        pred = SWEBenchPrediction.from_task_run(
            sample_task_run,
            model_name="claude-opus-4",
        )

        assert pred.instance_id == sample_task_run.task_id
        assert pred.model_name_or_path == "claude-opus-4"
        assert pred.model_patch == sample_task_run.generated_patch

    def test_from_task_run_no_patch_raises(self, sample_task_run_no_patch: TaskRun):
        """Test that creating from TaskRun without patch raises ValueError."""
        with pytest.raises(ValueError, match="no generated patch"):
            SWEBenchPrediction.from_task_run(sample_task_run_no_patch)

    def test_from_sdd_result(self, sample_sdd_result: SDDBenchResult, sample_patch: str):
        """Test creation from SDDBenchResult."""
        pred = SWEBenchPrediction.from_sdd_result(
            sample_sdd_result,
            model_name="claude-sonnet-4",
        )

        assert pred is not None
        assert pred.instance_id == sample_sdd_result.instance_id
        assert pred.model_name_or_path == "claude-sonnet-4"
        assert pred.model_patch == sample_patch

    def test_from_sdd_result_no_patch(self):
        """Test that from_sdd_result returns None if no patch."""
        result = SDDBenchResult(
            instance_id="test",
            degradation_level=SpecDegradationLevel.VAGUE,
            phase_results=[
                SDDPhaseResult(
                    phase="implementation",
                    success=False,
                    duration_seconds=0.0,
                ),
            ],
            final_status=EvaluationResult.FAIL,
        )

        pred = SWEBenchPrediction.from_sdd_result(result)
        assert pred is None


# =============================================================================
# PredictionWriter Tests
# =============================================================================


class TestPredictionWriter:
    """Tests for PredictionWriter class."""

    def test_write_predictions(self, sample_patch: str):
        """Test writing predictions to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "predictions.jsonl"

            with PredictionWriter(output_path, "claude-code") as writer:
                pred = SWEBenchPrediction(
                    instance_id="test-1",
                    model_name_or_path="claude-code",
                    model_patch=sample_patch,
                )
                writer.write(pred)
                assert writer.count == 1

            # Verify file contents
            content = output_path.read_text()
            lines = content.strip().split("\n")
            assert len(lines) == 1

            parsed = json.loads(lines[0])
            assert parsed["instance_id"] == "test-1"

    def test_write_multiple_predictions(self, sample_patch: str):
        """Test writing multiple predictions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "predictions.jsonl"

            with PredictionWriter(output_path, "model") as writer:
                for i in range(5):
                    pred = SWEBenchPrediction(
                        instance_id=f"test-{i}",
                        model_name_or_path="model",
                        model_patch=sample_patch,
                    )
                    writer.write(pred)

            # Verify file
            lines = output_path.read_text().strip().split("\n")
            assert len(lines) == 5

    def test_write_task_run(self, sample_task_run: TaskRun):
        """Test writing from TaskRun."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "preds.jsonl"

            with PredictionWriter(output_path, "claude") as writer:
                success = writer.write_task_run(sample_task_run)
                assert success is True
                assert writer.count == 1

    def test_write_task_run_skips_no_patch(self, sample_task_run_no_patch: TaskRun):
        """Test that write_task_run skips runs without patches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "preds.jsonl"

            with PredictionWriter(output_path, "claude") as writer:
                success = writer.write_task_run(sample_task_run_no_patch)
                assert success is False
                assert writer.count == 0

    def test_append_mode(self, sample_patch: str):
        """Test append mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "preds.jsonl"

            # Write initial
            with PredictionWriter(output_path, "model") as writer:
                writer.write(SWEBenchPrediction("test-1", "model", sample_patch))

            # Append more
            with PredictionWriter(output_path, "model", append=True) as writer:
                writer.write(SWEBenchPrediction("test-2", "model", sample_patch))

            # Should have 2 lines
            lines = output_path.read_text().strip().split("\n")
            assert len(lines) == 2

    def test_creates_parent_directories(self, sample_patch: str):
        """Test that parent directories are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir" / "nested" / "preds.jsonl"

            with PredictionWriter(output_path, "model") as writer:
                writer.write(SWEBenchPrediction("test", "model", sample_patch))

            assert output_path.exists()


# =============================================================================
# load_predictions Tests
# =============================================================================


class TestLoadPredictions:
    """Tests for load_predictions function."""

    def test_load_valid_predictions(self, sample_patch: str):
        """Test loading valid predictions file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "preds.jsonl"

            # Write test data
            preds_data = [
                {"instance_id": "test-1", "model_name_or_path": "m", "model_patch": sample_patch},
                {"instance_id": "test-2", "model_name_or_path": "m", "model_patch": sample_patch},
            ]
            with path.open("w") as f:
                for p in preds_data:
                    f.write(json.dumps(p) + "\n")

            # Load and verify
            preds = list(load_predictions(path))
            assert len(preds) == 2
            assert preds[0].instance_id == "test-1"
            assert preds[1].instance_id == "test-2"

    def test_load_skips_empty_lines(self, sample_patch: str):
        """Test that empty lines are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "preds.jsonl"

            content = (
                '{"instance_id": "t1", "model_name_or_path": "m", "model_patch": "p"}\n'
                "\n"
                '{"instance_id": "t2", "model_name_or_path": "m", "model_patch": "p"}\n'
            )
            path.write_text(content)

            preds = list(load_predictions(path))
            assert len(preds) == 2


# =============================================================================
# validate_predictions Tests
# =============================================================================


class TestValidatePredictions:
    """Tests for validate_predictions function."""

    def test_valid_predictions(self, sample_patch: str):
        """Test validation of valid predictions file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "preds.jsonl"

            data = {"instance_id": "test", "model_name_or_path": "m", "model_patch": sample_patch}
            path.write_text(json.dumps(data) + "\n")

            result = validate_predictions(path)
            assert result["valid"] is True
            assert result["valid_predictions"] == 1
            assert len(result["errors"]) == 0

    def test_missing_file(self):
        """Test validation of non-existent file."""
        result = validate_predictions("/nonexistent/path.jsonl")
        assert result["valid"] is False
        assert "File not found" in result["errors"][0]

    def test_invalid_json(self):
        """Test validation of invalid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "preds.jsonl"
            path.write_text("not valid json\n")

            result = validate_predictions(path)
            assert result["valid"] is False
            assert "Invalid JSON" in result["errors"][0]

    def test_missing_fields(self):
        """Test validation with missing required fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "preds.jsonl"
            path.write_text('{"instance_id": "test"}\n')

            result = validate_predictions(path)
            assert result["valid"] is False
            assert "Missing fields" in result["errors"][0]


# =============================================================================
# convert_batch_to_jsonl Tests
# =============================================================================


class TestConvertBatchToJsonl:
    """Tests for convert_batch_to_jsonl function."""

    def test_convert_batch(self, sample_task_run: TaskRun, sample_task_run_no_patch: TaskRun):
        """Test converting a batch of TaskRuns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.jsonl"

            task_runs = [sample_task_run, sample_task_run_no_patch, sample_task_run]
            count = convert_batch_to_jsonl(task_runs, output_path, "claude")

            # Should only write runs with patches (2 out of 3)
            assert count == 2

            lines = output_path.read_text().strip().split("\n")
            assert len(lines) == 2


# =============================================================================
# create_predictions_for_harness Tests
# =============================================================================


class TestCreatePredictionsForHarness:
    """Tests for create_predictions_for_harness function."""

    def test_create_predictions_directory(
        self,
        sample_swe_task: SWEBenchTask,
        sample_task_run: TaskRun,
    ):
        """Test creating predictions file in directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"

            task_runs = {sample_swe_task.instance_id: sample_task_run}
            result_path = create_predictions_for_harness(
                tasks=[sample_swe_task],
                task_runs=task_runs,
                output_path=output_dir,
                model_name="claude-opus-4",
            )

            assert result_path == output_dir / "predictions.jsonl"
            assert result_path.exists()

            # Verify content
            content = result_path.read_text()
            parsed = json.loads(content.strip())
            assert parsed["instance_id"] == sample_swe_task.instance_id
            assert parsed["model_name_or_path"] == "claude-opus-4"

    def test_create_predictions_skips_missing_patches(
        self,
        sample_swe_task: SWEBenchTask,
        sample_task_run_no_patch: TaskRun,
    ):
        """Test that tasks without patches are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            task_runs = {sample_swe_task.instance_id: sample_task_run_no_patch}
            result_path = create_predictions_for_harness(
                tasks=[sample_swe_task],
                task_runs=task_runs,
                output_path=output_dir,
            )

            # File should exist but be empty
            assert result_path.exists()
            assert result_path.read_text() == ""


# =============================================================================
# generate_harness_command Tests
# =============================================================================


class TestGenerateHarnessCommand:
    """Tests for generate_harness_command function."""

    def test_default_command(self):
        """Test generating command with defaults."""
        cmd = generate_harness_command(
            "predictions.jsonl",
            run_id="test_run",
        )

        assert "python" in cmd
        assert "-m" in cmd
        assert "swebench.harness.run_assessment" in cmd
        assert "--predictions_path" in cmd
        assert "predictions.jsonl" in cmd
        assert "--run_id" in cmd
        assert "test_run" in cmd

    def test_custom_options(self):
        """Test generating command with custom options."""
        cmd = generate_harness_command(
            "preds.jsonl",
            dataset="custom/dataset",
            split="dev",
            max_workers=8,
            run_id="custom_run",
        )

        assert "custom/dataset" in cmd
        assert "dev" in cmd
        assert "8" in cmd
        assert "custom_run" in cmd


# =============================================================================
# HarnessResult and HarnessRunSummary Tests
# =============================================================================


class TestHarnessModels:
    """Tests for harness result models."""

    def test_harness_result(self):
        """Test HarnessResult creation."""
        result = HarnessResult(
            instance_id="test",
            resolved=True,
            apply_status="success",
            test_status="passed",
            duration_seconds=10.5,
        )

        assert result.instance_id == "test"
        assert result.resolved is True
        assert result.apply_status == "success"

    def test_harness_run_summary_resolve_rate(self):
        """Test HarnessRunSummary resolve rate calculation."""
        summary = HarnessRunSummary(
            total=10,
            resolved=7,
            unresolved=3,
        )

        assert summary.resolve_rate == 0.7

    def test_harness_run_summary_empty(self):
        """Test resolve rate with no tasks."""
        summary = HarnessRunSummary()
        assert summary.resolve_rate == 0.0
