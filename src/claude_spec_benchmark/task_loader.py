"""Task loader for SWE-bench Lite dataset.

Loads tasks from HuggingFace datasets and provides filtering/iteration.
"""

import logging
from collections.abc import Iterator
from functools import lru_cache
from typing import Any

from datasets import Dataset, load_dataset
from pydantic import ValidationError

from claude_spec_benchmark.models import SWEBenchTask

logger = logging.getLogger(__name__)


class TaskLoadError(Exception):
    """Raised when task loading fails."""


class TaskLoader:
    """Loads and manages SWE-bench Lite tasks.

    Example:
        >>> loader = TaskLoader()
        >>> for task in loader.iter_tasks():
        ...     print(task.instance_id)
    """

    DEFAULT_DATASET = "princeton-nlp/SWE-bench_Lite"
    DEFAULT_SPLIT = "test"

    def __init__(
        self,
        dataset_name: str = DEFAULT_DATASET,
        split: str = DEFAULT_SPLIT,
        cache_dir: str | None = None,
    ) -> None:
        """Initialize task loader.

        Args:
            dataset_name: HuggingFace dataset identifier.
            split: Dataset split to load (test, dev, train).
            cache_dir: Optional cache directory for downloaded data.
        """
        self._dataset_name = dataset_name
        self._split = split
        self._cache_dir = cache_dir
        self._dataset: Dataset | None = None

    @property
    def dataset(self) -> Dataset:
        """Lazy-load the dataset on first access."""
        if self._dataset is None:
            self._dataset = self._load_dataset()
        return self._dataset

    def _load_dataset(self) -> Dataset:
        """Load dataset from HuggingFace."""
        try:
            logger.info(
                "Loading dataset %s (split=%s)",
                self._dataset_name,
                self._split,
            )
            ds = load_dataset(
                self._dataset_name,
                split=self._split,
                cache_dir=self._cache_dir,
            )
            if not isinstance(ds, Dataset):
                msg = f"Expected Dataset, got {type(ds)}"
                raise TaskLoadError(msg)
            logger.info("Loaded %d tasks", len(ds))
            return ds
        except Exception as e:
            msg = f"Failed to load dataset: {e}"
            raise TaskLoadError(msg) from e

    @lru_cache(maxsize=512)
    def get_task(self, instance_id: str) -> SWEBenchTask:
        """Get a specific task by instance ID.

        Args:
            instance_id: Unique task identifier.

        Returns:
            The task with the given ID.

        Raises:
            KeyError: If task not found.
            ValidationError: If task data is invalid.
        """
        for row in self.dataset:
            if row["instance_id"] == instance_id:  # type: ignore[index]
                return self._row_to_task(row)  # type: ignore[arg-type]
        msg = f"Task not found: {instance_id}"
        raise KeyError(msg)

    def _row_to_task(self, row: dict[str, Any]) -> SWEBenchTask:
        """Convert a dataset row to a SWEBenchTask model."""
        try:
            return SWEBenchTask(
                instance_id=row["instance_id"],
                repo=row["repo"],
                base_commit=row["base_commit"],
                problem_statement=row["problem_statement"],
                hints_text=row.get("hints_text", ""),
                created_at=row.get("created_at", ""),
                patch=row["patch"],
                test_patch=row.get("test_patch", ""),
                version=row.get("version", ""),
                environment_setup_commit=row.get("environment_setup_commit", ""),
                fail_to_pass=row.get("FAIL_TO_PASS", row.get("fail_to_pass", "")),
                pass_to_pass=row.get("PASS_TO_PASS", row.get("pass_to_pass", "")),
            )
        except ValidationError as e:
            logger.exception("Invalid task data for %s", row.get("instance_id", "unknown"))
            raise

    def iter_tasks(
        self,
        task_ids: list[str] | None = None,
        repos: list[str] | None = None,
    ) -> Iterator[SWEBenchTask]:
        """Iterate over tasks with optional filtering.

        Args:
            task_ids: If provided, only yield tasks with these IDs.
            repos: If provided, only yield tasks from these repos.

        Yields:
            SWEBenchTask objects matching the filters.
        """
        task_id_set = set(task_ids) if task_ids else None
        repo_set = set(repos) if repos else None

        for row in self.dataset:
            row_dict: dict[str, Any] = dict(row)  # type: ignore[arg-type]
            instance_id = row_dict["instance_id"]
            repo = row_dict["repo"]

            if task_id_set and instance_id not in task_id_set:
                continue
            if repo_set and repo not in repo_set:
                continue

            try:
                yield self._row_to_task(row_dict)
            except ValidationError:
                logger.warning("Skipping invalid task: %s", instance_id)
                continue

    def list_repos(self) -> list[str]:
        """Get unique repository names in the dataset."""
        repos: set[str] = set()
        for row in self.dataset:
            repos.add(row["repo"])  # type: ignore[index]
        return sorted(repos)

    def list_task_ids(self, repo: str | None = None) -> list[str]:
        """Get all task IDs, optionally filtered by repo."""
        ids: list[str] = []
        for row in self.dataset:
            if repo and row["repo"] != repo:  # type: ignore[index]
                continue
            ids.append(row["instance_id"])  # type: ignore[index]
        return ids

    def __len__(self) -> int:
        """Return total number of tasks."""
        return len(self.dataset)

    def stats(self) -> dict[str, Any]:
        """Return dataset statistics."""
        repos = self.list_repos()
        return {
            "dataset": self._dataset_name,
            "split": self._split,
            "total_tasks": len(self),
            "unique_repos": len(repos),
            "repos": repos,
            "tasks_per_repo": {repo: len(self.list_task_ids(repo)) for repo in repos},
        }
