"""Passthrough framework - baseline implementation.

A minimal SDDFrameworkProtocol implementation that passes the specification
unchanged without any elicitation, test generation, or refinement.
Useful as a baseline for measuring the impact of SDD phases.

Example:
    >>> framework = PassthroughFramework()
    >>> framework.start_elicitation("Login is broken")
    ''  # No questions - skips elicitation
    >>> framework.get_elicited_spec()
    'Login is broken'  # Returns original spec unchanged
"""

from pathlib import Path

from claude_spec_benchmark.frameworks.protocol import SDDFrameworkProtocol
from claude_spec_benchmark.runner import ClaudeCodeRunner, RunnerError


class PassthroughFramework:
    """Baseline framework that passes spec directly without SDD phases.

    This implementation:
    - Skips elicitation (returns empty string for first question)
    - Returns the original spec unchanged
    - Extracts no requirements from parsing
    - Generates no tests
    - Uses ClaudeCodeRunner directly for implementation (if provided)
    - Performs no refinement

    This serves as a control group for SDD-Bench evaluation - comparing
    against PassthroughFramework shows the value of actual SDD practices.

    Example:
        >>> from claude_spec_benchmark.frameworks import PassthroughFramework
        >>> from claude_spec_benchmark.frameworks import SDDFrameworkProtocol
        >>> framework = PassthroughFramework()
        >>> isinstance(framework, SDDFrameworkProtocol)
        True
        >>> framework.start_elicitation("Fix the auth bug")
        ''
        >>> framework.process_response("anything") is None
        True
        >>> framework.get_elicited_spec()
        'Fix the auth bug'

    Attributes:
        runner: Optional ClaudeCodeRunner for implementation phase.
            If None, implement() returns an empty patch.
    """

    def __init__(
        self,
        runner: ClaudeCodeRunner | None = None,
    ) -> None:
        """Initialize the passthrough framework.

        Args:
            runner: Optional ClaudeCodeRunner for implementation.
                If not provided, implement() returns an empty patch.
        """
        self._runner = runner
        self._initial_context: str = ""

    # =========================================================================
    # Elicitation Phase - Skipped
    # =========================================================================

    def start_elicitation(self, initial_context: str) -> str:
        """Skip elicitation, store context for later.

        Args:
            initial_context: The degraded specification text.

        Returns:
            Empty string to signal no elicitation is needed.
        """
        self._initial_context = initial_context
        return ""  # No questions - skip elicitation

    def process_response(self, _stakeholder_response: str) -> str | None:
        """End elicitation immediately.

        Args:
            _stakeholder_response: Ignored - elicitation is skipped.

        Returns:
            None to signal elicitation is complete.
        """
        return None  # Done immediately

    def get_elicited_spec(self) -> str:
        """Return the original spec unchanged.

        Returns:
            The initial context passed to start_elicitation.
        """
        return self._initial_context

    # =========================================================================
    # Specification Phase - Returns Empty
    # =========================================================================

    def parse_spec(self, _spec: str, _repo_path: Path) -> list[str]:
        """Extract no requirements from the spec.

        A passthrough framework doesn't analyze the spec structure.
        Returns an empty list to indicate no structured requirements.

        Args:
            _spec: The specification text (ignored).
            _repo_path: Path to repository (ignored).

        Returns:
            Empty list - no requirements extracted.
        """
        return []

    # =========================================================================
    # Test Generation Phase - Skipped
    # =========================================================================

    def generate_tests(
        self,
        _spec: str,
        _requirements: list[str],
        _repo_path: Path,
    ) -> str:
        """Generate no tests.

        Passthrough framework skips the TDD Red phase entirely.

        Args:
            _spec: The specification text (ignored).
            _requirements: List of requirements (ignored).
            _repo_path: Path to repository (ignored).

        Returns:
            Empty string - no tests generated.
        """
        return ""

    # =========================================================================
    # Implementation Phase - Optional Runner
    # =========================================================================

    def implement(
        self,
        spec: str,
        _requirements: list[str],
        _tests: str,
        repo_path: Path,
    ) -> str:
        """Generate implementation using ClaudeCodeRunner if available.

        If a runner was provided at initialization, delegates to it.
        Otherwise returns an empty patch (useful for testing).

        Args:
            spec: The specification text (used as prompt if runner exists).
            _requirements: List of requirements (ignored).
            _tests: Test code (ignored - passthrough doesn't use TDD).
            repo_path: Path to repository.

        Returns:
            Generated patch from runner, or empty string if no runner.

        Note:
            This is a synchronous wrapper. The runner.run_task is async,
            so this method creates a minimal SWEBenchTask and runs it
            synchronously using asyncio.run().
        """
        if self._runner is None:
            return ""

        # Import here to avoid circular imports
        import asyncio

        from claude_spec_benchmark.models import SWEBenchTask

        # Create a minimal task from the spec
        # The instance_id and repo are placeholders since we're using
        # the repo_path directly
        task = SWEBenchTask(
            instance_id="passthrough",
            repo=str(repo_path.name),
            base_commit="HEAD",
            problem_statement=spec,
            patch="",  # Gold patch unknown in passthrough mode
        )

        try:
            result = asyncio.run(self._runner.run_task(task, repo_path))
            return result.generated_patch or ""
        except RunnerError:
            return ""

    # =========================================================================
    # Refinement Phase - Skipped
    # =========================================================================

    def refine(
        self,
        implementation: str,
        _test_failures: list[str],
        _repo_path: Path,
    ) -> str:
        """Skip refinement, return implementation unchanged.

        Passthrough framework doesn't iterate on failures.

        Args:
            implementation: The current patch.
            _test_failures: List of failures (ignored).
            _repo_path: Path to repository (ignored).

        Returns:
            The original implementation unchanged.
        """
        return implementation


# Type assertion to verify protocol compliance at module load time
def _verify_protocol_compliance() -> None:
    """Verify PassthroughFramework implements SDDFrameworkProtocol."""
    framework: SDDFrameworkProtocol = PassthroughFramework()
    if not isinstance(framework, SDDFrameworkProtocol):
        msg = "PassthroughFramework does not implement SDDFrameworkProtocol"
        raise TypeError(msg)


_verify_protocol_compliance()
