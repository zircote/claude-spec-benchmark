"""Claude Code framework implementation.

A full SDDFrameworkProtocol implementation that uses Claude Code CLI
to perform all SDD phases: elicitation, specification parsing,
test generation, implementation, and refinement.

Example:
    >>> from claude_spec_benchmark.runner import ClaudeCodeRunner
    >>> runner = ClaudeCodeRunner()
    >>> framework = ClaudeCodeFramework(runner)
    >>> question = framework.start_elicitation("Something is broken")
    >>> print(question)
    "Can you describe what happens when the error occurs?"
"""

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path

from claude_spec_benchmark.models import SWEBenchTask, TaskStatus
from claude_spec_benchmark.runner import ClaudeCodeRunner, RunnerError

logger = logging.getLogger(__name__)


@dataclass
class ElicitationState:
    """Tracks the state of an elicitation dialogue.

    Attributes:
        initial_context: The degraded specification provided at start.
        dialogue_history: List of (question, response) tuples.
        round_count: Number of Q&A rounds completed.
        done: Whether elicitation is complete.
    """

    initial_context: str = ""
    dialogue_history: list[tuple[str, str]] = field(default_factory=list)
    round_count: int = 0
    done: bool = False


class ClaudeCodeFramework:
    """SDD framework using Claude Code CLI for all phases.

    This implementation uses the Claude Code CLI subprocess to handle
    each phase of spec-driven development:

    1. **Elicitation**: Claude generates questions to uncover requirements
    2. **Specification**: Claude parses spec into structured requirements
    3. **Test Generation**: Claude writes tests encoding requirements
    4. **Implementation**: Claude generates code to pass tests
    5. **Refinement**: Claude iterates on test failures

    Example:
        >>> from claude_spec_benchmark.runner import ClaudeCodeRunner
        >>> runner = ClaudeCodeRunner()
        >>> framework = ClaudeCodeFramework(runner, model="sonnet")
        >>> # Start elicitation with degraded spec
        >>> q = framework.start_elicitation("Login broken")
        >>> # Process oracle response
        >>> q = framework.process_response("It shows 500 error")
        >>> # Continue until None (done)
        >>> while q is not None:
        ...     q = framework.process_response(oracle_answer)
        >>> # Get combined spec
        >>> spec = framework.get_elicited_spec()

    Attributes:
        runner: The ClaudeCodeRunner for executing Claude CLI.
        model: Model to use (e.g., 'sonnet', 'opus').
        max_elicitation_rounds: Maximum Q&A rounds before stopping.
        elicitation_state: Current elicitation session state.
    """

    DEFAULT_MAX_ROUNDS = 10
    DEFAULT_MODEL = "sonnet"

    def __init__(
        self,
        runner: ClaudeCodeRunner,
        model: str = DEFAULT_MODEL,
        max_elicitation_rounds: int = DEFAULT_MAX_ROUNDS,
    ) -> None:
        """Initialize the Claude Code framework.

        Args:
            runner: ClaudeCodeRunner instance for CLI execution.
            model: Model name for generation (passed to runner if supported).
            max_elicitation_rounds: Maximum questions to ask during elicitation.
        """
        self._runner = runner
        self._model = model
        self._max_rounds = max_elicitation_rounds
        self._state = ElicitationState()

    # =========================================================================
    # Elicitation Phase
    # =========================================================================

    def start_elicitation(self, initial_context: str) -> str:
        """Begin elicitation by generating the first question.

        Args:
            initial_context: The degraded specification text.

        Returns:
            First question to ask the stakeholder, or empty string
            if elicitation cannot start.
        """
        self._state = ElicitationState(initial_context=initial_context)

        prompt = self._build_elicitation_prompt(is_first=True)
        question = self._run_claude_sync(prompt)

        if question:
            self._state.round_count = 1
            return question.strip()

        # If Claude returns empty, skip elicitation
        self._state.done = True
        return ""

    def process_response(self, stakeholder_response: str) -> str | None:
        """Process stakeholder response and generate follow-up question.

        Args:
            stakeholder_response: The oracle's answer.

        Returns:
            Next question, or None if elicitation is complete.
        """
        if self._state.done:
            return None

        # Record the dialogue
        last_question = self._get_last_question()
        self._state.dialogue_history.append((last_question, stakeholder_response))

        # Check if we've hit the max rounds
        if self._state.round_count >= self._max_rounds:
            self._state.done = True
            return None

        # Generate next question
        prompt = self._build_elicitation_prompt(is_first=False)
        response = self._run_claude_sync(prompt)

        # Check if Claude signals completion (empty or "DONE")
        if not response or response.strip().upper() in ("DONE", "NO MORE QUESTIONS"):
            self._state.done = True
            return None

        self._state.round_count += 1
        return response.strip()

    def get_elicited_spec(self) -> str:
        """Combine initial context and dialogue into elicited spec.

        Returns:
            Combined specification incorporating all gathered information.
        """
        if not self._state.dialogue_history:
            return self._state.initial_context

        # Build a combined spec from the dialogue
        parts = [
            "## Original Specification",
            self._state.initial_context,
            "",
            "## Elicited Requirements",
        ]

        for i, (question, answer) in enumerate(self._state.dialogue_history, 1):
            parts.extend(
                [
                    f"### Q{i}: {question}",
                    f"**A{i}:** {answer}",
                    "",
                ]
            )

        return "\n".join(parts)

    def _get_last_question(self) -> str:
        """Get the last question asked (for recording)."""
        # This is a bit of a hack - we need to track the question we asked
        # In practice, the prompt template includes generating a question
        # and we return it, so we need to cache it
        last_q: str = getattr(self, "_last_question", "")
        return last_q

    def _build_elicitation_prompt(self, is_first: bool) -> str:
        """Build prompt for elicitation question generation.

        Args:
            is_first: Whether this is the first question.

        Returns:
            Prompt string for Claude.
        """
        if is_first:
            return f"""You are helping gather requirements for a software bug fix.

The initial problem description is:
{self._state.initial_context}

This description is incomplete. Generate ONE clarifying question to ask the
stakeholder that would help understand the problem better.

Focus on:
- Specific error messages or behaviors
- Steps to reproduce
- Expected vs actual behavior
- Affected files or components

Output ONLY the question, nothing else. Do not include preamble like "I'd like to ask..."
"""
        # Follow-up question based on dialogue history
        dialogue = "\n".join(
            f"Q: {q}\nA: {a}" for q, a in self._state.dialogue_history
        )

        return f"""You are helping gather requirements for a software bug fix.

Original problem:
{self._state.initial_context}

Previous dialogue:
{dialogue}

Based on the information gathered so far, generate ONE more clarifying question,
or respond with exactly "DONE" if you have enough information to understand
the problem fully.

Focus on gaps in understanding:
- Missing technical details
- Unclear requirements
- Ambiguous behaviors

Output ONLY the question or "DONE", nothing else.
"""

    # =========================================================================
    # Specification Phase
    # =========================================================================

    def parse_spec(self, spec: str, repo_path: Path) -> list[str]:
        """Use Claude to extract structured requirements from spec.

        Args:
            spec: The elicited specification text.
            repo_path: Path to repository for context.

        Returns:
            List of requirement strings.
        """
        prompt = f"""Analyze this software bug specification and extract a list of
testable requirements. Each requirement should be specific and verifiable.

Specification:
{spec}

Repository: {repo_path.name}

Output requirements as a numbered list, one per line:
1. [Requirement 1]
2. [Requirement 2]
...

Output ONLY the numbered list, no other text.
"""

        response = self._run_claude_sync(prompt)
        return self._parse_numbered_list(response)

    # =========================================================================
    # Test Generation Phase
    # =========================================================================

    def generate_tests(
        self,
        spec: str,
        requirements: list[str],
        repo_path: Path,
    ) -> str:
        """Use Claude to generate tests for the requirements.

        Args:
            spec: The elicited specification text.
            requirements: List of requirements to test.
            repo_path: Path to repository.

        Returns:
            Test code as a string.
        """
        reqs_text = "\n".join(f"- {r}" for r in requirements)

        prompt = f"""Generate pytest tests for this bug fix specification.

Specification:
{spec}

Requirements to test:
{reqs_text}

Repository: {repo_path.name}

Generate comprehensive tests that:
1. Verify each requirement is met
2. Test edge cases
3. Follow pytest best practices

Output ONLY the Python test code, no other text.
"""

        return self._run_claude_sync(prompt)

    # =========================================================================
    # Implementation Phase
    # =========================================================================

    def implement(
        self,
        spec: str,
        requirements: list[str],
        tests: str,
        repo_path: Path,
    ) -> str:
        """Use Claude to generate implementation patch.

        Args:
            spec: The elicited specification text.
            requirements: List of requirements.
            tests: Generated test code.
            repo_path: Path to repository.

        Returns:
            Unified diff patch.
        """
        reqs_text = "\n".join(f"- {r}" for r in requirements) if requirements else "N/A"

        # Create a task for the runner
        task = SWEBenchTask(
            instance_id="sdd-impl",
            repo=str(repo_path.name),
            base_commit="HEAD",
            problem_statement=f"""## Specification
{spec}

## Requirements
{reqs_text}

## Generated Tests
```python
{tests}
```

Implement a fix that satisfies these requirements and passes the tests.
""",
            patch="",
        )

        try:
            result = asyncio.run(self._runner.run_task(task, repo_path))
            if result.status == TaskStatus.COMPLETED and result.generated_patch:
                return result.generated_patch
            return ""
        except RunnerError as e:
            logger.warning("Implementation failed: %s", e)
            return ""

    # =========================================================================
    # Refinement Phase
    # =========================================================================

    def refine(
        self,
        implementation: str,
        test_failures: list[str],
        repo_path: Path,
    ) -> str:
        """Use Claude to refine implementation based on test failures.

        Args:
            implementation: Current patch.
            test_failures: List of failure messages.
            repo_path: Path to repository.

        Returns:
            Refined patch.
        """
        if not test_failures:
            return implementation

        failures_text = "\n".join(f"- {f}" for f in test_failures)

        prompt = f"""The following patch was generated but some tests failed:

Current patch:
```diff
{implementation}
```

Test failures:
{failures_text}

Repository: {repo_path.name}

Analyze the failures and generate a refined patch that fixes the issues.
Output ONLY the unified diff patch, starting with "diff --git".
"""

        refined = self._run_claude_sync(prompt)

        # If Claude returned a patch, use it; otherwise return original
        if refined and refined.strip().startswith("diff --git"):
            return refined.strip()

        return implementation

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _run_claude_sync(self, prompt: str) -> str:
        """Run Claude CLI synchronously and return output.

        Args:
            prompt: The prompt to send to Claude.

        Returns:
            Claude's response text.
        """
        # Create a minimal task just to use the runner's infrastructure
        task = SWEBenchTask(
            instance_id="sdd-prompt",
            repo="prompt",
            base_commit="HEAD",
            problem_statement=prompt,
            patch="",
        )

        try:
            # Use a temporary directory since we don't need repo context
            import tempfile

            with tempfile.TemporaryDirectory() as tmpdir:
                result = asyncio.run(self._runner.run_task(task, Path(tmpdir)))
                return result.stdout if result.status == TaskStatus.COMPLETED else ""
        except RunnerError as e:
            logger.warning("Claude CLI error: %s", e)
            return ""

    def _parse_numbered_list(self, text: str) -> list[str]:
        """Parse a numbered list from Claude's response.

        Args:
            text: Response text containing numbered items.

        Returns:
            List of requirement strings.
        """
        import re

        requirements: list[str] = []
        pattern = re.compile(r"^\d+\.\s*(.+)$", re.MULTILINE)

        for match in pattern.finditer(text):
            req = match.group(1).strip()
            if req:
                requirements.append(req)

        return requirements


# Type assertion to verify protocol compliance at module load time
def _verify_protocol_compliance() -> None:
    """Verify ClaudeCodeFramework implements SDDFrameworkProtocol.

    Note: We can't fully instantiate without a runner, so we just check
    that the class has all required methods.
    """
    required_methods = [
        "start_elicitation",
        "process_response",
        "get_elicited_spec",
        "parse_spec",
        "generate_tests",
        "implement",
        "refine",
    ]
    for method in required_methods:
        if not hasattr(ClaudeCodeFramework, method):
            msg = f"ClaudeCodeFramework missing required method: {method}"
            raise TypeError(msg)


_verify_protocol_compliance()
