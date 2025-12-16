"""SDD Framework Protocol definition.

Defines the interface that all SDD (Spec-Driven Development) frameworks
must implement to be evaluated by SDD-Bench. Uses Python's typing.Protocol
for structural subtyping - frameworks don't need to inherit, just implement
the required methods.

Example:
    >>> from claude_spec_benchmark.frameworks import SDDFrameworkProtocol
    >>> class MyFramework:
    ...     def start_elicitation(self, initial_context: str) -> str:
    ...         return "What error are you seeing?"
    ...     # ... implement all other methods
    >>> def is_framework(f: SDDFrameworkProtocol) -> bool:
    ...     return True  # Type checker verifies protocol compliance
"""

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class SDDFrameworkProtocol(Protocol):
    """Interface for spec-driven development frameworks.

    This protocol defines the contract that frameworks must implement to
    participate in SDD-Bench evaluation. The pipeline executes these methods
    in order to simulate the full SDD lifecycle:

    1. **Elicitation** (start_elicitation, process_response, get_elicited_spec):
       Dialogue with a simulated stakeholder to uncover hidden requirements.

    2. **Specification** (parse_spec):
       Extract structured requirements from the elicited specification.

    3. **Test Generation** (generate_tests):
       Create tests that encode the requirements (TDD Red phase).

    4. **Implementation** (implement):
       Generate code that passes the tests (TDD Green phase).

    5. **Refinement** (refine):
       Iterate on test failures (TDD Refactor phase).

    Frameworks are evaluated by comparing their generated patches against
    SWE-bench gold patches using the standard evaluation harness.

    Example:
        >>> class MyFramework:
        ...     def start_elicitation(self, initial_context: str) -> str:
        ...         return "Can you describe the expected behavior?"
        ...
        ...     def process_response(self, response: str) -> str | None:
        ...         return None  # Done after one question
        ...
        ...     def get_elicited_spec(self) -> str:
        ...         return self._context
        ...
        ...     def parse_spec(self, spec: str, repo_path: Path) -> list[str]:
        ...         return ["REQ-001: Fix the bug"]
        ...
        ...     def generate_tests(
        ...         self, spec: str, requirements: list[str], repo_path: Path
        ...     ) -> str:
        ...         return "def test_fix(): pass"
        ...
        ...     def implement(
        ...         self,
        ...         spec: str,
        ...         requirements: list[str],
        ...         tests: str,
        ...         repo_path: Path,
        ...     ) -> str:
        ...         return "diff --git a/fix.py b/fix.py\\n..."
        ...
        ...     def refine(
        ...         self,
        ...         implementation: str,
        ...         test_failures: list[str],
        ...         repo_path: Path,
        ...     ) -> str:
        ...         return implementation  # No refinement needed

    Note:
        The @runtime_checkable decorator allows isinstance() checks:
        >>> isinstance(my_framework, SDDFrameworkProtocol)
        True
    """

    # =========================================================================
    # Elicitation Phase
    # =========================================================================

    def start_elicitation(self, initial_context: str) -> str:
        """Begin the elicitation dialogue.

        Called with the degraded specification as initial context. The framework
        should return its first question to ask the simulated stakeholder (oracle).

        Args:
            initial_context: The degraded specification text. May be vague,
                minimal, or even ambiguous depending on the degradation level.

        Returns:
            The first question to ask the stakeholder. Return an empty string
            to skip elicitation entirely (useful for passthrough frameworks).

        Example:
            >>> framework.start_elicitation("Login doesn't work")
            "What specific error are you encountering when trying to log in?"
        """
        ...

    def process_response(self, stakeholder_response: str) -> str | None:
        """Process a stakeholder response and optionally ask another question.

        Called after the oracle provides an answer to the previous question.
        The framework should analyze the response and either ask a follow-up
        question or signal that elicitation is complete.

        Args:
            stakeholder_response: The oracle's answer to the previous question.
                Contains information about hidden requirements based on question
                relevance.

        Returns:
            The next question to ask, or None to end elicitation.
            Returning None signals that the framework has gathered enough
            information and is ready to produce the elicited specification.

        Example:
            >>> framework.process_response("The form returns 500 on submit")
            "What endpoint is the form submitting to?"
            >>> framework.process_response("POST /api/auth/login")
            None  # Done gathering info
        """
        ...

    def get_elicited_spec(self) -> str:
        """Return the specification derived from elicitation.

        Called after elicitation is complete (process_response returned None).
        Should return a comprehensive specification incorporating all information
        gathered during the elicitation dialogue.

        Returns:
            The elicited specification text. This will be used for the parse,
            test generation, and implementation phases.

        Example:
            >>> spec = framework.get_elicited_spec()
            >>> print(spec)
            "## Problem\\nLogin form returns 500 on submit...\\n## Requirements..."
        """
        ...

    # =========================================================================
    # Specification Phase
    # =========================================================================

    def parse_spec(self, spec: str, repo_path: Path) -> list[str]:
        """Extract structured requirements from the specification.

        Analyzes the specification and produces a list of atomic, testable
        requirements. These requirements guide test generation and implementation.

        Args:
            spec: The elicited specification text.
            repo_path: Path to the checked-out repository. Can be used to
                understand code structure and naming conventions.

        Returns:
            List of requirement strings. Each should be a clear, testable
            statement. Format is flexible but should be consistent.

        Example:
            >>> requirements = framework.parse_spec(spec, Path("/tmp/django"))
            >>> requirements
            ["REQ-001: Login endpoint should return 200 on valid credentials",
             "REQ-002: Login endpoint should return 401 on invalid credentials",
             "REQ-003: Login endpoint should rate-limit after 5 failed attempts"]
        """
        ...

    # =========================================================================
    # Test Generation Phase (TDD Red)
    # =========================================================================

    def generate_tests(
        self,
        spec: str,
        requirements: list[str],
        repo_path: Path,
    ) -> str:
        """Generate tests that encode the requirements.

        Creates test code that, when run, will validate whether the requirements
        are satisfied. This is the "Red" phase of TDD - tests should initially
        fail against the current codebase.

        Args:
            spec: The elicited specification text.
            requirements: List of requirements extracted by parse_spec.
            repo_path: Path to the checked-out repository.

        Returns:
            Test code as a string. Should be in a format appropriate for the
            repository's test framework (pytest for Python, jest for JS, etc.).
            The returned string may be a complete test file or a patch to add
            tests to existing test files.

        Example:
            >>> tests = framework.generate_tests(spec, requirements, repo_path)
            >>> print(tests)
            "import pytest\\nfrom app.auth import login\\n\\n
            def test_login_valid_credentials():\\n    result = login(...)\\n..."
        """
        ...

    # =========================================================================
    # Implementation Phase (TDD Green)
    # =========================================================================

    def implement(
        self,
        spec: str,
        requirements: list[str],
        tests: str,
        repo_path: Path,
    ) -> str:
        """Generate implementation that passes the tests.

        Creates a patch that, when applied, should make the generated tests pass.
        This is the "Green" phase of TDD - the implementation should satisfy
        the requirements as encoded in the tests.

        Args:
            spec: The elicited specification text.
            requirements: List of requirements extracted by parse_spec.
            tests: Test code generated by generate_tests.
            repo_path: Path to the checked-out repository.

        Returns:
            A unified diff patch that can be applied with `git apply`.
            The patch should make minimal changes necessary to pass the tests.

        Example:
            >>> patch = framework.implement(spec, requirements, tests, repo_path)
            >>> print(patch)
            "diff --git a/app/auth.py b/app/auth.py\\n--- a/app/auth.py\\n..."
        """
        ...

    # =========================================================================
    # Refinement Phase (TDD Refactor)
    # =========================================================================

    def refine(
        self,
        implementation: str,
        test_failures: list[str],
        repo_path: Path,
    ) -> str:
        """Refine implementation based on test failures.

        Called when the generated tests don't pass with the initial implementation.
        The framework should analyze the failures and produce an improved patch.

        This is the "Refactor" phase of TDD, though here it's more about fixing
        implementation bugs than improving code quality.

        Args:
            implementation: The current patch (from implement or previous refine).
            test_failures: List of test failure messages/names.
            repo_path: Path to the checked-out repository.

        Returns:
            A refined unified diff patch. Should address the test failures
            while maintaining any passing tests.

        Note:
            This method may be called multiple times in a loop until tests pass
            or a maximum iteration limit is reached.

        Example:
            >>> failures = ["test_login_valid: AssertionError: Expected 200"]
            >>> new_patch = framework.refine(patch, failures, repo_path)
        """
        ...
