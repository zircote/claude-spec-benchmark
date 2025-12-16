"""Tests for SDD framework implementations.

Tests protocol compliance and behavior for:
- PassthroughFramework
- ClaudeCodeFramework
- SDDFrameworkProtocol type checking
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from claude_spec_benchmark.frameworks import (
    ClaudeCodeFramework,
    PassthroughFramework,
    SDDFrameworkProtocol,
)


class TestSDDFrameworkProtocol:
    """Tests for the SDDFrameworkProtocol type checking."""

    def test_passthrough_is_protocol_compliant(self):
        """PassthroughFramework should implement SDDFrameworkProtocol."""
        framework = PassthroughFramework()
        assert isinstance(framework, SDDFrameworkProtocol)

    def test_claude_code_has_protocol_methods(self):
        """ClaudeCodeFramework should have all protocol methods."""
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
            assert hasattr(ClaudeCodeFramework, method), f"Missing method: {method}"

    def test_protocol_is_runtime_checkable(self):
        """SDDFrameworkProtocol should support isinstance checks."""
        # Create a class that doesn't implement the protocol
        class NotAFramework:
            pass

        assert not isinstance(NotAFramework(), SDDFrameworkProtocol)


class TestPassthroughFramework:
    """Tests for the PassthroughFramework baseline implementation."""

    @pytest.fixture
    def framework(self):
        """Create a PassthroughFramework instance."""
        return PassthroughFramework()

    def test_start_elicitation_returns_empty(self, framework):
        """start_elicitation should return empty string (skip elicitation)."""
        result = framework.start_elicitation("Test problem description")
        assert result == ""

    def test_process_response_returns_none(self, framework):
        """process_response should return None (end immediately)."""
        framework.start_elicitation("Test")
        result = framework.process_response("Any response")
        assert result is None

    def test_get_elicited_spec_returns_original(self, framework):
        """get_elicited_spec should return original context unchanged."""
        original = "This is the original problem statement"
        framework.start_elicitation(original)
        result = framework.get_elicited_spec()
        assert result == original

    def test_parse_spec_returns_empty_list(self, framework):
        """parse_spec should return empty list (no requirement extraction)."""
        result = framework.parse_spec("Some spec", Path("/tmp"))
        assert result == []

    def test_generate_tests_returns_empty(self, framework):
        """generate_tests should return empty string (no tests)."""
        result = framework.generate_tests("Spec", ["REQ-1"], Path("/tmp"))
        assert result == ""

    def test_implement_without_runner_returns_empty(self, framework):
        """implement without runner should return empty patch."""
        result = framework.implement("Spec", ["REQ-1"], "tests", Path("/tmp"))
        assert result == ""

    def test_refine_returns_original_implementation(self, framework):
        """refine should return the original implementation unchanged."""
        original_impl = "diff --git a/file.py b/file.py\n..."
        result = framework.refine(original_impl, ["test_failed"], Path("/tmp"))
        assert result == original_impl

    def test_full_elicitation_flow(self, framework):
        """Test complete elicitation workflow."""
        # Start
        question = framework.start_elicitation("Bug: login fails")
        assert question == ""

        # Try process (should end immediately)
        next_q = framework.process_response("Some answer")
        assert next_q is None

        # Get spec
        spec = framework.get_elicited_spec()
        assert spec == "Bug: login fails"


class TestPassthroughFrameworkWithRunner:
    """Tests for PassthroughFramework with a ClaudeCodeRunner."""

    @pytest.fixture
    def mock_runner(self):
        """Create a mock ClaudeCodeRunner."""
        runner = MagicMock()
        return runner

    def test_passthrough_stores_runner(self, mock_runner):
        """Framework should store the provided runner."""
        framework = PassthroughFramework(runner=mock_runner)
        assert framework._runner is mock_runner


class TestClaudeCodeFrameworkInit:
    """Tests for ClaudeCodeFramework initialization."""

    @pytest.fixture
    def mock_runner(self):
        """Create a mock ClaudeCodeRunner."""
        runner = MagicMock()
        return runner

    def test_init_with_defaults(self, mock_runner):
        """ClaudeCodeFramework should initialize with defaults."""
        framework = ClaudeCodeFramework(mock_runner)
        assert framework._runner is mock_runner
        assert framework._model == "sonnet"
        assert framework._max_rounds == 10

    def test_init_with_custom_params(self, mock_runner):
        """ClaudeCodeFramework should accept custom parameters."""
        framework = ClaudeCodeFramework(
            runner=mock_runner,
            model="opus",
            max_elicitation_rounds=5,
        )
        assert framework._model == "opus"
        assert framework._max_rounds == 5


class TestClaudeCodeFrameworkElicitation:
    """Tests for ClaudeCodeFramework elicitation methods."""

    @pytest.fixture
    def mock_runner(self):
        """Create a mock ClaudeCodeRunner."""
        runner = MagicMock()
        return runner

    @pytest.fixture
    def framework(self, mock_runner):
        """Create a ClaudeCodeFramework with mock runner."""
        return ClaudeCodeFramework(mock_runner)

    def test_start_elicitation_stores_context(self, framework, mock_runner):
        """start_elicitation should store the initial context."""
        # Mock the run_task to return empty (simulating no CLI available)
        mock_run_result = MagicMock()
        mock_run_result.status.value = "failed"
        mock_run_result.stdout = ""

        with patch.object(
            framework, "_run_claude_sync", return_value=""
        ):
            result = framework.start_elicitation("Test context")

        assert framework._state.initial_context == "Test context"
        # Since _run_claude_sync returns empty, elicitation is done
        assert framework._state.done is True

    def test_get_elicited_spec_with_no_dialogue(self, framework):
        """get_elicited_spec should return original if no dialogue occurred."""
        framework._state.initial_context = "Original spec"
        framework._state.dialogue_history = []

        result = framework.get_elicited_spec()
        assert result == "Original spec"

    def test_get_elicited_spec_with_dialogue(self, framework):
        """get_elicited_spec should combine dialogue into spec."""
        framework._state.initial_context = "Original spec"
        framework._state.dialogue_history = [
            ("Q1: What happens?", "A1: Error occurs"),
            ("Q2: Which file?", "A2: auth.py"),
        ]

        result = framework.get_elicited_spec()

        assert "Original Specification" in result
        assert "Original spec" in result
        assert "Elicited Requirements" in result
        assert "Q1: What happens?" in result
        assert "A1: Error occurs" in result
        assert "Q2: Which file?" in result
        assert "A2: auth.py" in result


class TestClaudeCodeFrameworkParseSpec:
    """Tests for ClaudeCodeFramework parse_spec method."""

    @pytest.fixture
    def mock_runner(self):
        """Create a mock ClaudeCodeRunner."""
        return MagicMock()

    @pytest.fixture
    def framework(self, mock_runner):
        """Create a ClaudeCodeFramework with mock runner."""
        return ClaudeCodeFramework(mock_runner)

    def test_parse_numbered_list_valid(self, framework):
        """_parse_numbered_list should extract requirements from numbered list."""
        text = """1. The login should return 200
2. Invalid credentials should return 401
3. Rate limiting should apply after 5 failures"""

        result = framework._parse_numbered_list(text)

        assert len(result) == 3
        assert "login should return 200" in result[0]
        assert "Invalid credentials" in result[1]
        assert "Rate limiting" in result[2]

    def test_parse_numbered_list_empty(self, framework):
        """_parse_numbered_list should return empty for non-numbered text."""
        text = "This is just a plain paragraph without numbers."
        result = framework._parse_numbered_list(text)
        assert result == []

    def test_parse_numbered_list_mixed(self, framework):
        """_parse_numbered_list should extract only numbered items."""
        text = """Some intro text
1. First requirement
More text in between
2. Second requirement
Final text"""

        result = framework._parse_numbered_list(text)
        assert len(result) == 2


class TestProtocolMethodSignatures:
    """Tests to verify protocol method signatures."""

    @pytest.fixture
    def passthrough(self):
        return PassthroughFramework()

    def test_start_elicitation_signature(self, passthrough):
        """start_elicitation takes str, returns str."""
        result = passthrough.start_elicitation("context")
        assert isinstance(result, str)

    def test_process_response_signature(self, passthrough):
        """process_response takes str, returns str or None."""
        passthrough.start_elicitation("context")
        result = passthrough.process_response("response")
        assert result is None or isinstance(result, str)

    def test_get_elicited_spec_signature(self, passthrough):
        """get_elicited_spec takes no args, returns str."""
        passthrough.start_elicitation("context")
        result = passthrough.get_elicited_spec()
        assert isinstance(result, str)

    def test_parse_spec_signature(self, passthrough):
        """parse_spec takes str and Path, returns list[str]."""
        result = passthrough.parse_spec("spec", Path("/tmp"))
        assert isinstance(result, list)

    def test_generate_tests_signature(self, passthrough):
        """generate_tests takes str, list[str], Path, returns str."""
        result = passthrough.generate_tests("spec", ["req"], Path("/tmp"))
        assert isinstance(result, str)

    def test_implement_signature(self, passthrough):
        """implement takes str, list[str], str, Path, returns str."""
        result = passthrough.implement("spec", ["req"], "tests", Path("/tmp"))
        assert isinstance(result, str)

    def test_refine_signature(self, passthrough):
        """refine takes str, list[str], Path, returns str."""
        result = passthrough.refine("impl", ["failure"], Path("/tmp"))
        assert isinstance(result, str)
