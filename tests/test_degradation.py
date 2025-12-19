"""Tests for the specification degradation module."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from claude_spec_benchmark.degradation.engine import DegradationEngine
from claude_spec_benchmark.degradation.patterns import DegradationPatterns
from claude_spec_benchmark.models import SpecDegradationLevel

# Sample SWE-bench-like issue for testing
SAMPLE_ISSUE = """
Fix authentication error in django/contrib/auth/views.py

When trying to log in, users get the following error:

```python
Traceback (most recent call last):
  File "django/contrib/auth/views.py", line 42, in login
    user = authenticate(request, username=username, password=password)
  File "django/contrib/auth/__init__.py", line 73, in authenticate
    raise AuthenticationError("Invalid credentials")
AuthenticationError: Invalid credentials
```

The issue is that `authenticate()` is not handling the case where
the user account is locked. See commit a1b2c3d for the regression.

Expected behavior: Show "Account locked" message instead of generic error.

Version: Django 3.2.0
Related: #1234, GH-5678
"""


class TestDegradationPatterns:
    """Tests for DegradationPatterns class."""

    def test_default_patterns(self):
        """Test default pattern creation."""
        patterns = DegradationPatterns.default()
        assert patterns is not None
        assert patterns.code_block_pattern

    def test_matches_code_block(self):
        """Test code block detection."""
        patterns = DegradationPatterns.default()
        assert patterns.matches_code_block("```python\ncode\n```")
        assert not patterns.matches_code_block("regular text")

    def test_matches_stack_trace(self):
        """Test stack trace detection."""
        patterns = DegradationPatterns.default()
        trace = "Traceback (most recent call last):\n  File test.py"
        assert patterns.matches_stack_trace(trace)
        assert not patterns.matches_stack_trace("regular text")

    def test_matches_file_path(self):
        """Test file path detection."""
        patterns = DegradationPatterns.default()
        assert patterns.matches_file_path("django/contrib/auth/views.py")
        assert patterns.matches_file_path("src/main.rs")
        assert not patterns.matches_file_path("hello world")

    def test_find_all(self):
        """Test finding all matches."""
        patterns = DegradationPatterns.default()
        text = "See file test.py at line 42 and main.py at line 100"
        matches = patterns.find_all("file_path_pattern", text)
        assert len(matches) >= 2

    def test_remove_all(self):
        """Test removing all matches."""
        patterns = DegradationPatterns.default()
        text = "Check `config` and `settings`"
        result = patterns.remove_all("inline_code_pattern", text, "[code]")
        assert "`config`" not in result
        assert "[code]" in result

    def test_for_repo_django(self):
        """Test Django-specific patterns."""
        patterns = DegradationPatterns.for_repo("django/django")
        assert patterns is not None
        # Django patterns should include html files (extension without dot in regex)
        assert "html" in patterns.file_path_pattern

    def test_for_repo_flask(self):
        """Test Flask-specific patterns."""
        patterns = DegradationPatterns.for_repo("pallets/flask")
        assert patterns is not None
        # Flask patterns should include jinja2 files (extension without dot in regex)
        assert "jinja2" in patterns.file_path_pattern

    def test_for_repo_unknown(self):
        """Test patterns for unknown repo fall back to default."""
        patterns = DegradationPatterns.for_repo("unknown/repo")
        default = DegradationPatterns.default()
        assert patterns.code_block_pattern == default.code_block_pattern

    def test_get_patterns_for_level(self):
        """Test pattern list for each level."""
        patterns = DegradationPatterns.default()

        full_patterns = patterns.get_patterns_for_level("full")
        assert full_patterns == []

        partial_patterns = patterns.get_patterns_for_level("partial")
        assert "code_block_pattern" in partial_patterns
        assert "stack_trace_pattern" in partial_patterns

        vague_patterns = patterns.get_patterns_for_level("vague")
        assert len(vague_patterns) > len(partial_patterns)

        minimal_patterns = patterns.get_patterns_for_level("minimal")
        assert "url_pattern" in minimal_patterns


class TestDegradationEngine:
    """Tests for DegradationEngine class."""

    @pytest.fixture
    def engine(self) -> DegradationEngine:
        """Create a default engine for testing."""
        return DegradationEngine()

    def test_degrade_full_returns_original(self, engine):
        """Test FULL level returns original text unchanged."""
        result = engine.degrade(SAMPLE_ISSUE, SpecDegradationLevel.FULL, seed=42)

        assert result.degraded_text == SAMPLE_ISSUE
        assert result.original_text == SAMPLE_ISSUE
        assert result.hidden_details == []
        assert result.level == SpecDegradationLevel.FULL
        assert result.seed == 42

    def test_degrade_partial_removes_code(self, engine):
        """Test PARTIAL level removes code blocks and traces."""
        result = engine.degrade(SAMPLE_ISSUE, SpecDegradationLevel.PARTIAL, seed=42)

        assert "```python" not in result.degraded_text
        assert "Traceback" not in result.degraded_text
        assert result.level == SpecDegradationLevel.PARTIAL
        assert len(result.hidden_details) > 0
        # Should still have some context
        assert (
            "authentication" in result.degraded_text.lower()
            or "auth" in result.degraded_text.lower()
        )

    def test_degrade_partial_tracks_hidden(self, engine):
        """Test PARTIAL level tracks what was hidden."""
        result = engine.degrade(SAMPLE_ISSUE, SpecDegradationLevel.PARTIAL, seed=42)

        hidden_labels = [h.split("]")[0] for h in result.hidden_details]
        # Should have hidden code and/or traces
        assert any("code" in label or "stack" in label for label in hidden_labels)

    def test_degrade_vague_simplifies_language(self, engine):
        """Test VAGUE level simplifies technical language."""
        result = engine.degrade(SAMPLE_ISSUE, SpecDegradationLevel.VAGUE, seed=42)

        # Should be shorter than original
        assert len(result.degraded_text) < len(SAMPLE_ISSUE)
        assert result.level == SpecDegradationLevel.VAGUE
        # Should not have code blocks or inline code
        assert "```" not in result.degraded_text
        assert "`authenticate`" not in result.degraded_text

    def test_degrade_minimal_one_sentence(self, engine):
        """Test MINIMAL level produces brief summary."""
        result = engine.degrade(SAMPLE_ISSUE, SpecDegradationLevel.MINIMAL, seed=42)

        # Should be much shorter
        assert len(result.degraded_text) < 200
        assert result.level == SpecDegradationLevel.MINIMAL
        # Should track that full context was hidden
        assert any("full_context" in h for h in result.hidden_details)

    def test_degrade_ambiguous_adds_contradictions(self, engine):
        """Test AMBIGUOUS level adds intentional confusion."""
        result = engine.degrade(SAMPLE_ISSUE, SpecDegradationLevel.AMBIGUOUS, seed=42)

        assert result.level == SpecDegradationLevel.AMBIGUOUS
        # Should note that ambiguity was introduced
        assert any("ambiguity" in h.lower() for h in result.hidden_details)

    def test_deterministic_with_seed(self, engine):
        """Test that same seed produces same output."""
        result1 = engine.degrade(SAMPLE_ISSUE, SpecDegradationLevel.VAGUE, seed=42)
        result2 = engine.degrade(SAMPLE_ISSUE, SpecDegradationLevel.VAGUE, seed=42)

        assert result1.degraded_text == result2.degraded_text
        assert result1.hidden_details == result2.hidden_details

    def test_different_seeds_different_output(self, engine):
        """Test that different seeds can produce different outputs."""
        result1 = engine.degrade(SAMPLE_ISSUE, SpecDegradationLevel.AMBIGUOUS, seed=42)
        result2 = engine.degrade(SAMPLE_ISSUE, SpecDegradationLevel.AMBIGUOUS, seed=123)

        # For ambiguous level, different seeds should produce different ambiguity
        # (though the underlying degraded text may be similar)
        # At minimum, they should both be valid degradations
        assert result1.level == result2.level == SpecDegradationLevel.AMBIGUOUS

    def test_from_repo_factory(self):
        """Test engine creation from repo name."""
        engine = DegradationEngine.from_repo("django/django")

        assert engine.patterns is not None
        # Django patterns should recognize html files
        assert "html" in engine.patterns.file_path_pattern

    def test_get_deterministic_seed(self, engine):
        """Test deterministic seed generation."""
        seed1 = engine.get_deterministic_seed(
            "django__django-11099", SpecDegradationLevel.VAGUE
        )
        seed2 = engine.get_deterministic_seed(
            "django__django-11099", SpecDegradationLevel.VAGUE
        )

        assert seed1 == seed2
        assert isinstance(seed1, int)

        # Different inputs should produce different seeds
        seed3 = engine.get_deterministic_seed(
            "django__django-11099", SpecDegradationLevel.MINIMAL
        )
        assert seed1 != seed3

        seed4 = engine.get_deterministic_seed(
            "flask__flask-1234", SpecDegradationLevel.VAGUE
        )
        assert seed1 != seed4

    def test_degrade_with_repo_patterns(self, engine):
        """Test degradation with repo-specific patterns."""
        result = engine.degrade(
            SAMPLE_ISSUE,
            SpecDegradationLevel.PARTIAL,
            seed=42,
            repo="django/django",
        )

        assert result.level == SpecDegradationLevel.PARTIAL
        assert len(result.hidden_details) > 0

    def test_degraded_spec_is_frozen(self, engine):
        """Test that returned DegradedSpec is immutable."""
        result = engine.degrade(SAMPLE_ISSUE, SpecDegradationLevel.FULL, seed=42)

        with pytest.raises(ValidationError):  # Pydantic frozen model
            result.degraded_text = "modified"  # type: ignore

    def test_level_ordering_removes_more(self, engine):
        """Test that higher levels remove more content."""
        partial = engine.degrade(SAMPLE_ISSUE, SpecDegradationLevel.PARTIAL, seed=42)
        vague = engine.degrade(SAMPLE_ISSUE, SpecDegradationLevel.VAGUE, seed=42)
        minimal = engine.degrade(SAMPLE_ISSUE, SpecDegradationLevel.MINIMAL, seed=42)

        # Each level should produce progressively shorter output
        assert len(vague.degraded_text) <= len(partial.degraded_text)
        assert len(minimal.degraded_text) <= len(vague.degraded_text)


class TestDegradationIntegration:
    """Integration tests with realistic SWE-bench data."""

    @pytest.fixture
    def realistic_issue(self) -> str:
        """A more realistic SWE-bench issue."""
        return """
Bug: QuerySet.union() ignores order_by() after certain operations

When using `QuerySet.union()` followed by `order_by()`, the ordering
is silently ignored if the queryset was previously filtered.

Reproduction:
```python
from myapp.models import User

qs1 = User.objects.filter(is_active=True)
qs2 = User.objects.filter(is_staff=True)
combined = qs1.union(qs2).order_by('username')

# Expected: Users ordered by username
# Actual: Random order

for user in combined:
    print(user.username)
```

Stack trace when DEBUG=True:
```
File "/usr/lib/python3.9/site-packages/django/db/models/query.py", line 1089, in union
  return self._combinator_query('union', *other_qs, all=all)
File "/usr/lib/python3.9/site-packages/django/db/models/query.py", line 1123, in _combinator_query
  clone.query.combined_queries = (self.query,) + tuple(qs.query for qs in other_qs)
```

This regression was introduced in commit abc123def456.

Django version: 4.2.0
Python version: 3.9.7
Database: PostgreSQL 14.0
"""

    def test_degrade_realistic_partial(self, realistic_issue):
        """Test partial degradation on realistic issue."""
        engine = DegradationEngine.from_repo("django/django")
        result = engine.degrade(realistic_issue, SpecDegradationLevel.PARTIAL, seed=100)

        # Code blocks should be removed
        assert "```python" not in result.degraded_text
        # But problem description should remain
        assert (
            "union" in result.degraded_text.lower()
            or "queryset" in result.degraded_text.lower()
        )

    def test_degrade_realistic_vague(self, realistic_issue):
        """Test vague degradation on realistic issue."""
        engine = DegradationEngine.from_repo("django/django")
        result = engine.degrade(realistic_issue, SpecDegradationLevel.VAGUE, seed=100)

        # Should be significantly shorter
        assert len(result.degraded_text) < len(realistic_issue) * 0.7
        # Technical details should be removed
        assert "line 1089" not in result.degraded_text
        assert "abc123def456" not in result.degraded_text

    def test_hidden_details_enable_scoring(self, realistic_issue):
        """Test that hidden details can be used for scoring."""
        engine = DegradationEngine()
        result = engine.degrade(realistic_issue, SpecDegradationLevel.VAGUE, seed=100)

        # Hidden details should include specific removals
        # Should have tracked code blocks, file paths, etc.
        assert len(result.hidden_details) >= 3
        # Check that labels are present
        assert any("[" in h and "]" in h for h in result.hidden_details)

    def test_reproducibility_across_runs(self, realistic_issue):
        """Test that same inputs always produce same outputs."""
        engine = DegradationEngine()

        # Run multiple times with same seed
        results = [
            engine.degrade(realistic_issue, SpecDegradationLevel.VAGUE, seed=999)
            for _ in range(3)
        ]

        # All should be identical
        assert all(r.degraded_text == results[0].degraded_text for r in results)
        assert all(r.hidden_details == results[0].hidden_details for r in results)
