"""Tests for the elicitation simulation module."""

from __future__ import annotations

import pytest

from claude_spec_benchmark.elicitation.extraction import RequirementsExtractor
from claude_spec_benchmark.elicitation.oracle import (
    ElicitationOracle,
    OraclePersonality,
)
from claude_spec_benchmark.elicitation.scoring import (
    QuestionScorer,
    extract_keywords,
)
from claude_spec_benchmark.models import QuestionCategory, Requirement


class TestExtractKeywords:
    """Tests for keyword extraction function."""

    def test_basic_extraction(self):
        """Test extracting keywords from simple text."""
        keywords = extract_keywords("How does the login form validate emails?")
        assert "login" in keywords
        assert "form" in keywords
        assert "validate" in keywords
        assert "emails" in keywords

    def test_filters_stop_words(self):
        """Test that stop words are filtered out."""
        keywords = extract_keywords("the quick brown fox jumps over the lazy dog")
        # "the" and "over" should be filtered
        assert "the" not in keywords
        assert "over" not in keywords
        # Content words should remain
        assert "quick" in keywords
        assert "brown" in keywords
        assert "fox" in keywords

    def test_filters_short_words(self):
        """Test that very short words are filtered."""
        keywords = extract_keywords("I am in a fix")
        # Short words filtered
        assert "in" not in keywords
        assert "am" not in keywords
        assert "fix" in keywords  # 3 chars is ok

    def test_unique_keywords(self):
        """Test that keywords are unique."""
        keywords = extract_keywords("login login form form login")
        assert keywords.count("login") == 1
        assert keywords.count("form") == 1

    def test_preserves_order(self):
        """Test that first occurrence order is preserved."""
        keywords = extract_keywords("alpha beta gamma alpha")
        assert keywords == ["alpha", "beta", "gamma"]

    def test_empty_string(self):
        """Test handling of empty string."""
        keywords = extract_keywords("")
        assert keywords == []

    def test_only_stop_words(self):
        """Test string containing only stop words."""
        keywords = extract_keywords("the a an is are was")
        assert keywords == []


class TestQuestionScorer:
    """Tests for QuestionScorer class."""

    @pytest.fixture
    def sample_requirements(self) -> list[Requirement]:
        """Create sample requirements for testing."""
        return [
            Requirement(
                id="REQ-001",
                text="The login form should validate email format",
                keywords=["login", "email", "validation", "format"],
            ),
            Requirement(
                id="REQ-002",
                text="Failed login attempts should be logged",
                keywords=["login", "failed", "log", "attempts"],
            ),
            Requirement(
                id="REQ-003",
                text="Password reset should send confirmation email",
                keywords=["password", "reset", "email", "confirmation"],
            ),
        ]

    @pytest.fixture
    def scorer(self, sample_requirements) -> QuestionScorer:
        """Create a scorer with sample requirements."""
        return QuestionScorer(sample_requirements)

    def test_scorer_initialization(self, scorer, sample_requirements):
        """Test scorer initializes correctly."""
        assert scorer.requirements == sample_requirements
        assert len(scorer._req_keywords) == 3
        assert len(scorer._idf) > 0

    def test_score_relevant_question(self, scorer):
        """Test scoring a relevant question."""
        score = scorer.score("How does the login form validate emails?")
        assert score >= 0.5  # Should be fairly relevant (>= includes boundary)

    def test_score_irrelevant_question(self, scorer):
        """Test scoring an irrelevant question."""
        score = scorer.score("What color is the sky today?")
        assert score == 0.0  # No overlap

    def test_score_partial_match(self, scorer):
        """Test scoring with partial keyword match."""
        score = scorer.score("What happens when login fails?")
        assert 0.0 < score < 1.0  # Partial match

    def test_score_empty_question(self, scorer):
        """Test scoring an empty question."""
        score = scorer.score("")
        assert score == 0.0

    def test_score_only_stop_words(self, scorer):
        """Test scoring question with only stop words."""
        score = scorer.score("What is the?")
        assert score == 0.0

    def test_score_empty_requirements(self):
        """Test scorer with no requirements."""
        scorer = QuestionScorer([])
        score = scorer.score("Any question at all")
        assert score == 0.0

    def test_score_detailed(self, scorer):
        """Test detailed scoring breakdown."""
        score, details = scorer.score_detailed("How does email validation work?")
        assert score >= 0.0
        assert len(details) == 3  # One per requirement
        # Details should be sorted by score descending
        scores = [s for _, s in details]
        assert scores == sorted(scores, reverse=True)

    def test_score_capped_at_one(self, scorer):
        """Test that score never exceeds 1.0."""
        # Create a question that matches all keywords of a requirement
        score = scorer.score("login email validation format form")
        assert score <= 1.0

    def test_idf_weighting(self, sample_requirements):
        """Test that IDF weights rare terms higher."""
        scorer = QuestionScorer(sample_requirements)
        # "login" appears in multiple requirements (lower IDF)
        # "confirmation" appears in one (higher IDF)
        assert scorer._idf.get("login", 0) < scorer._idf.get("confirmation", 0)


class TestQuestionClassification:
    """Tests for Socratic question classification."""

    @pytest.fixture
    def scorer(self) -> QuestionScorer:
        """Create minimal scorer for classification tests."""
        return QuestionScorer([])

    def test_classify_clarification(self, scorer):
        """Test classification of clarification questions."""
        questions = [
            "What do you mean by 'valid email'?",
            "Can you explain the validation rules?",
            "What exactly happens during login?",
        ]
        for q in questions:
            assert scorer.classify_question(q) == QuestionCategory.CLARIFICATION

    def test_classify_assumption(self, scorer):
        """Test classification of assumption questions."""
        questions = [
            "What are you assuming about the user?",
            "Why do you think this is the cause?",
        ]
        for q in questions:
            assert scorer.classify_question(q) == QuestionCategory.ASSUMPTION

    def test_classify_evidence(self, scorer):
        """Test classification of evidence questions."""
        questions = [
            "What evidence supports this approach?",
            "How do we know the bug is in auth?",
            "Can you show me an example?",
        ]
        for q in questions:
            assert scorer.classify_question(q) == QuestionCategory.EVIDENCE

    def test_classify_viewpoint(self, scorer):
        """Test classification of viewpoint questions."""
        questions = [
            "Is there an alternative approach?",
            "What about another way to do this?",
        ]
        for q in questions:
            assert scorer.classify_question(q) == QuestionCategory.VIEWPOINT

    def test_classify_implication(self, scorer):
        """Test classification of implication questions."""
        questions = [
            "If we make this change then what happens?",
            "What would happen if we do this?",
        ]
        for q in questions:
            assert scorer.classify_question(q) == QuestionCategory.IMPLICATION

    def test_classify_meta(self, scorer):
        """Test classification of meta questions."""
        questions = [
            "What is the purpose of this question?",
            "Why asking this question?",
        ]
        for q in questions:
            assert scorer.classify_question(q) == QuestionCategory.META

    def test_classify_unknown_fallback(self, scorer):
        """Test fallback classification for ambiguous questions."""
        # Simple what/why/how should get reasonable defaults
        assert scorer.classify_question("What is X?") == QuestionCategory.CLARIFICATION
        assert scorer.classify_question("Why X?") == QuestionCategory.ASSUMPTION
        assert scorer.classify_question("How X?") == QuestionCategory.EVIDENCE

        # Completely ambiguous
        assert scorer.classify_question("X?") == QuestionCategory.UNKNOWN


class TestGetMatchingRequirements:
    """Tests for requirement matching."""

    @pytest.fixture
    def scorer(self) -> QuestionScorer:
        """Create scorer with varied requirements."""
        return QuestionScorer([
            Requirement(id="R1", text="Auth error handling", keywords=["auth", "error"]),
            Requirement(id="R2", text="Log storage", keywords=["log", "storage"]),
            Requirement(id="R3", text="Auth logging", keywords=["auth", "log"]),
        ])

    def test_get_matching_requirements(self, scorer):
        """Test getting requirements above threshold."""
        matches = scorer.get_matching_requirements("auth error?", threshold=0.3)
        assert "R1" in matches  # Direct match
        # R3 might match if "auth" alone is enough

    def test_no_matches_below_threshold(self, scorer):
        """Test that low-scoring requirements are excluded."""
        matches = scorer.get_matching_requirements("random question", threshold=0.3)
        assert len(matches) == 0

    def test_threshold_filtering(self, scorer):
        """Test that threshold correctly filters results."""
        # High threshold excludes partial matches
        high_matches = scorer.get_matching_requirements(
            "auth error logging", threshold=0.9
        )
        # Low threshold includes more
        low_matches = scorer.get_matching_requirements(
            "auth error logging", threshold=0.1
        )
        assert len(low_matches) >= len(high_matches)


class TestEdgeCases:
    """Edge case tests."""

    def test_special_characters_in_question(self):
        """Test handling of special characters."""
        reqs = [Requirement(id="R1", text="Handle auth", keywords=["auth", "handle"])]
        scorer = QuestionScorer(reqs)
        # Should handle special chars gracefully
        score = scorer.score("What about auth??? Really!!!")
        assert score > 0

    def test_unicode_in_question(self):
        """Test handling of unicode characters."""
        reqs = [Requirement(id="R1", text="Handle auth", keywords=["auth"])]
        scorer = QuestionScorer(reqs)
        score = scorer.score("What about auth? 🤔")
        assert score > 0

    def test_very_long_question(self):
        """Test handling of very long questions."""
        reqs = [Requirement(id="R1", text="Handle auth", keywords=["auth"])]
        scorer = QuestionScorer(reqs)
        long_q = "auth " * 1000 + "?"
        score = scorer.score(long_q)
        assert score > 0

    def test_requirement_with_empty_keywords(self):
        """Test requirement with no keywords."""
        reqs = [
            Requirement(id="R1", text="Handle authentication errors", keywords=[])
        ]
        scorer = QuestionScorer(reqs)
        # Should extract keywords from text
        score = scorer.score("What about authentication?")
        assert score > 0  # Should match via extracted keywords

    def test_case_insensitive_matching(self):
        """Test that matching is case-insensitive."""
        reqs = [Requirement(id="R1", text="Auth", keywords=["AUTH", "Error"])]
        scorer = QuestionScorer(reqs)
        score = scorer.score("what about auth error?")
        assert score > 0


# =============================================================================
# RequirementsExtractor Tests
# =============================================================================


class TestRequirementsExtractor:
    """Tests for RequirementsExtractor class."""

    @pytest.fixture
    def extractor(self) -> RequirementsExtractor:
        """Create a default extractor."""
        return RequirementsExtractor()

    def test_extract_should_statements(self, extractor):
        """Test extraction of should statements."""
        text = """
        The login form should validate email format.
        Users should be notified on successful login.
        """
        requirements = extractor.extract(text)
        assert len(requirements) >= 2
        assert any("should validate" in r.text.lower() for r in requirements)

    def test_extract_must_statements(self, extractor):
        """Test extraction of must statements."""
        text = "The system must encrypt all passwords before storing them."
        requirements = extractor.extract(text)
        assert len(requirements) >= 1
        assert any("must encrypt" in r.text.lower() for r in requirements)

    def test_extract_bug_descriptions(self, extractor):
        """Test extraction of bug-like requirements."""
        text = """
        Bug: Users cannot log in with special characters in password.
        The login fails when password contains @ or #.
        """
        requirements = extractor.extract(text)
        assert len(requirements) >= 1

    def test_extract_expected_behavior(self, extractor):
        """Test extraction of expected behavior statements."""
        text = """
        Expected: The form should accept all printable ASCII characters.
        Expected behavior: Email validation should use RFC 5322.
        """
        requirements = extractor.extract(text)
        assert len(requirements) >= 1

    def test_categorize_functional(self, extractor):
        """Test categorization of functional requirements."""
        text = "The system should send an email notification on order completion."
        requirements = extractor.extract(text)
        assert len(requirements) >= 1
        assert requirements[0].category == "functional"

    def test_categorize_non_functional(self, extractor):
        """Test categorization of non-functional requirements."""
        text = "The login page should have fast performance and quick response times."
        requirements = extractor.extract(text)
        assert len(requirements) >= 1
        assert requirements[0].category == "non-functional"

    def test_categorize_constraint(self, extractor):
        """Test categorization of constraint requirements."""
        text = "The password must be at least 8 characters and never stored in plaintext."
        requirements = extractor.extract(text)
        assert len(requirements) >= 1
        # Should be constraint due to "at least" and "never"
        assert any(r.category == "constraint" for r in requirements)

    def test_extracts_keywords(self, extractor):
        """Test that keywords are extracted for each requirement."""
        text = "The authentication system should validate user credentials."
        requirements = extractor.extract(text)
        assert len(requirements) >= 1
        req = requirements[0]
        assert len(req.keywords) > 0
        assert "authentication" in req.keywords or "validate" in req.keywords

    def test_assigns_unique_ids(self, extractor):
        """Test that each requirement gets a unique ID."""
        text = """
        The system should do A.
        The system should do B.
        The system should do C.
        """
        requirements = extractor.extract(text)
        ids = [r.id for r in requirements]
        assert len(ids) == len(set(ids))  # All unique

    def test_respects_max_requirements(self):
        """Test that max_requirements limit is respected."""
        extractor = RequirementsExtractor(max_requirements=3)
        text = "\n".join([f"The system should do task {i}." for i in range(10)])
        requirements = extractor.extract(text)
        assert len(requirements) <= 3

    def test_filters_short_sentences(self):
        """Test that very short sentences are filtered."""
        extractor = RequirementsExtractor(min_requirement_length=20)
        text = """
        Should work.
        The system should properly validate user input.
        """
        requirements = extractor.extract(text)
        # "Should work." is too short
        assert all(len(r.text) >= 20 for r in requirements)

    def test_deduplicates_requirements(self, extractor):
        """Test that duplicate requirements are removed."""
        text = """
        The login should validate email.
        The login should validate email.
        THE LOGIN SHOULD VALIDATE EMAIL.
        """
        requirements = extractor.extract(text)
        assert len(requirements) == 1

    def test_handles_code_blocks(self, extractor):
        """Test that code blocks don't break sentence splitting."""
        text = """
        The function should return True.

        ```python
        def example():
            return True
        ```

        The result should be logged.
        """
        requirements = extractor.extract(text)
        assert len(requirements) >= 2

    def test_empty_text(self, extractor):
        """Test handling of empty text."""
        requirements = extractor.extract("")
        assert requirements == []

    def test_no_requirements_in_text(self, extractor):
        """Test text with no recognizable requirements."""
        text = "This is just some random text without any requirements."
        requirements = extractor.extract(text)
        # Should return empty or minimal results
        assert len(requirements) <= 1

    def test_source_span_included(self, extractor):
        """Test that source spans are computed when possible."""
        text = "The system should validate input correctly."
        requirements = extractor.extract(text)
        if requirements:
            # Span should be present for simple matches
            req = requirements[0]
            if req.source_span:
                start, end = req.source_span
                assert text[start:end] == req.text or req.text in text

    def test_extract_with_context(self, extractor):
        """Test extraction with title context."""
        title = "Bug: Login validation broken"
        text = "Users cannot log in when password contains special chars."
        requirements = extractor.extract_with_context(text, title)
        # Should extract from combined text
        assert len(requirements) >= 1

    def test_get_summary(self, extractor):
        """Test summary generation."""
        text = """
        The system should send notifications.
        The page must load within 500ms for performance.
        Values must never exceed the maximum limit.
        """
        requirements = extractor.extract(text)
        summary = extractor.get_summary(requirements)
        assert "functional" in summary
        assert "non-functional" in summary
        assert "constraint" in summary
        # Total should match
        assert sum(summary.values()) == len(requirements)


class TestRequirementsExtractionPatterns:
    """Test specific extraction patterns."""

    @pytest.fixture
    def extractor(self) -> RequirementsExtractor:
        return RequirementsExtractor()

    def test_when_then_pattern(self, extractor):
        """Test when/then style requirements."""
        text = "When the user clicks submit, then the form should validate."
        requirements = extractor.extract(text)
        assert len(requirements) >= 1

    def test_given_pattern(self, extractor):
        """Test Given/As a/So that style (BDD)."""
        text = "Given a logged-in user, they should see the dashboard."
        requirements = extractor.extract(text)
        assert len(requirements) >= 1

    def test_cannot_pattern(self, extractor):
        """Test cannot/unable to patterns."""
        text = "Users cannot access admin pages without proper roles."
        requirements = extractor.extract(text)
        assert len(requirements) >= 1

    def test_fails_to_pattern(self, extractor):
        """Test fails to pattern."""
        text = "The system fails to handle concurrent requests properly."
        requirements = extractor.extract(text)
        assert len(requirements) >= 1

    def test_realistic_swebench_issue(self, extractor):
        """Test with realistic SWE-bench-like issue."""
        text = """
        QuerySet.union() ignores order_by() after certain operations

        Description:

        When using QuerySet.union() followed by order_by(), the ordering
        should be applied to the final result. Currently, the ordering
        is ignored if union() is called after distinct().

        Expected behavior:
        - The union result should respect the order_by() clause
        - All records should be sorted by the specified field

        Bug: Results are returned in arbitrary order instead of sorted.

        Steps to reproduce:
        1. Create a queryset with distinct()
        2. Call union() with another queryset
        3. Apply order_by()
        4. Observe that results are not sorted
        """
        requirements = extractor.extract(text)
        # Should extract multiple requirements
        assert len(requirements) >= 3
        # Should have reasonable distribution
        categories = [r.category for r in requirements]
        assert "functional" in categories


# =============================================================================
# ElicitationOracle Tests
# =============================================================================


SAMPLE_ISSUE = """
Bug: Login fails with special characters in password

Users cannot log in when their password contains @ or # characters.
The system should accept all printable ASCII characters in passwords.

Expected behavior:
- Login should validate password format correctly
- Error message should be displayed for invalid credentials
- Session should be created on successful login

Steps to reproduce:
1. Create account with password containing @
2. Try to log in
3. Observe login failure
"""


class TestElicitationOracle:
    """Tests for ElicitationOracle class."""

    @pytest.fixture
    def oracle(self) -> ElicitationOracle:
        """Create oracle from sample issue."""
        return ElicitationOracle.from_issue(SAMPLE_ISSUE, seed=42)

    def test_from_issue_extracts_requirements(self, oracle):
        """Test that from_issue extracts requirements."""
        assert len(oracle.requirements) > 0

    def test_from_issue_creates_scorer(self, oracle):
        """Test that scorer is initialized."""
        assert oracle.scorer is not None

    def test_ask_returns_response(self, oracle):
        """Test that ask returns an OracleResponse."""
        response = oracle.ask("What error do users see?")
        assert response is not None
        assert hasattr(response, "answer")
        assert hasattr(response, "relevance_score")
        assert hasattr(response, "revealed_requirements")
        assert hasattr(response, "question_category")

    def test_ask_scores_relevance(self, oracle):
        """Test that relevant questions get higher scores."""
        relevant_response = oracle.ask("What happens with special characters in password?")
        irrelevant_response = oracle.ask("What color is the sky?")
        assert relevant_response.relevance_score > irrelevant_response.relevance_score

    def test_ask_reveals_requirements(self, oracle):
        """Test that good questions reveal requirements."""
        response = oracle.ask("What error occurs during login with special characters?")
        # Should reveal at least one requirement
        if response.relevance_score >= oracle.reveal_threshold:
            assert len(response.revealed_requirements) > 0

    def test_ask_classifies_question(self, oracle):
        """Test that questions are classified."""
        response = oracle.ask("What do you mean by login failure?")
        assert response.question_category in QuestionCategory

    def test_revealed_count_tracks_progress(self, oracle):
        """Test that revealed_count tracks discoveries."""
        initial = oracle.revealed_count
        oracle.ask("What password characters cause login issues?")
        oracle.ask("What error message appears during login failure?")
        # May or may not reveal, but count should be >= initial
        assert oracle.revealed_count >= initial

    def test_questions_asked_tracks_history(self, oracle):
        """Test that questions_asked tracks count."""
        assert oracle.questions_asked == 0
        oracle.ask("Question 1?")
        assert oracle.questions_asked == 1
        oracle.ask("Question 2?")
        assert oracle.questions_asked == 2

    def test_reset_clears_state(self, oracle):
        """Test that reset clears revealed and history."""
        oracle.ask("What password issue occurs?")
        oracle.ask("What error appears?")
        assert oracle.questions_asked > 0

        oracle.reset()
        assert oracle.questions_asked == 0
        assert oracle.revealed_count == 0

    def test_get_metrics_returns_valid_metrics(self, oracle):
        """Test that get_metrics returns ElicitationMetrics."""
        oracle.ask("What login error occurs with special chars?")
        oracle.ask("What password characters fail?")
        metrics = oracle.get_metrics()

        assert 0.0 <= metrics.discovery_rate <= 1.0
        assert metrics.total_questions == 2
        assert isinstance(metrics.question_distribution, dict)
        assert isinstance(metrics.revealed_requirements, list)
        assert isinstance(metrics.hidden_requirements, list)


class TestOraclePersonalities:
    """Test different oracle personality modes."""

    def test_helpful_personality(self):
        """Test helpful personality gives informative responses."""
        oracle = ElicitationOracle.from_issue(
            SAMPLE_ISSUE, personality="helpful", seed=42
        )
        response = oracle.ask("What login issues occur?")
        # Helpful responses should be substantive
        assert len(response.answer) > 10

    def test_terse_personality(self):
        """Test terse personality gives brief responses."""
        oracle = ElicitationOracle.from_issue(
            SAMPLE_ISSUE, personality="terse", seed=42
        )
        response = oracle.ask("What login issues occur?")
        # Terse responses exist
        assert response.answer

    def test_confused_personality(self):
        """Test confused personality mode."""
        oracle = ElicitationOracle.from_issue(
            SAMPLE_ISSUE, personality="confused", seed=42
        )
        response = oracle.ask("What login issues occur?")
        assert response.answer

    def test_personality_enum(self):
        """Test using personality enum directly."""
        oracle = ElicitationOracle.from_issue(
            SAMPLE_ISSUE, personality=OraclePersonality.HELPFUL
        )
        assert oracle.personality == OraclePersonality.HELPFUL


class TestOracleFromRequirements:
    """Test creating oracle from pre-extracted requirements."""

    def test_from_requirements(self):
        """Test creating oracle from requirement list."""
        requirements = [
            Requirement(
                id="R1",
                text="Login should validate password",
                keywords=["login", "password", "validate"],
            ),
            Requirement(
                id="R2",
                text="Error message should be clear",
                keywords=["error", "message", "clear"],
            ),
        ]
        oracle = ElicitationOracle.from_requirements(requirements, seed=42)
        assert oracle.total_requirements == 2

    def test_ask_with_custom_requirements(self):
        """Test asking questions with custom requirements."""
        requirements = [
            Requirement(
                id="R1",
                text="Login validates password format",
                keywords=["login", "password", "format"],
            ),
        ]
        oracle = ElicitationOracle.from_requirements(requirements, seed=42)
        response = oracle.ask("How does login validate password?")
        assert response.relevance_score > 0


class TestOracleThreshold:
    """Test reveal threshold behavior."""

    def test_high_threshold_reveals_less(self):
        """Test that high threshold reveals fewer requirements."""
        strict = ElicitationOracle.from_issue(
            SAMPLE_ISSUE, reveal_threshold=0.8, seed=42
        )
        lenient = ElicitationOracle.from_issue(
            SAMPLE_ISSUE, reveal_threshold=0.2, seed=42
        )

        # Ask the same questions to both
        questions = [
            "What login problem exists?",
            "What characters cause issues?",
            "What error appears?",
        ]
        for q in questions:
            strict.ask(q)
            lenient.ask(q)

        # Lenient should reveal at least as many
        assert lenient.revealed_count >= strict.revealed_count


class TestOracleDeterminism:
    """Test deterministic behavior with seeds."""

    def test_same_seed_same_results(self):
        """Test that same seed produces same results."""
        oracle1 = ElicitationOracle.from_issue(SAMPLE_ISSUE, seed=42)
        oracle2 = ElicitationOracle.from_issue(SAMPLE_ISSUE, seed=42)

        response1 = oracle1.ask("What login error occurs?")
        response2 = oracle2.ask("What login error occurs?")

        assert response1.relevance_score == response2.relevance_score
        assert response1.revealed_requirements == response2.revealed_requirements

    def test_different_seed_may_differ(self):
        """Test that different seeds may produce different responses."""
        oracle1 = ElicitationOracle.from_issue(SAMPLE_ISSUE, seed=42)
        oracle2 = ElicitationOracle.from_issue(SAMPLE_ISSUE, seed=123)

        # Scores should be the same (deterministic scoring)
        response1 = oracle1.ask("What login error occurs?")
        response2 = oracle2.ask("What login error occurs?")

        assert response1.relevance_score == response2.relevance_score
        # But response text may vary due to random template selection


class TestOracleDialogue:
    """Test multi-turn dialogue scenarios."""

    def test_dialogue_session(self):
        """Test a complete dialogue session."""
        oracle = ElicitationOracle.from_issue(SAMPLE_ISSUE, seed=42)

        # Simulate a dialogue
        questions = [
            "What is the main problem?",
            "What happens when special characters are in password?",
            "What error message do users see?",
            "How should the login be fixed?",
            "Are there any edge cases to consider?",
        ]

        for q in questions:
            response = oracle.ask(q)
            assert response.answer
            assert 0.0 <= response.relevance_score <= 1.0

        # Get final metrics
        metrics = oracle.get_metrics()
        assert metrics.total_questions == len(questions)
        assert 0.0 <= metrics.discovery_rate <= 1.0
        assert len(metrics.question_distribution) > 0

    def test_requirement_not_revealed_twice(self):
        """Test that same requirement isn't revealed twice."""
        oracle = ElicitationOracle.from_issue(SAMPLE_ISSUE, seed=42)

        # Ask the same question twice
        response1 = oracle.ask("What password validation is needed?")
        revealed1 = response1.revealed_requirements

        response2 = oracle.ask("What password validation is needed?")
        revealed2 = response2.revealed_requirements

        # Second response shouldn't reveal same requirements
        for req_id in revealed1:
            assert req_id not in revealed2
