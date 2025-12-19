"""Elicitation oracle module for SDD-Bench.

This module provides the ElicitationOracle class that simulates a stakeholder
responding to questions during requirements elicitation. The oracle has
"hidden knowledge" (extracted requirements) and reveals information based
on how relevant the agent's questions are.

The oracle supports different personality modes to simulate various
stakeholder behaviors:
- helpful: Provides complete, direct answers
- terse: Gives minimal, brief responses
- confused: Occasionally misunderstands or gives irrelevant answers

Example:
    >>> from claude_spec_benchmark.elicitation.oracle import ElicitationOracle
    >>> oracle = ElicitationOracle.from_issue(issue_text)
    >>> response = oracle.ask("What error do users see when login fails?")
    >>> print(f"Score: {response.relevance_score:.2f}")
    >>> print(f"Answer: {response.answer}")
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from claude_spec_benchmark.elicitation.extraction import RequirementsExtractor
from claude_spec_benchmark.elicitation.scoring import QuestionScorer
from claude_spec_benchmark.models import (
    ElicitationMetrics,
    OracleResponse,
    QuestionCategory,
    Requirement,
)


class OraclePersonality(str, Enum):
    """Oracle personality modes affecting response behavior."""

    HELPFUL = "helpful"  # Complete, direct answers
    TERSE = "terse"  # Minimal, brief responses
    CONFUSED = "confused"  # May misunderstand or digress


# Response templates by personality and relevance level
RESPONSE_TEMPLATES: dict[OraclePersonality, dict[str, list[str]]] = {
    OraclePersonality.HELPFUL: {
        "high": [
            "Yes, exactly! {detail}",
            "That's a great question. {detail}",
            "Right, so {detail}",
            "{detail} Does that help clarify?",
        ],
        "medium": [
            "Let me think... {detail}",
            "Well, {detail}",
            "I believe {detail}",
            "From what I understand, {detail}",
        ],
        "low": [
            "Hmm, I'm not sure that's quite the issue. What we're seeing is {vague}.",
            "That's not really related. The problem is more about {vague}.",
            "I don't think that's relevant. We should focus on {vague}.",
        ],
        "none": [
            "I'm not sure I understand what you're asking.",
            "Could you rephrase that?",
            "That doesn't seem related to the issue we're discussing.",
        ],
    },
    OraclePersonality.TERSE: {
        "high": [
            "{detail}",
            "Yes. {detail}",
            "{detail} That's correct.",
        ],
        "medium": [
            "Maybe. {detail}",
            "Possibly. {detail}",
        ],
        "low": [
            "No.",
            "Not relevant.",
            "Try another angle.",
        ],
        "none": [
            "?",
            "Unclear.",
            "Rephrase.",
        ],
    },
    OraclePersonality.CONFUSED: {
        "high": [
            "Oh yes! Wait, or was it... no, {detail}",
            "Um, I think {detail} But I might be mixing things up.",
            "{detail} At least I'm pretty sure.",
        ],
        "medium": [
            "Well, um, {detail} Or something like that.",
            "I think... {detail} Maybe?",
            "Let me see... {vague}",
        ],
        "low": [
            "What? Oh, sorry, I was thinking about something else. What were you asking?",
            "Hmm? I don't think so but I could be wrong.",
            "That reminds me of something else entirely...",
        ],
        "none": [
            "Sorry, could you repeat that?",
            "I got distracted. What was the question?",
            "...",
        ],
    },
}


def _relevance_to_level(score: float) -> str:
    """Convert a relevance score to a response level.

    Args:
        score: Relevance score between 0.0 and 1.0.

    Returns:
        Level string: "high", "medium", "low", or "none".
    """
    if score >= 0.6:
        return "high"
    if score >= 0.35:
        return "medium"
    if score > 0.0:
        return "low"
    return "none"


@dataclass
class ElicitationOracle:
    """Simulates a stakeholder for requirements elicitation.

    The oracle holds "hidden knowledge" extracted from the full issue text
    and reveals requirements based on how relevant the agent's questions are.
    Different personality modes simulate various stakeholder behaviors.

    Attributes:
        requirements: List of hidden requirements to reveal.
        scorer: QuestionScorer for evaluating question relevance.
        personality: Oracle personality mode.
        reveal_threshold: Minimum relevance score to reveal a requirement.
        rng: Random generator for response selection.
        _revealed: Set of requirement IDs already revealed.
        _question_history: List of (question, category, score) tuples.

    Example:
        >>> oracle = ElicitationOracle.from_issue(issue_text)
        >>> response = oracle.ask("What error message appears?")
        >>> print(f"Revealed: {response.revealed_requirements}")
    """

    requirements: list[Requirement]
    scorer: QuestionScorer = field(init=False)
    personality: OraclePersonality = OraclePersonality.HELPFUL
    reveal_threshold: float = 0.4
    rng: random.Random = field(default_factory=random.Random)
    _revealed: set[str] = field(default_factory=set, init=False, repr=False)
    _question_history: list[tuple[str, QuestionCategory, float]] = field(
        default_factory=list, init=False, repr=False
    )

    def __post_init__(self) -> None:
        """Initialize the scorer from requirements."""
        self.scorer = QuestionScorer(self.requirements)

    @classmethod
    def from_issue(
        cls,
        issue_text: str,
        personality: OraclePersonality | Literal["helpful", "terse", "confused"] = "helpful",
        reveal_threshold: float = 0.4,
        seed: int | None = None,
    ) -> ElicitationOracle:
        """Create an oracle from issue text.

        Automatically extracts requirements from the issue to use as
        hidden knowledge.

        Args:
            issue_text: Full issue text from SWE-bench.
            personality: Oracle personality mode.
            reveal_threshold: Score threshold for revealing requirements.
            seed: Random seed for reproducibility.

        Returns:
            Configured ElicitationOracle instance.

        Example:
            >>> oracle = ElicitationOracle.from_issue(
            ...     "Bug: Login fails with special characters...",
            ...     personality="helpful",
            ...     seed=42,
            ... )
        """
        # Extract requirements
        extractor = RequirementsExtractor()
        requirements = extractor.extract(issue_text)

        # Convert string personality to enum
        if isinstance(personality, str):
            personality = OraclePersonality(personality)

        # Create RNG (not for cryptography - only for deterministic response selection)
        rng = random.Random(seed) if seed is not None else random.Random()  # noqa: S311

        return cls(
            requirements=requirements,
            personality=personality,
            reveal_threshold=reveal_threshold,
            rng=rng,
        )

    @classmethod
    def from_requirements(
        cls,
        requirements: list[Requirement],
        personality: OraclePersonality | Literal["helpful", "terse", "confused"] = "helpful",
        reveal_threshold: float = 0.4,
        seed: int | None = None,
    ) -> ElicitationOracle:
        """Create an oracle from pre-extracted requirements.

        Args:
            requirements: List of Requirement objects.
            personality: Oracle personality mode.
            reveal_threshold: Score threshold for revealing requirements.
            seed: Random seed for reproducibility.

        Returns:
            Configured ElicitationOracle instance.
        """
        if isinstance(personality, str):
            personality = OraclePersonality(personality)

        # Create RNG (not for cryptography - only for deterministic response selection)
        rng = random.Random(seed) if seed is not None else random.Random()  # noqa: S311

        return cls(
            requirements=requirements,
            personality=personality,
            reveal_threshold=reveal_threshold,
            rng=rng,
        )

    def ask(self, question: str) -> OracleResponse:
        """Ask the oracle a question.

        The oracle scores the question's relevance to hidden requirements,
        generates an appropriate response based on personality, and tracks
        which requirements are revealed.

        Args:
            question: The question text to ask.

        Returns:
            OracleResponse with answer, score, and revealed requirements.

        Example:
            >>> response = oracle.ask("What happens when login fails?")
            >>> if response.relevance_score > 0.5:
            ...     print("Good question!")
        """
        # Score the question
        score, details = self.scorer.score_detailed(question)

        # Classify the question
        category = self.scorer.classify_question(question)

        # Record in history
        self._question_history.append((question, category, score))

        # Determine which requirements to reveal
        newly_revealed: list[str] = []
        for req_id, req_score in details:
            if req_score >= self.reveal_threshold and req_id not in self._revealed:
                self._revealed.add(req_id)
                newly_revealed.append(req_id)

        # Generate response
        answer = self._generate_response(score, newly_revealed)

        return OracleResponse(
            answer=answer,
            relevance_score=score,
            revealed_requirements=newly_revealed,
            question_category=category,
        )

    def _generate_response(
        self, score: float, newly_revealed: list[str]
    ) -> str:
        """Generate a response based on score and personality.

        Args:
            score: Question relevance score.
            newly_revealed: List of requirement IDs just revealed.

        Returns:
            Response text string.
        """
        level = _relevance_to_level(score)
        templates = RESPONSE_TEMPLATES[self.personality][level]
        template = self.rng.choice(templates)

        # Get detail text from revealed requirements
        if newly_revealed:
            req_texts = []
            for req_id in newly_revealed:
                for req in self.requirements:
                    if req.id == req_id:
                        req_texts.append(req.text)
                        break
            detail = " ".join(req_texts[:2])  # Limit to 2 requirements
        else:
            # Generic detail for medium/low relevance
            if self.requirements:
                # Hint at requirements without revealing
                detail = f"something related to {self.requirements[0].category} requirements"
            else:
                detail = "the issue at hand"

        # Vague hint for low relevance
        vague = "the general behavior of the system"
        if self.requirements:
            keywords = []
            for req in self.requirements[:3]:
                keywords.extend(req.keywords[:2])
            if keywords:
                vague = f"issues with {', '.join(keywords[:3])}"

        return template.format(detail=detail, vague=vague)

    def reset(self) -> None:
        """Reset the oracle state for a new elicitation session.

        Clears revealed requirements and question history while keeping
        the same hidden knowledge.
        """
        self._revealed.clear()
        self._question_history.clear()

    def get_metrics(self) -> ElicitationMetrics:
        """Get metrics summarizing the elicitation session.

        Returns:
            ElicitationMetrics with discovery rate, efficiency, etc.

        Example:
            >>> metrics = oracle.get_metrics()
            >>> print(f"Discovery rate: {metrics.discovery_rate:.1%}")
        """
        # Count discoverable requirements
        discoverable = [r for r in self.requirements if r.discoverable]
        total_discoverable = len(discoverable)

        # Discovery rate
        revealed_discoverable = len(
            self._revealed & {r.id for r in discoverable}
        )
        discovery_rate = (
            revealed_discoverable / total_discoverable
            if total_discoverable > 0
            else 0.0
        )

        # Question efficiency
        total_questions = len(self._question_history)
        efficiency = (
            revealed_discoverable / total_questions
            if total_questions > 0
            else 0.0
        )

        # Question distribution
        distribution: dict[str, int] = {}
        for _, category, _ in self._question_history:
            cat_name = category.value
            distribution[cat_name] = distribution.get(cat_name, 0) + 1

        # Revealed vs hidden requirements
        revealed_list = list(self._revealed)
        hidden_list = [
            r.id for r in self.requirements if r.id not in self._revealed
        ]

        return ElicitationMetrics(
            discovery_rate=discovery_rate,
            question_efficiency=efficiency,
            total_questions=total_questions,
            question_distribution=distribution,
            revealed_requirements=revealed_list,
            hidden_requirements=hidden_list,
        )

    @property
    def revealed_count(self) -> int:
        """Number of requirements revealed so far."""
        return len(self._revealed)

    @property
    def total_requirements(self) -> int:
        """Total number of hidden requirements."""
        return len(self.requirements)

    @property
    def questions_asked(self) -> int:
        """Number of questions asked so far."""
        return len(self._question_history)
