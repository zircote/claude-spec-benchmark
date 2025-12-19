"""Question scoring module for elicitation evaluation.

This module provides the QuestionScorer class that measures how relevant
an agent's question is to hidden requirements. It uses a combination of
keyword matching and TF-IDF weighting to produce relevance scores.

The scorer supports Socratic question classification to measure question
diversity and quality across the elicitation dialogue.

Example:
    >>> from claude_spec_benchmark.elicitation.scoring import QuestionScorer
    >>> from claude_spec_benchmark.models import Requirement
    >>> requirements = [
    ...     Requirement(id="REQ-1", text="Login should validate email format",
    ...                 keywords=["login", "email", "validation"]),
    ... ]
    >>> scorer = QuestionScorer(requirements)
    >>> score = scorer.score("How does the login form validate emails?")
    >>> print(f"Relevance: {score:.2f}")
    Relevance: 0.75
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from claude_spec_benchmark.models import QuestionCategory, Requirement

# Common stop words to filter out during keyword extraction
STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "dare",
    "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
    "from", "up", "about", "into", "over", "after", "beneath", "under",
    "above", "and", "but", "or", "nor", "so", "yet", "both", "either",
    "neither", "not", "only", "own", "same", "than", "too", "very",
    "just", "also", "now", "here", "there", "when", "where", "why",
    "how", "all", "each", "every", "few", "more", "most",
    "other", "some", "such", "no", "any", "this", "that", "these",
    "those", "it", "its", "what", "which", "who", "whom", "whose",
})

# Patterns for Socratic question classification
SOCRATIC_PATTERNS: dict[QuestionCategory, list[re.Pattern[str]]] = {
    QuestionCategory.CLARIFICATION: [
        re.compile(r"\bwhat\s+(do|does|is|are)\s+you\s+mean\b", re.IGNORECASE),
        re.compile(r"\bclarify\b", re.IGNORECASE),
        re.compile(r"\bexplain\b.*\?$", re.IGNORECASE),
        re.compile(r"\bwhat\s+(exactly|specifically)\b", re.IGNORECASE),
        re.compile(r"\bcan\s+you\s+(describe|elaborate)\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+do\s+you\s+mean\b", re.IGNORECASE),
        re.compile(r"\bhow\s+would\s+you\s+define\b", re.IGNORECASE),
    ],
    QuestionCategory.ASSUMPTION: [
        re.compile(r"\bwhat\s+(are\s+)?you\s+assum", re.IGNORECASE),
        re.compile(r"\bassum(e|ing|ption)\b", re.IGNORECASE),
        re.compile(r"\btake\s+for\s+granted\b", re.IGNORECASE),
        re.compile(r"\bwhy\s+do\s+you\s+think\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+if\s+.+\s+not\s+true\b", re.IGNORECASE),
    ],
    QuestionCategory.EVIDENCE: [
        re.compile(r"\bwhat\s+evidence\b", re.IGNORECASE),
        re.compile(r"\bhow\s+do\s+(you|we)\s+know\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+(data|proof|example)\b", re.IGNORECASE),
        re.compile(r"\bcan\s+you\s+(show|prove|demonstrate)\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+supports\b", re.IGNORECASE),
    ],
    QuestionCategory.VIEWPOINT: [
        re.compile(r"\balternative\b", re.IGNORECASE),
        re.compile(r"\banother\s+(way|approach|perspective)\b", re.IGNORECASE),
        re.compile(r"\bdifferent\s+(view|opinion|angle)\b", re.IGNORECASE),
        re.compile(r"\bhow\s+else\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+about\b", re.IGNORECASE),
    ],
    QuestionCategory.IMPLICATION: [
        re.compile(r"\bwhat\s+(would|will)\s+happen\b", re.IGNORECASE),
        re.compile(r"\bconsequence\b", re.IGNORECASE),
        re.compile(r"\bimplication\b", re.IGNORECASE),
        re.compile(r"\bif\s+.+\s+then\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+follows\b", re.IGNORECASE),
        re.compile(r"\bresult\s+in\b", re.IGNORECASE),
    ],
    QuestionCategory.META: [
        re.compile(r"\bwhy\s+(is|are)\s+(this|these)\s+question", re.IGNORECASE),
        re.compile(r"\bwhy\s+(ask|asking)\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+.+\s+trying\s+to\s+(understand|learn)\b", re.IGNORECASE),
        re.compile(r"\bpurpose\s+of\s+(this|the)\s+question\b", re.IGNORECASE),
    ],
}


def extract_keywords(text: str) -> list[str]:
    """Extract meaningful keywords from text.

    Filters out stop words and short tokens to identify substantive terms.

    Args:
        text: Input text to extract keywords from.

    Returns:
        List of lowercase keywords (unique, ordered by first occurrence).

    Example:
        >>> extract_keywords("How does the login form validate emails?")
        ['login', 'form', 'validate', 'emails']
    """
    # Tokenize: split on non-alphanumeric
    tokens = re.findall(r"\b[a-zA-Z][a-zA-Z0-9_]*\b", text.lower())

    # Filter: remove stop words and short tokens
    seen: set[str] = set()
    keywords: list[str] = []
    for token in tokens:
        if token not in STOP_WORDS and len(token) > 2 and token not in seen:
            seen.add(token)
            keywords.append(token)

    return keywords


@dataclass
class QuestionScorer:
    """Scores questions against hidden requirements for relevance.

    Uses keyword matching with TF-IDF weighting to determine how
    relevant a question is to the set of hidden requirements. Also
    classifies questions into Socratic categories.

    Attributes:
        requirements: List of requirements to score against.
        _idf: Computed IDF weights for requirement keywords.
        _req_keywords: Precomputed keyword sets per requirement.

    Example:
        >>> from claude_spec_benchmark.models import Requirement
        >>> reqs = [
        ...     Requirement(id="R1", text="Handle auth errors",
        ...                 keywords=["auth", "error", "handle"]),
        ...     Requirement(id="R2", text="Log failed attempts",
        ...                 keywords=["log", "failed", "attempts"]),
        ... ]
        >>> scorer = QuestionScorer(reqs)
        >>> scorer.score("What errors occur during authentication?")
        0.67  # High relevance to R1
    """

    requirements: list[Requirement]
    _idf: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _req_keywords: dict[str, set[str]] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        """Compute IDF weights and index requirement keywords."""
        self._build_index()

    def _build_index(self) -> None:
        """Build keyword index and compute IDF weights."""
        # Collect all keywords per requirement
        doc_freq: Counter[str] = Counter()
        total_docs = len(self.requirements)

        for req in self.requirements:
            # Combine explicit keywords with extracted keywords from text
            all_keywords = set(req.keywords) | set(extract_keywords(req.text))
            self._req_keywords[req.id] = all_keywords

            # Track document frequency for IDF
            for kw in all_keywords:
                doc_freq[kw] += 1

        # Compute IDF: log(N / df) for each keyword
        for keyword, df in doc_freq.items():
            self._idf[keyword] = math.log((total_docs + 1) / (df + 1)) + 1.0

    def score(self, question: str) -> float:
        """Score a question's relevance to hidden requirements.

        Uses TF-IDF weighted keyword overlap to produce a score
        between 0.0 (irrelevant) and 1.0 (highly relevant).

        Args:
            question: The question text to score.

        Returns:
            Relevance score between 0.0 and 1.0.

        Example:
            >>> scorer.score("How do I log in?")
            0.45
        """
        if not self.requirements:
            return 0.0

        question_keywords = set(extract_keywords(question))
        if not question_keywords:
            return 0.0

        # Compute weighted match score against each requirement
        max_score = 0.0
        for req_keywords in self._req_keywords.values():
            overlap = question_keywords & req_keywords
            if not overlap:
                continue

            # TF-IDF weighted score
            weighted_overlap = sum(self._idf.get(kw, 1.0) for kw in overlap)
            total_weight = sum(self._idf.get(kw, 1.0) for kw in req_keywords)

            if total_weight > 0:
                score = weighted_overlap / total_weight
                max_score = max(max_score, score)

        # Cap at 1.0
        return min(1.0, max_score)

    def score_detailed(
        self, question: str
    ) -> tuple[float, list[tuple[str, float]]]:
        """Score a question with per-requirement breakdown.

        Args:
            question: The question text to score.

        Returns:
            Tuple of (overall_score, list of (requirement_id, score) pairs).

        Example:
            >>> score, details = scorer.score_detailed("auth errors?")
            >>> for req_id, req_score in details:
            ...     print(f"{req_id}: {req_score:.2f}")
        """
        if not self.requirements:
            return 0.0, []

        question_keywords = set(extract_keywords(question))
        if not question_keywords:
            return 0.0, []

        scores: list[tuple[str, float]] = []
        for req_id, req_keywords in self._req_keywords.items():
            overlap = question_keywords & req_keywords
            if not overlap:
                scores.append((req_id, 0.0))
                continue

            weighted_overlap = sum(self._idf.get(kw, 1.0) for kw in overlap)
            total_weight = sum(self._idf.get(kw, 1.0) for kw in req_keywords)

            score = weighted_overlap / total_weight if total_weight > 0 else 0.0
            scores.append((req_id, min(1.0, score)))

        overall = max(s for _, s in scores) if scores else 0.0
        return overall, sorted(scores, key=lambda x: x[1], reverse=True)

    def classify_question(self, question: str) -> QuestionCategory:
        """Classify a question into a Socratic category.

        Uses pattern matching to determine what type of question
        is being asked (clarification, assumption, evidence, etc.).

        Args:
            question: The question text to classify.

        Returns:
            QuestionCategory enum value.

        Example:
            >>> scorer.classify_question("What do you mean by 'valid'?")
            QuestionCategory.CLARIFICATION
        """
        for category, patterns in SOCRATIC_PATTERNS.items():
            for pattern in patterns:
                if pattern.search(question):
                    return category

        # Check for basic question words as fallback
        question_lower = question.lower()
        if "what" in question_lower:
            return QuestionCategory.CLARIFICATION
        if "why" in question_lower:
            return QuestionCategory.ASSUMPTION
        if "how" in question_lower:
            return QuestionCategory.EVIDENCE

        return QuestionCategory.UNKNOWN

    def get_matching_requirements(
        self, question: str, threshold: float = 0.3
    ) -> list[str]:
        """Get requirement IDs that match a question above threshold.

        Args:
            question: The question to match against.
            threshold: Minimum score for a match (default 0.3).

        Returns:
            List of requirement IDs with scores >= threshold.

        Example:
            >>> scorer.get_matching_requirements("auth errors?", threshold=0.5)
            ["R1"]
        """
        _, details = self.score_detailed(question)
        return [req_id for req_id, score in details if score >= threshold]
