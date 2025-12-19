"""Core degradation engine for specification degradation.

This module implements the DegradationEngine class that transforms complete
SWE-bench issues into progressively vaguer specifications to simulate
incomplete requirements in real-world development scenarios.

The engine supports 5 degradation levels:
- FULL: No degradation (original text)
- PARTIAL: Remove code snippets, stack traces, file paths
- VAGUE: High-level description only
- MINIMAL: Single sentence summary
- AMBIGUOUS: Intentionally unclear with contradictions

Example:
    >>> from claude_spec_benchmark.degradation import DegradationEngine
    >>> from claude_spec_benchmark.models import SpecDegradationLevel
    >>> engine = DegradationEngine()
    >>> result = engine.degrade(
    ...     full_issue="Error in auth/views.py line 42...",
    ...     level=SpecDegradationLevel.VAGUE,
    ...     seed=42,
    ... )
    >>> print(result.degraded_text)
    "There's an issue with the authentication system..."
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field

from claude_spec_benchmark.models import DegradedSpec, SpecDegradationLevel

from .patterns import DegradationPatterns


@dataclass
class DegradationEngine:
    """Engine for degrading specifications to simulate incomplete requirements.

    Transforms complete issue descriptions into progressively vaguer versions
    while tracking what information was hidden for later scoring.

    Attributes:
        patterns: Pattern configuration for matching content to remove.

    Example:
        >>> engine = DegradationEngine()
        >>> result = engine.degrade(
        ...     "Fix bug in django/contrib/auth/views.py at line 42",
        ...     SpecDegradationLevel.PARTIAL,
        ...     seed=42,
        ... )
        >>> "django/contrib/auth/views.py" in result.hidden_details
        True
    """

    patterns: DegradationPatterns = field(default_factory=DegradationPatterns.default)

    def degrade(
        self,
        full_issue: str,
        level: SpecDegradationLevel,
        seed: int | None = None,
        repo: str | None = None,
    ) -> DegradedSpec:
        """Degrade a specification to the given level.

        Args:
            full_issue: Complete issue text from SWE-bench.
            level: Target degradation level.
            seed: Random seed for reproducible degradation.
            repo: Optional repository name for repo-specific patterns.

        Returns:
            DegradedSpec with degraded text and list of hidden details.

        Example:
            >>> engine = DegradationEngine()
            >>> result = engine.degrade(
            ...     "File auth.py has bug", SpecDegradationLevel.FULL
            ... )
            >>> result.degraded_text == result.original_text
            True
        """
        # Use repo-specific patterns if available
        patterns = DegradationPatterns.for_repo(repo) if repo else self.patterns

        # Initialize deterministic random if seed provided
        # Note: Using standard random for reproducibility, not cryptographic purposes
        rng = random.Random(seed) if seed is not None else random.Random()  # noqa: S311

        # Dispatch to appropriate level handler
        match level:
            case SpecDegradationLevel.FULL:
                return self._degrade_full(full_issue, seed)
            case SpecDegradationLevel.PARTIAL:
                return self._degrade_partial(full_issue, patterns, rng, seed)
            case SpecDegradationLevel.VAGUE:
                return self._degrade_vague(full_issue, patterns, rng, seed)
            case SpecDegradationLevel.MINIMAL:
                return self._degrade_minimal(full_issue, patterns, rng, seed)
            case SpecDegradationLevel.AMBIGUOUS:
                return self._degrade_ambiguous(full_issue, patterns, rng, seed)
            case _:
                msg = f"Unknown degradation level: {level}"
                raise ValueError(msg)

    def _degrade_full(
        self,
        full_issue: str,
        seed: int | None,
    ) -> DegradedSpec:
        """FULL level: Return original text unchanged (baseline).

        Used as a control condition to compare against degraded versions.
        """
        return DegradedSpec(
            degraded_text=full_issue,
            hidden_details=[],
            original_text=full_issue,
            level=SpecDegradationLevel.FULL,
            seed=seed,
        )

    def _degrade_partial(
        self,
        full_issue: str,
        patterns: DegradationPatterns,
        rng: random.Random,  # noqa: ARG002 - kept for API consistency
        seed: int | None,
    ) -> DegradedSpec:
        """PARTIAL level: Remove code snippets, stack traces, file paths.

        Removes technical details while preserving the narrative description.
        This simulates a user who provides a good problem description but
        doesn't include technical debugging information.
        """
        hidden_details: list[str] = []
        text = full_issue

        # Apply patterns for partial level
        pattern_names = patterns.get_patterns_for_level("partial")

        for pattern_name in pattern_names:
            matches = patterns.find_all(pattern_name, text)
            for match_text, _, _ in matches:
                if match_text.strip():
                    hidden_details.append(
                        f"[{self._pattern_to_label(pattern_name)}] {match_text[:100]}..."
                        if len(match_text) > 100
                        else f"[{self._pattern_to_label(pattern_name)}] {match_text}"
                    )
            text = patterns.remove_all(pattern_name, text, replacement=" [removed] ")

        # Clean up excessive whitespace and placeholders
        text = self._clean_text(text)

        return DegradedSpec(
            degraded_text=text,
            hidden_details=hidden_details,
            original_text=full_issue,
            level=SpecDegradationLevel.PARTIAL,
            seed=seed,
        )

    def _degrade_vague(
        self,
        full_issue: str,
        patterns: DegradationPatterns,
        rng: random.Random,
        seed: int | None,
    ) -> DegradedSpec:
        """VAGUE level: Only high-level problem description retained.

        Aggressively removes technical content, leaving only the general
        nature of the problem. Simulates a non-technical stakeholder's
        bug report.
        """
        hidden_details: list[str] = []
        text = full_issue

        # Apply all vague-level patterns
        pattern_names = patterns.get_patterns_for_level("vague")

        for pattern_name in pattern_names:
            matches = patterns.find_all(pattern_name, text)
            for match_text, _, _ in matches:
                if match_text.strip():
                    hidden_details.append(
                        f"[{self._pattern_to_label(pattern_name)}] {match_text[:80]}..."
                        if len(match_text) > 80
                        else f"[{self._pattern_to_label(pattern_name)}] {match_text}"
                    )
            text = patterns.remove_all(pattern_name, text, replacement=" ")

        # Clean up and simplify language
        text = self._clean_text(text)
        text = self._simplify_language(text, rng)

        return DegradedSpec(
            degraded_text=text,
            hidden_details=hidden_details,
            original_text=full_issue,
            level=SpecDegradationLevel.VAGUE,
            seed=seed,
        )

    def _degrade_minimal(
        self,
        full_issue: str,
        patterns: DegradationPatterns,
        rng: random.Random,
        seed: int | None,
    ) -> DegradedSpec:
        """MINIMAL level: Single sentence summary of the issue.

        Reduces the entire issue to a brief, high-level summary.
        Simulates the initial complaint from a user who hasn't provided
        any details yet.
        """
        hidden_details: list[str] = []
        text = full_issue

        # Track all content that will be hidden
        pattern_names = patterns.get_patterns_for_level("minimal")
        for pattern_name in pattern_names:
            matches = patterns.find_all(pattern_name, text)
            for match_text, _, _ in matches:
                if match_text.strip():
                    hidden_details.append(
                        f"[{self._pattern_to_label(pattern_name)}] {match_text[:60]}..."
                        if len(match_text) > 60
                        else f"[{self._pattern_to_label(pattern_name)}] {match_text}"
                    )

        # Generate a minimal summary
        summary = self._generate_summary(full_issue, rng)

        # The entire original text (minus summary) is hidden detail
        hidden_details.insert(0, "[full_context] Original issue text hidden")

        return DegradedSpec(
            degraded_text=summary,
            hidden_details=hidden_details,
            original_text=full_issue,
            level=SpecDegradationLevel.MINIMAL,
            seed=seed,
        )

    def _degrade_ambiguous(
        self,
        full_issue: str,
        patterns: DegradationPatterns,
        rng: random.Random,
        seed: int | None,
    ) -> DegradedSpec:
        """AMBIGUOUS level: Intentionally unclear or contradictory description.

        Creates a degraded spec that contains misleading or contradictory
        information. This tests an agent's ability to detect and resolve
        ambiguity through elicitation.
        """
        # First apply minimal-level degradation
        hidden_details: list[str] = []
        text = full_issue

        pattern_names = patterns.get_patterns_for_level("ambiguous")
        for pattern_name in pattern_names:
            matches = patterns.find_all(pattern_name, text)
            for match_text, _, _ in matches:
                if match_text.strip():
                    hidden_details.append(
                        f"[{self._pattern_to_label(pattern_name)}] {match_text[:60]}..."
                        if len(match_text) > 60
                        else f"[{self._pattern_to_label(pattern_name)}] {match_text}"
                    )
            text = patterns.remove_all(pattern_name, text, replacement=" ")

        text = self._clean_text(text)
        text = self._simplify_language(text, rng)

        # Add contradictions and ambiguity
        text = self._add_ambiguity(text, full_issue, rng)

        hidden_details.insert(0, "[ambiguity] Contradictions intentionally introduced")

        return DegradedSpec(
            degraded_text=text,
            hidden_details=hidden_details,
            original_text=full_issue,
            level=SpecDegradationLevel.AMBIGUOUS,
            seed=seed,
        )

    def _pattern_to_label(self, pattern_name: str) -> str:
        """Convert pattern name to human-readable label."""
        labels = {
            "code_block_pattern": "code",
            "inline_code_pattern": "code",
            "stack_trace_pattern": "stack_trace",
            "exception_pattern": "exception",
            "file_path_pattern": "file_path",
            "line_number_pattern": "line_number",
            "function_ref_pattern": "function",
            "class_ref_pattern": "class",
            "method_call_pattern": "method",
            "commit_sha_pattern": "commit",
            "version_pattern": "version",
            "url_pattern": "url",
            "github_ref_pattern": "github_ref",
            "log_line_pattern": "log",
        }
        return labels.get(pattern_name, "detail")

    def _clean_text(self, text: str) -> str:
        """Clean up text after pattern removal."""
        # Remove [removed] placeholders that are adjacent
        text = re.sub(r"(\s*\[removed\]\s*)+", " ", text)
        # Collapse multiple whitespace
        text = re.sub(r"\s+", " ", text)
        # Collapse multiple newlines
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
        # Remove leading/trailing whitespace per line
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines)
        # Remove empty lines at start/end
        text = text.strip()
        return text

    def _simplify_language(self, text: str, rng: random.Random) -> str:
        """Simplify technical language to more vague descriptions."""
        replacements = [
            (r"\berror\b", "issue"),
            (r"\bexception\b", "problem"),
            (r"\bfails?\b", "doesn't work"),
            (r"\bcrash(?:es|ed)?\b", "stops working"),
            (r"\bbug\b", "problem"),
            (r"\breturn(?:s|ed)?\b", "gives"),
            (r"\braise(?:s|d)?\b", "causes"),
            (r"\bfix(?:es|ed)?\b", "resolve"),
            (r"\bimplement(?:s|ed)?\b", "add"),
        ]

        for pattern, replacement in replacements:
            # Only apply some replacements randomly for variety
            if rng.random() > 0.3:
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text

    def _generate_summary(self, full_issue: str, rng: random.Random) -> str:
        """Generate a minimal one-sentence summary of an issue."""
        # Extract key indicators from the text
        keywords = []

        # Check for common issue types
        if re.search(r"\berror\b|\bexception\b|\bfail", full_issue, re.IGNORECASE):
            keywords.append("error")
        if re.search(r"\bcrash\b|\bhang\b|\bfreeze", full_issue, re.IGNORECASE):
            keywords.append("crash")
        if re.search(r"\bslow\b|\bperformance\b|\btimeout", full_issue, re.IGNORECASE):
            keywords.append("performance issue")
        if re.search(
            r"\bincorrect\b|\bwrong\b|\bunexpected", full_issue, re.IGNORECASE
        ):
            keywords.append("incorrect behavior")

        if not keywords:
            keywords.append("issue")

        # Generate vague summary templates
        templates = [
            "Something is not working correctly with {keyword}.",
            "There seems to be a {keyword} that needs to be fixed.",
            "Users are experiencing {keyword}.",
            "The system has a {keyword} that affects functionality.",
            "A {keyword} was reported.",
        ]

        template = rng.choice(templates)
        keyword = rng.choice(keywords) if keywords else "problem"

        return template.format(keyword=keyword)

    def _add_ambiguity(
        self,
        text: str,
        original: str,  # noqa: ARG002 - reserved for future use
        rng: random.Random,
    ) -> str:
        """Add contradictions or ambiguity to the text."""
        ambiguous_additions = [
            " (though it might work sometimes)",
            " or maybe it's something else",
            " - but I'm not sure if that's the real issue",
            " although some say it works fine",
            " unless I'm mistaken about what I saw",
        ]

        # Split into sentences and add ambiguity to some
        sentences = re.split(r"(?<=[.!?])\s+", text)
        result_sentences = []

        for sentence in sentences:
            if sentence.strip():
                # Add ambiguity to ~30% of sentences
                if rng.random() < 0.3 and len(sentence) > 20:
                    sentence = sentence.rstrip(".!?") + rng.choice(ambiguous_additions)
                result_sentences.append(sentence)

        # Maybe add a contradictory statement
        if rng.random() < 0.4:
            contradictions = [
                "Actually, it might be the opposite of what I described.",
                "Or wait, maybe the problem is completely different.",
                "I think I might have misremembered some details.",
            ]
            result_sentences.append(rng.choice(contradictions))

        return " ".join(result_sentences)

    @classmethod
    def from_repo(cls, repo: str) -> DegradationEngine:
        """Create an engine with repo-specific patterns.

        Args:
            repo: Repository name in 'owner/name' format.

        Returns:
            DegradationEngine configured for the repository.

        Example:
            >>> engine = DegradationEngine.from_repo("django/django")
            >>> # Engine now uses Django-specific patterns
        """
        return cls(patterns=DegradationPatterns.for_repo(repo))

    def get_deterministic_seed(
        self, instance_id: str, level: SpecDegradationLevel
    ) -> int:
        """Generate a deterministic seed from instance ID and level.

        Useful for ensuring the same degradation is applied consistently
        across runs for the same task.

        Args:
            instance_id: SWE-bench instance ID.
            level: Degradation level.

        Returns:
            Integer seed derived from inputs.

        Example:
            >>> engine = DegradationEngine()
            >>> seed = engine.get_deterministic_seed(
            ...     "django__django-11099", SpecDegradationLevel.VAGUE
            ... )
            >>> result1 = engine.degrade(issue, SpecDegradationLevel.VAGUE, seed=seed)
            >>> result2 = engine.degrade(issue, SpecDegradationLevel.VAGUE, seed=seed)
            >>> result1.degraded_text == result2.degraded_text
            True
        """
        seed_string = f"{instance_id}:{level.value}"
        hash_bytes = hashlib.sha256(seed_string.encode()).digest()
        return int.from_bytes(hash_bytes[:4], byteorder="big")
