"""Elicitation simulation module for SDD-Bench.

This module provides the oracle system for simulating stakeholder
interactions during requirements elicitation. It includes:

- QuestionScorer: Scores question relevance to hidden requirements
- RequirementsExtractor: Extracts atomic requirements from issue text
- ElicitationOracle: Simulates stakeholder responses

Example:
    >>> from claude_spec_benchmark.elicitation import ElicitationOracle
    >>> oracle = ElicitationOracle.from_issue(issue_text, seed=42)
    >>> response = oracle.ask("What happens when the user clicks submit?")
    >>> print(response.relevance_score)
    0.75
    >>> print(response.revealed_requirements)
    ['REQ-001', 'REQ-003']
"""

from claude_spec_benchmark.elicitation.extraction import RequirementsExtractor
from claude_spec_benchmark.elicitation.oracle import (
    ElicitationOracle,
    OraclePersonality,
)
from claude_spec_benchmark.elicitation.scoring import QuestionScorer, extract_keywords

__all__ = [
    "ElicitationOracle",
    "OraclePersonality",
    "QuestionScorer",
    "RequirementsExtractor",
    "extract_keywords",
]
