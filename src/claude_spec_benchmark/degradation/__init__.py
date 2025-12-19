"""Specification degradation module for SDD-Bench.

This module provides tools for creating progressively vaguer versions
of SWE-bench issues to simulate incomplete requirements.

Example:
    >>> from claude_spec_benchmark.degradation import DegradationEngine
    >>> from claude_spec_benchmark.models import SpecDegradationLevel
    >>> engine = DegradationEngine()
    >>> result = engine.degrade(full_issue, SpecDegradationLevel.VAGUE, seed=42)
    >>> print(result.degraded_text)
"""

from claude_spec_benchmark.degradation.engine import DegradationEngine
from claude_spec_benchmark.degradation.patterns import DegradationPatterns

__all__ = [
    "DegradationEngine",
    "DegradationPatterns",
]
