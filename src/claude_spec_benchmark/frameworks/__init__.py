"""SDD framework protocol and implementations for SDD-Bench.

This module defines the protocol that SDD frameworks must implement
to be evaluated, plus reference implementations.

Example:
    >>> from claude_spec_benchmark.frameworks import SDDFrameworkProtocol
    >>> from claude_spec_benchmark.frameworks import PassthroughFramework
    >>> framework = PassthroughFramework()
    >>> isinstance(framework, SDDFrameworkProtocol)
    True
"""

from claude_spec_benchmark.frameworks.claude_code import ClaudeCodeFramework
from claude_spec_benchmark.frameworks.passthrough import PassthroughFramework
from claude_spec_benchmark.frameworks.protocol import SDDFrameworkProtocol

__all__ = [
    "ClaudeCodeFramework",
    "PassthroughFramework",
    "SDDFrameworkProtocol",
]
