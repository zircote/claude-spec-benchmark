# Changelog

All notable changes to this spec project will be documented here.

## [1.0.0] - 2025-12-16

### Added
- **REQUIREMENTS.md**: Complete product requirements document
  - 8 features (F1-F8) with 30+ functional requirements
  - Non-functional requirements (performance, security, reliability)
  - Risk matrix and mitigations
  - Success metrics and acceptance criteria

- **ARCHITECTURE.md**: Technical architecture design
  - System component diagram
  - 4 new subpackages: degradation/, elicitation/, frameworks/, reporting/
  - Data model extensions for SDD pipeline
  - API design for CLI and programmatic use
  - Integration strategy with existing codebase

- **DECISIONS.md**: 10 Architecture Decision Records
  - ADR-001: Extend existing package vs new package
  - ADR-002: Rules-first oracle vs LLM-first
  - ADR-003: Subpackage organization
  - ADR-004: Local Docker only for v1
  - ADR-005: CLI framework migration (Click → Typer)
  - ADR-006: Elicitation question scoring algorithm
  - ADR-007: Checkpoint/resume strategy
  - ADR-008: Test generation evaluation strategy
  - ADR-009: Framework protocol design philosophy
  - ADR-010: Python version requirement

- **IMPLEMENTATION_PLAN.md**: Phased implementation roadmap
  - 5 phases, 28 tasks
  - Dependency graph
  - Risk mitigation tasks
  - Testing and documentation checklists

### Research Conducted
- Existing codebase analysis: 8 modules, 12 data models, 27 unit tests
- SWE-bench/SWT-bench integration patterns
- Elicitation oracle methodologies
- Dependency analysis for Python 3.14+

### Key Decisions
- Package name: Keep `claude_spec_benchmark` (ADR-001)
- Oracle mode: Rules-first for reproducibility (ADR-002)
- Docker: Local only, no Modal (ADR-004)
- Structure: Subpackages for clear separation (ADR-003)

### Notes
- Project brief defines 8 major features (F1-F8)
- Existing codebase provides foundation for F4/F5
- New modules needed for F1, F2, F3, F6, F7, F8
