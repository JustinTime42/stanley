"""Tests for self-healing test system components.

This package contains comprehensive unit tests for:
- Failure analysis (FailureAnalyzer and all analyzer strategies)
- Test repair (TestRepairer and all repair strategies)
- Flaky test detection (FlakyDetector with statistical analysis)
- Test optimization (TestOptimizer with performance suggestions)

All tests follow PRP-11 specifications with:
- pytest-asyncio for async tests
- Comprehensive fixtures and mocking
- Edge case and error handling coverage
- 85%+ code coverage target
"""
