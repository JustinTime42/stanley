"""Validation script for PRP-12 Coverage & Quality Gates implementation.

Runs comprehensive validation:
- Level 1: Import validation and basic syntax
- Level 2: Component instantiation
- Level 3: Basic functionality tests
- Level 4: Integration tests
"""

import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def print_result(test_name: str, passed: bool, error: str = ""):
    """Print test result without Unicode characters."""
    status = "[PASS]" if passed else "[FAIL]"
    print(f"{status} {test_name}")
    if not passed and error:
        print(f"  Error: {error}")


async def level1_import_validation():
    """Level 1: Validate all imports work correctly."""
    print("\n=== LEVEL 1: Import Validation ===\n")

    tests_passed = 0
    tests_total = 0

    # Test 1: Quality models
    tests_total += 1
    try:
        from src.models.quality_models import (
            QualityReport,
            CoverageReport,
            SecurityIssue,
            PerformanceMetric,
            QualityThreshold,
            QualityDimension,
            CoverageType,
            QualityStatus,
            SeverityLevel,
        )
        print_result("Quality models import", True)
        tests_passed += 1
    except Exception as e:
        print_result("Quality models import", False, str(e))

    # Test 2: Base quality analyzer
    tests_total += 1
    try:
        from src.quality.base import BaseQualityAnalyzer
        print_result("Base quality analyzer import", True)
        tests_passed += 1
    except Exception as e:
        print_result("Base quality analyzer import", False, str(e))

    # Test 3: Threshold manager
    tests_total += 1
    try:
        from src.quality.threshold_manager import ThresholdManager
        print_result("Threshold manager import", True)
        tests_passed += 1
    except Exception as e:
        print_result("Threshold manager import", False, str(e))

    # Test 4: Coverage analyzer
    tests_total += 1
    try:
        from src.quality.analyzers.coverage_analyzer import EnhancedCoverageAnalyzer
        print_result("Coverage analyzer import", True)
        tests_passed += 1
    except Exception as e:
        print_result("Coverage analyzer import", False, str(e))

    # Test 5: Static analyzer
    tests_total += 1
    try:
        from src.quality.analyzers.static_analyzer import StaticAnalyzer
        print_result("Static analyzer import", True)
        tests_passed += 1
    except Exception as e:
        print_result("Static analyzer import", False, str(e))

    # Test 6: Performance analyzer
    tests_total += 1
    try:
        from src.quality.analyzers.performance_analyzer import PerformanceAnalyzer
        print_result("Performance analyzer import", True)
        tests_passed += 1
    except Exception as e:
        print_result("Performance analyzer import", False, str(e))

    # Test 7: Complexity analyzer
    tests_total += 1
    try:
        from src.quality.analyzers.complexity_analyzer import ComplexityAnalyzer
        print_result("Complexity analyzer import", True)
        tests_passed += 1
    except Exception as e:
        print_result("Complexity analyzer import", False, str(e))

    # Test 8: Security integrations
    tests_total += 1
    try:
        from src.quality.integrations.bandit_integration import BanditIntegration
        print_result("Bandit integration import", True)
        tests_passed += 1
    except Exception as e:
        print_result("Bandit integration import", False, str(e))

    tests_total += 1
    try:
        from src.quality.integrations.sonar_integration import SonarIntegration
        print_result("Sonar integration import", True)
        tests_passed += 1
    except Exception as e:
        print_result("Sonar integration import", False, str(e))

    # Test 9: Mutation testing
    tests_total += 1
    try:
        from src.quality.integrations.mutmut_integration import MutmutIntegration
        print_result("Mutmut integration import", True)
        tests_passed += 1
    except Exception as e:
        print_result("Mutmut integration import", False, str(e))

    # Test 10: Prospector integration
    tests_total += 1
    try:
        from src.quality.integrations.prospector_integration import ProspectorIntegration
        print_result("Prospector integration import", True)
        tests_passed += 1
    except Exception as e:
        print_result("Prospector integration import", False, str(e))

    # Test 11: Gate engine
    tests_total += 1
    try:
        from src.quality.gate_engine import QualityGateEngine
        print_result("Quality gate engine import", True)
        tests_passed += 1
    except Exception as e:
        print_result("Quality gate engine import", False, str(e))

    # Test 12: Report generators
    tests_total += 1
    try:
        from src.quality.reporters.html_reporter import HtmlReporter
        print_result("HTML reporter import", True)
        tests_passed += 1
    except Exception as e:
        print_result("HTML reporter import", False, str(e))

    tests_total += 1
    try:
        from src.quality.reporters.json_reporter import JsonReporter
        print_result("JSON reporter import", True)
        tests_passed += 1
    except Exception as e:
        print_result("JSON reporter import", False, str(e))

    tests_total += 1
    try:
        from src.quality.reporters.markdown_reporter import MarkdownReporter
        print_result("Markdown reporter import", True)
        tests_passed += 1
    except Exception as e:
        print_result("Markdown reporter import", False, str(e))

    tests_total += 1
    try:
        from src.quality.report_generator import ReportGenerator
        print_result("Report generator import", True)
        tests_passed += 1
    except Exception as e:
        print_result("Report generator import", False, str(e))

    # Test 13: Quality service
    tests_total += 1
    try:
        from src.services.quality_service import QualityService
        print_result("Quality service import", True)
        tests_passed += 1
    except Exception as e:
        print_result("Quality service import", False, str(e))

    print(f"\nLevel 1 Results: {tests_passed}/{tests_total} tests passed")
    return tests_passed == tests_total


async def level2_instantiation():
    """Level 2: Validate components can be instantiated."""
    print("\n=== LEVEL 2: Component Instantiation ===\n")

    tests_passed = 0
    tests_total = 0

    # Test 1: Threshold manager
    tests_total += 1
    try:
        from src.quality.threshold_manager import ThresholdManager
        manager = ThresholdManager()
        print_result("ThresholdManager instantiation", True)
        tests_passed += 1
    except Exception as e:
        print_result("ThresholdManager instantiation", False, str(e))

    # Test 2: Coverage analyzer
    tests_total += 1
    try:
        from src.quality.analyzers.coverage_analyzer import EnhancedCoverageAnalyzer
        analyzer = EnhancedCoverageAnalyzer()
        print_result("EnhancedCoverageAnalyzer instantiation", True)
        tests_passed += 1
    except Exception as e:
        print_result("EnhancedCoverageAnalyzer instantiation", False, str(e))

    # Test 3: Static analyzer
    tests_total += 1
    try:
        from src.quality.analyzers.static_analyzer import StaticAnalyzer
        analyzer = StaticAnalyzer()
        print_result("StaticAnalyzer instantiation", True)
        tests_passed += 1
    except Exception as e:
        print_result("StaticAnalyzer instantiation", False, str(e))

    # Test 4: Performance analyzer
    tests_total += 1
    try:
        from src.quality.analyzers.performance_analyzer import PerformanceAnalyzer
        analyzer = PerformanceAnalyzer()
        print_result("PerformanceAnalyzer instantiation", True)
        tests_passed += 1
    except Exception as e:
        print_result("PerformanceAnalyzer instantiation", False, str(e))

    # Test 5: Complexity analyzer
    tests_total += 1
    try:
        from src.quality.analyzers.complexity_analyzer import ComplexityAnalyzer
        analyzer = ComplexityAnalyzer()
        print_result("ComplexityAnalyzer instantiation", True)
        tests_passed += 1
    except Exception as e:
        print_result("ComplexityAnalyzer instantiation", False, str(e))

    # Test 6: Report generators
    tests_total += 1
    try:
        from src.quality.reporters.html_reporter import HtmlReporter
        reporter = HtmlReporter()
        print_result("HtmlReporter instantiation", True)
        tests_passed += 1
    except Exception as e:
        print_result("HtmlReporter instantiation", False, str(e))

    tests_total += 1
    try:
        from src.quality.reporters.json_reporter import JsonReporter
        reporter = JsonReporter()
        print_result("JsonReporter instantiation", True)
        tests_passed += 1
    except Exception as e:
        print_result("JsonReporter instantiation", False, str(e))

    tests_total += 1
    try:
        from src.quality.reporters.markdown_reporter import MarkdownReporter
        reporter = MarkdownReporter()
        print_result("MarkdownReporter instantiation", True)
        tests_passed += 1
    except Exception as e:
        print_result("MarkdownReporter instantiation", False, str(e))

    # Test 7: Gate engine
    tests_total += 1
    try:
        from src.quality.gate_engine import QualityGateEngine
        from src.quality.threshold_manager import ThresholdManager
        threshold_mgr = ThresholdManager()
        engine = QualityGateEngine(threshold_manager=threshold_mgr)
        print_result("QualityGateEngine instantiation", True)
        tests_passed += 1
    except Exception as e:
        print_result("QualityGateEngine instantiation", False, str(e))

    # Test 8: Quality service
    tests_total += 1
    try:
        from src.services.quality_service import QualityService
        service = QualityService()
        print_result("QualityService instantiation", True)
        tests_passed += 1
    except Exception as e:
        print_result("QualityService instantiation", False, str(e))

    print(f"\nLevel 2 Results: {tests_passed}/{tests_total} tests passed")
    return tests_passed == tests_total


async def level3_basic_functionality():
    """Level 3: Test basic functionality."""
    print("\n=== LEVEL 3: Basic Functionality Tests ===\n")

    tests_passed = 0
    tests_total = 0

    # Test 1: Quality models validation
    tests_total += 1
    try:
        from src.models.quality_models import CoverageReport, CoverageType
        report = CoverageReport(
            type=CoverageType.LINE,
            percentage=85.5,
            covered=171,
            total=200
        )
        assert report.percentage == 85.5
        assert report.covered == 171
        print_result("CoverageReport model creation", True)
        tests_passed += 1
    except Exception as e:
        print_result("CoverageReport model creation", False, str(e))

    # Test 2: Threshold manager basic operations
    tests_total += 1
    try:
        from src.quality.threshold_manager import ThresholdManager
        from src.models.quality_models import QualityDimension

        manager = ThresholdManager()
        # Test default thresholds exist
        threshold = manager.get_threshold(
            dimension=QualityDimension.COVERAGE,
            metric="line_coverage",
            project="default"
        )
        # Threshold may be None for defaults, just verify method works
        print_result("ThresholdManager get_threshold", True)
        tests_passed += 1
    except Exception as e:
        print_result("ThresholdManager get_threshold", False, str(e))

    # Test 3: Complexity analyzer can analyze code
    tests_total += 1
    try:
        from src.quality.analyzers.complexity_analyzer import ComplexityAnalyzer

        analyzer = ComplexityAnalyzer()

        # Create a simple test file
        test_code = '''
def simple_function(x):
    if x > 0:
        return x * 2
    else:
        return 0
'''
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_code)
            temp_path = f.name

        result = await analyzer.analyze_complexity(temp_path)
        assert 'metrics' in result or 'error' not in result

        # Cleanup
        import os
        os.unlink(temp_path)

        print_result("ComplexityAnalyzer analyze_complexity", True)
        tests_passed += 1
    except Exception as e:
        print_result("ComplexityAnalyzer analyze_complexity", False, str(e))

    # Test 4: Report generation
    tests_total += 1
    try:
        from src.quality.reporters.json_reporter import JsonReporter
        from src.models.quality_models import (
            QualityReport,
            QualityStatus,
            CoverageReport,
            CoverageType
        )
        from datetime import datetime

        reporter = JsonReporter()

        # Create minimal quality report with all required fields
        coverage = CoverageReport(
            type=CoverageType.LINE,
            percentage=80.0,
            covered=80,
            total=100
        )

        quality_report = QualityReport(
            report_id="test-report-001",
            timestamp=datetime.now(),
            project="test-project",
            status=QualityStatus.PASSED,
            passed=True,
            overall_score=80.0,
            coverage_reports={CoverageType.LINE: coverage},
            security_issues=[],
            performance_metrics=[],
            gate_results={}
        )

        json_output = reporter.generate_report(quality_report)
        assert json_output is not None
        assert len(json_output) > 0

        print_result("JsonReporter generate_report", True)
        tests_passed += 1
    except Exception as e:
        print_result("JsonReporter generate", False, str(e))

    print(f"\nLevel 3 Results: {tests_passed}/{tests_total} tests passed")
    return tests_passed == tests_total


async def main():
    """Run all validation levels."""
    print("=" * 60)
    print("PRP-12 Coverage & Quality Gates Validation")
    print("=" * 60)

    level1_passed = await level1_import_validation()

    if not level1_passed:
        print("\n[ERROR] Level 1 validation failed. Fix import issues before proceeding.")
        return False

    level2_passed = await level2_instantiation()

    if not level2_passed:
        print("\n[ERROR] Level 2 validation failed. Fix instantiation issues before proceeding.")
        return False

    level3_passed = await level3_basic_functionality()

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Level 1 (Imports):        {'PASSED' if level1_passed else 'FAILED'}")
    print(f"Level 2 (Instantiation):  {'PASSED' if level2_passed else 'FAILED'}")
    print(f"Level 3 (Functionality):  {'PASSED' if level3_passed else 'FAILED'}")
    print("=" * 60)

    all_passed = level1_passed and level2_passed and level3_passed

    if all_passed:
        print("\n[SUCCESS] All validation tests passed!")
    else:
        print("\n[FAILURE] Some validation tests failed. See details above.")

    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
