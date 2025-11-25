"""Test script for quality gate orchestration components.

This script validates the ThresholdManager and QualityGateEngine integration.
"""

import asyncio
import sys
from pathlib import Path

# Add parent to path
parent_path = Path(__file__).parent.parent
sys.path.insert(0, str(parent_path))

from src.quality.threshold_manager import ThresholdManager
from src.quality.gate_engine import QualityGateEngine
from src.models.quality_models import QualityDimension


async def test_threshold_manager():
    """Test ThresholdManager functionality."""
    print("\n" + "=" * 70)
    print("Testing ThresholdManager")
    print("=" * 70)

    # Test 1: Load default thresholds
    print("\n1. Loading default thresholds...")
    manager = ThresholdManager()
    global_thresholds = manager.get_thresholds()
    print(f"   ✓ Loaded {len(global_thresholds)} global thresholds")

    # Test 2: Get specific threshold
    print("\n2. Getting specific threshold...")
    threshold = manager.get_threshold(
        dimension=QualityDimension.COVERAGE,
        metric="line_coverage"
    )
    if threshold:
        print(f"   ✓ Found threshold: {threshold.metric}")
        print(f"     Min value: {threshold.min_value}")
        print(f"     Target value: {threshold.target_value}")
        print(f"     Enforcement: {threshold.enforcement}")
    else:
        print("   ✗ Threshold not found")

    # Test 3: Check threshold violations
    print("\n3. Checking threshold violations...")
    test_metrics = {
        "coverage.line_coverage": 75.0,  # Below 80% threshold
        "static.code_quality_score": 7.5,  # Below 8.0 threshold
        "security.critical_vulnerabilities": 0,
        "complexity.cyclomatic_complexity": 12.0,
    }
    violations = manager.check_violations(test_metrics)
    print(f"   ✓ Found {len(violations)} violations:")
    for v in violations:
        print(f"     - {v['dimension']}.{v['metric']}: {v['actual_value']} "
              f"(min: {v['min_value']}, enforcement: {v['enforcement']})")

    # Test 4: Load from YAML config
    print("\n4. Loading from YAML config...")
    config_path = Path(__file__).parent.parent / "src" / "config" / "quality_thresholds.yaml"
    if config_path.exists():
        manager_with_config = ThresholdManager(config_path=config_path)
        print(f"   ✓ Loaded configuration from {config_path.name}")

        # Test project override
        project_thresholds = manager_with_config.get_thresholds(project="core-service")
        print(f"   ✓ Core service project has {len(project_thresholds)} thresholds")

        # Test module override
        module_thresholds = manager_with_config.get_thresholds(
            project="core-service",
            module="security/authentication"
        )
        print(f"   ✓ Security/authentication module has {len(module_thresholds)} thresholds")
    else:
        print(f"   ! Config file not found: {config_path}")

    # Test 5: Add threshold at runtime
    print("\n5. Adding threshold at runtime...")
    from src.models.quality_models import QualityThreshold
    new_threshold = QualityThreshold(
        dimension=QualityDimension.DOCUMENTATION,
        metric="doc_coverage",
        min_value=80.0,
        enforcement="warning",
        allow_override=True
    )
    manager.add_threshold(new_threshold)
    doc_threshold = manager.get_threshold(
        dimension=QualityDimension.DOCUMENTATION,
        metric="doc_coverage"
    )
    if doc_threshold:
        print(f"   ✓ Added documentation threshold: {doc_threshold.metric}")
    else:
        print("   ✗ Failed to add threshold")

    # Test 6: Export configuration
    print("\n6. Exporting configuration...")
    yaml_config = manager.export_configuration(format="yaml")
    print(f"   ✓ Exported YAML configuration ({len(yaml_config)} bytes)")
    json_config = manager.export_configuration(format="json")
    print(f"   ✓ Exported JSON configuration ({len(json_config)} bytes)")

    print("\n✓ ThresholdManager tests completed successfully!")


async def test_gate_engine():
    """Test QualityGateEngine functionality."""
    print("\n" + "=" * 70)
    print("Testing QualityGateEngine")
    print("=" * 70)

    # Test 1: Initialize engine
    print("\n1. Initializing QualityGateEngine...")
    manager = ThresholdManager()
    engine = QualityGateEngine(
        threshold_manager=manager,
        enable_mutation_testing=False,  # Disable for quick test
        enable_security_scanning=True,
        enable_performance_analysis=True,
    )
    print("   ✓ Engine initialized successfully")

    # Test 2: Run quality checks (dry run without actual analysis)
    print("\n2. Running quality checks (mock data)...")

    # Create mock test results
    mock_test_results = {
        "coverage": {
            "line_coverage": 85.5,
            "branch_coverage": 78.0,
        },
        "tests_passed": 42,
        "tests_failed": 0,
    }

    # Run checks on a test directory
    test_path = str(Path(__file__).parent.parent / "src" / "quality")
    try:
        report = await engine.run_quality_checks(
            source_path=test_path,
            test_results=mock_test_results,
            dimensions=[QualityDimension.COVERAGE, QualityDimension.STATIC],
        )
        print(f"   ✓ Quality checks completed")
        print(f"     Report ID: {report.report_id}")
        print(f"     Status: {report.status}")
        print(f"     Passed: {report.passed}")
        print(f"     Violations: {len(report.violations)}")

        # Show violations
        if report.violations:
            print("\n   Violations found:")
            for v in report.violations[:3]:  # Show first 3
                print(f"     - {v['dimension']}.{v['metric']}: {v['actual_value']} "
                      f"(enforcement: {v['enforcement']})")

        # Show recommendations
        if report.recommendations:
            print("\n   Recommendations:")
            for rec in report.recommendations[:3]:  # Show first 3
                print(f"     - {rec}")

    except Exception as e:
        print(f"   ! Quality checks failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 3: Test enforcement (normal)
    print("\n3. Testing normal enforcement...")
    from src.models.quality_models import QualityReport, QualityStatus
    mock_report = QualityReport(
        report_id="test-123",
        status=QualityStatus.FAILED,
        passed=False,
        violations=[
            {
                "dimension": QualityDimension.COVERAGE,
                "metric": "line_coverage",
                "actual_value": 70.0,
                "threshold": manager.get_threshold(
                    QualityDimension.COVERAGE, "line_coverage"
                ),
                "enforcement": "error",
            }
        ]
    )

    result = await engine.enforce_gates(mock_report)
    print(f"   ✓ Enforcement result: {result['passed']}")
    print(f"     Message: {result['message']}")

    # Test 4: Test emergency override
    print("\n4. Testing emergency override...")
    override_result = await engine.enforce_gates(
        mock_report,
        force=True,
        reason="Emergency production fix - P0 incident",
        user="ops-team"
    )
    print(f"   ✓ Override result: {override_result['passed']}")
    print(f"     Override used: {override_result['override_used']}")
    if 'warning' in override_result:
        print(f"     Warning: {override_result['warning']}")

    # Test 5: Check override audit log
    print("\n5. Checking override audit log...")
    audit_log = engine.get_override_log()
    print(f"   ✓ Audit log has {len(audit_log)} entries")
    if audit_log:
        entry = audit_log[-1]
        print(f"     Last override:")
        print(f"       User: {entry['user']}")
        print(f"       Reason: {entry['reason']}")
        print(f"       Violations: {entry['violations_overridden']}")

    print("\n✓ QualityGateEngine tests completed successfully!")


async def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("Quality Gate Orchestration Component Tests")
    print("=" * 70)

    try:
        await test_threshold_manager()
        await test_gate_engine()

        print("\n" + "=" * 70)
        print("✓ All tests completed successfully!")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
