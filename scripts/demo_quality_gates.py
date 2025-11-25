"""Demo script for quality gate orchestration components.

This script demonstrates the ThresholdManager and QualityGateEngine functionality.
"""

import asyncio
import sys
from pathlib import Path

# Add parent to path
parent_path = Path(__file__).parent.parent
sys.path.insert(0, str(parent_path))

from src.quality.threshold_manager import ThresholdManager
from src.quality.gate_engine import QualityGateEngine
from src.models.quality_models import (
    QualityDimension,
    QualityThreshold,
    QualityReport,
    QualityStatus,
)


async def demo_threshold_manager():
    """Demonstrate ThresholdManager functionality."""
    print("\n" + "=" * 70)
    print("DEMO: ThresholdManager")
    print("=" * 70)

    # Initialize with defaults
    print("\n1. Initializing ThresholdManager with defaults...")
    manager = ThresholdManager()
    thresholds = manager.get_thresholds()
    print(f"   Loaded {len(thresholds)} global thresholds")

    # Get specific threshold
    print("\n2. Getting coverage threshold...")
    threshold = manager.get_threshold(
        dimension=QualityDimension.COVERAGE,
        metric="line_coverage"
    )
    print(f"   Line Coverage Threshold:")
    print(f"     - Min value: {threshold.min_value}%")
    print(f"     - Target value: {threshold.target_value}%")
    print(f"     - Enforcement: {threshold.enforcement}")
    print(f"     - Allow override: {threshold.allow_override}")

    # Check violations
    print("\n3. Checking threshold violations...")
    test_metrics = {
        "coverage.line_coverage": 75.0,  # Below 80% threshold
        "static.code_quality_score": 7.5,  # Below 8.0 threshold
        "security.critical_vulnerabilities": 0,
        "complexity.cyclomatic_complexity": 12.0,
    }
    violations = manager.check_violations(test_metrics)
    print(f"   Found {len(violations)} violations:")
    for v in violations:
        print(f"     - {v['dimension']}.{v['metric']}: {v['actual_value']} ")
        print(f"       (min: {v['min_value']}, max: {v['max_value']}, "
              f"enforcement: {v['enforcement']})")

    # Load from config file
    print("\n4. Loading from YAML configuration...")
    config_path = Path(__file__).parent.parent / "src" / "config" / "quality_thresholds.yaml"
    if config_path.exists():
        manager_with_config = ThresholdManager(config_path=config_path)
        print(f"   Successfully loaded configuration from {config_path.name}")

        # Demonstrate project overrides
        core_threshold = manager_with_config.get_threshold(
            dimension=QualityDimension.COVERAGE,
            metric="line_coverage",
            project="core-service"
        )
        print(f"\n   Core Service Project Override:")
        print(f"     - Line coverage min: {core_threshold.min_value}% (stricter)")

        # Demonstrate module overrides
        auth_threshold = manager_with_config.get_threshold(
            dimension=QualityDimension.COVERAGE,
            metric="line_coverage",
            module="security/authentication"
        )
        print(f"\n   Security/Authentication Module Override:")
        print(f"     - Line coverage min: {auth_threshold.min_value}% (strictest)")
    else:
        print(f"   Config file not found: {config_path}")

    # Add custom threshold
    print("\n5. Adding custom threshold at runtime...")
    custom_threshold = QualityThreshold(
        dimension=QualityDimension.DOCUMENTATION,
        metric="doc_coverage",
        min_value=80.0,
        target_value=90.0,
        enforcement="warning",
        allow_override=True
    )
    manager.add_threshold(custom_threshold)
    print(f"   Added documentation coverage threshold (min: 80%)")

    # Export configuration
    print("\n6. Exporting configuration...")
    yaml_config = manager.export_configuration(format="yaml")
    print(f"   Exported YAML configuration ({len(yaml_config)} bytes)")
    print(f"\n   First 300 characters:")
    print("   " + "-" * 66)
    for line in yaml_config.split('\n')[:10]:
        print(f"   {line}")
    print("   ...")

    print("\n   ThresholdManager demo completed!")


async def demo_gate_engine():
    """Demonstrate QualityGateEngine functionality."""
    print("\n" + "=" * 70)
    print("DEMO: QualityGateEngine")
    print("=" * 70)

    # Initialize engine
    print("\n1. Initializing QualityGateEngine...")
    manager = ThresholdManager()
    engine = QualityGateEngine(
        threshold_manager=manager,
        enable_mutation_testing=False,  # Disabled for demo
        enable_security_scanning=True,
        enable_performance_analysis=True,
    )
    print("   Engine initialized with all analyzers")

    # Create mock quality report
    print("\n2. Creating mock quality report...")
    mock_report = QualityReport(
        report_id="demo-123",
        status=QualityStatus.FAILED,
        passed=False,
        violations=[
            {
                "dimension": QualityDimension.COVERAGE,
                "metric": "line_coverage",
                "actual_value": 75.0,
                "min_value": 80.0,
                "threshold": manager.get_threshold(
                    QualityDimension.COVERAGE, "line_coverage"
                ),
                "enforcement": "error",
            },
            {
                "dimension": QualityDimension.STATIC,
                "metric": "high_issues",
                "actual_value": 8,
                "max_value": 5,
                "threshold": manager.get_threshold(
                    QualityDimension.STATIC, "high_issues"
                ),
                "enforcement": "warning",
            }
        ]
    )
    print(f"   Created report with {len(mock_report.violations)} violations")

    # Test normal enforcement
    print("\n3. Testing normal quality gate enforcement...")
    result = await engine.enforce_gates(mock_report)
    print(f"   Gate result: {'PASSED' if result['passed'] else 'FAILED'}")
    print(f"   Error violations: {result.get('error_count', 0)}")
    print(f"   Warning violations: {result.get('warning_count', 0)}")
    print(f"   Message: {result['message']}")

    # Test emergency override
    print("\n4. Testing emergency override...")
    print("   Scenario: Critical production issue requiring immediate deployment")
    override_result = await engine.enforce_gates(
        mock_report,
        force=True,
        reason="P0 production incident - customer data access blocked",
        user="ops-lead@example.com"
    )
    print(f"   Override result: {'PASSED' if override_result['passed'] else 'FAILED'}")
    print(f"   Override used: {override_result['override_used']}")
    print(f"   Override user: {override_result.get('override_user')}")
    print(f"   Override reason: {override_result.get('override_reason')}")
    if 'warning' in override_result:
        print(f"   Warning: {override_result['warning']}")

    # Check audit log
    print("\n5. Checking override audit log...")
    audit_log = engine.get_override_log()
    print(f"   Audit log contains {len(audit_log)} entries")
    if audit_log:
        entry = audit_log[-1]
        print(f"\n   Latest override entry:")
        print(f"     - Timestamp: {entry['timestamp']}")
        print(f"     - Report ID: {entry['report_id']}")
        print(f"     - User: {entry['user']}")
        print(f"     - Reason: {entry['reason']}")
        print(f"     - Violations overridden: {entry['violations_overridden']}")

    # Export audit log
    print("\n6. Exporting audit log...")
    log_export = engine.export_override_log(format="text")
    print(f"   Exported audit log ({len(log_export)} bytes)")
    print("\n   Preview:")
    print("   " + "-" * 66)
    for line in log_export.split('\n')[:15]:
        print(f"   {line}")
    if len(log_export.split('\n')) > 15:
        print("   ...")

    print("\n   QualityGateEngine demo completed!")


async def demo_integration():
    """Demonstrate integrated workflow."""
    print("\n" + "=" * 70)
    print("DEMO: Integrated Workflow")
    print("=" * 70)

    print("\n1. Setting up quality gate system...")
    config_path = Path(__file__).parent.parent / "src" / "config" / "quality_thresholds.yaml"
    manager = ThresholdManager(config_path=config_path if config_path.exists() else None)
    engine = QualityGateEngine(threshold_manager=manager)
    print("   System ready")

    print("\n2. Simulating quality check workflow...")
    print("   [Analyzer] Running coverage analysis...")
    print("   [Analyzer] Running static analysis...")
    print("   [Analyzer] Running security scan...")
    print("   [Analyzer] Running complexity analysis...")

    print("\n3. Generating quality report...")
    # In real usage, this would be from actual analysis
    sample_metrics = {
        "coverage.line_coverage": 85.5,
        "coverage.branch_coverage": 78.0,
        "static.code_quality_score": 8.5,
        "static.critical_issues": 0,
        "static.high_issues": 2,
        "security.critical_vulnerabilities": 0,
        "security.high_vulnerabilities": 0,
        "complexity.cyclomatic_complexity": 12.0,
        "complexity.maintainability_index": 82.0,
    }

    violations = manager.check_violations(sample_metrics, project="core-service")
    print(f"   Quality check complete: {len(violations)} violations found")

    if violations:
        print("\n   Violations:")
        for v in violations:
            print(f"     - {v['dimension']}.{v['metric']}: {v['actual_value']}")
            print(f"       Expected: min={v.get('min_value')}, max={v.get('max_value')}")
            print(f"       Enforcement: {v['enforcement']}")

    print("\n4. Determining deployment decision...")
    error_violations = [v for v in violations if v['enforcement'] == 'error']
    if error_violations:
        print(f"   DECISION: Deployment BLOCKED ({len(error_violations)} error violations)")
        print("   Action: Fix issues or request emergency override")
    else:
        print("   DECISION: Deployment APPROVED")
        warning_violations = [v for v in violations if v['enforcement'] == 'warning']
        if warning_violations:
            print(f"   Note: {len(warning_violations)} warnings present")

    print("\n   Integrated workflow demo completed!")


async def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("Quality Gate Orchestration System - Demo")
    print("=" * 70)
    print("\nThis demo showcases the ThresholdManager and QualityGateEngine")
    print("components for enforcing code quality standards.")

    try:
        await demo_threshold_manager()
        await demo_gate_engine()
        await demo_integration()

        print("\n" + "=" * 70)
        print("All demos completed successfully!")
        print("=" * 70)
        print("\nKey Features Demonstrated:")
        print("  - Configurable quality thresholds (YAML/JSON)")
        print("  - Hierarchical overrides (global < project < module)")
        print("  - Parallel analyzer orchestration")
        print("  - Quality gate enforcement with violations")
        print("  - Emergency override with audit trail")
        print("  - Comprehensive quality reporting")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
