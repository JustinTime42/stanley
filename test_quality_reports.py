"""Quick test script for quality report generation.

This script tests the report generation and service layer implementation.
"""

import asyncio
import sys
from datetime import datetime

# Set UTF-8 encoding for console output
if sys.platform == "win32":
    import os
    os.system("chcp 65001 >nul 2>&1")
from src.models.quality_models import (
    QualityReport,
    QualityStatus,
    CoverageType,
    CoverageReport,
    SecurityIssue,
    SeverityLevel,
)
from src.quality.report_generator import ReportGenerator
from src.services.quality_service import QualityService


async def test_report_generation():
    """Test report generation in all formats."""
    print("Testing Quality Report Generation\n" + "=" * 50)

    # Create a sample quality report
    report = QualityReport(
        report_id="TEST-001",
        timestamp=datetime.now(),
        status=QualityStatus.PASSED,
        passed=True,
        coverage_reports={
            CoverageType.LINE: CoverageReport(
                type=CoverageType.LINE,
                percentage=85.5,
                covered=171,
                total=200,
            ),
            CoverageType.BRANCH: CoverageReport(
                type=CoverageType.BRANCH,
                percentage=78.3,
                covered=94,
                total=120,
            ),
        },
        security_score=95.0,
        code_quality_score=88.5,
        maintainability_index=82.0,
        cyclomatic_complexity=8.5,
        cognitive_complexity=12.0,
        trend="improving",
        recommendations=[
            "Increase branch coverage to 80% or higher",
            "Review cyclomatic complexity in module X",
            "Add integration tests for API endpoints",
        ],
    )

    # Add a sample security issue
    report.security_issues.append(
        SecurityIssue(
            issue_id="SEC-001",
            severity=SeverityLevel.MEDIUM,
            confidence="High",
            file_path="src/example.py",
            line_number=42,
            issue_type="Potential SQL Injection",
            description="User input is concatenated directly into SQL query",
            remediation="Use parameterized queries or an ORM",
            cwe_id="CWE-89",
        )
    )

    print(f"\n1. Testing Report Generator")
    print("-" * 50)

    # Test ReportGenerator
    generator = ReportGenerator(output_dir="test_reports")

    # Test HTML report
    print("Generating HTML report...")
    html_path = generator.generate_and_save(report, format="html")
    print(f"[OK] HTML report saved: {html_path}")

    # Test JSON report
    print("Generating JSON report...")
    json_path = generator.generate_and_save(report, format="json")
    print(f"[OK] JSON report saved: {json_path}")

    # Test Markdown report
    print("Generating Markdown report...")
    md_path = generator.generate_and_save(report, format="markdown")
    print(f"[OK] Markdown report saved: {md_path}")

    # Test PR comment
    print("\nGenerating PR comment...")
    pr_comment = generator.generate_pr_comment(report)
    print(f"[OK] PR comment generated ({len(pr_comment)} chars)")
    print("\nPR Comment Preview:")
    print("-" * 50)
    # Remove emojis for Windows console compatibility
    preview = pr_comment[:500] if len(pr_comment) > 500 else pr_comment
    preview_clean = preview.encode('ascii', 'ignore').decode('ascii')
    print(preview_clean + "..." if len(pr_comment) > 500 else preview_clean)

    # Test CI output
    print("\n\nGenerating CI output...")
    ci_output = generator.generate_ci_output(report)
    print(f"[OK] CI output generated")
    print("\nCI Output:")
    print("-" * 50)
    print(ci_output)

    # Test all formats at once
    print("\n\nGenerating all formats at once...")
    all_paths = generator.generate_and_save(report, format="all")
    print("[OK] All formats generated:")
    for fmt, path in all_paths.items():
        print(f"  - {fmt}: {path}")

    print(f"\n2. Testing Quality Service")
    print("-" * 50)

    # Test QualityService
    service = QualityService()

    # Test service stats
    print("Getting service stats...")
    stats = await service.get_service_stats()
    print(f"[OK] Service stats retrieved:")
    print(f"  - Gate Engine: mutation_testing={stats['gate_engine']['mutation_testing_enabled']}")
    print(f"  - Available formats: {stats['report_generator']['available_formats']}")
    print(f"  - Analytics enabled: {stats['analytics_enabled']}")

    # Test PR comment generation through service
    print("\nGenerating PR comment through service...")
    service_pr_comment = await service.generate_pr_comment(report)
    print(f"[OK] Service PR comment generated ({len(service_pr_comment)} chars)")

    # Test CI output through service
    print("Generating CI output through service...")
    service_ci_output = await service.generate_ci_output(report)
    print(f"[OK] Service CI output generated")

    print("\n" + "=" * 50)
    print("All tests completed successfully!")
    print("=" * 50)

    # Display summary
    print("\nSummary:")
    print(f"  - Reports generated in test_reports/")
    print(f"  - Formats tested: HTML, JSON, Markdown")
    print(f"  - Service integration: PASSED")
    print(f"  - PR comment generation: PASSED")
    print(f"  - CI output generation: PASSED")


if __name__ == "__main__":
    asyncio.run(test_report_generation())
