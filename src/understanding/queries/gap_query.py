"""'What don't you know?' query handler."""

import logging
from typing import List, Dict, Optional

from ..knowledge import KnowledgeStore, GapDetector
from ..models import (
    CodebaseUnderstanding,
    KnowledgeGap,
    ConfidenceLevel,
)

logger = logging.getLogger(__name__)


class GapQueryHandler:
    """
    Handle 'What don't you know?' queries.

    PATTERN: List unanalyzed areas and low-confidence knowledge
    """

    def __init__(
        self,
        knowledge_store: KnowledgeStore,
        gap_detector: Optional[GapDetector] = None,
    ):
        """
        Initialize handler.

        Args:
            knowledge_store: Knowledge store to analyze
            gap_detector: Optional gap detector
        """
        self.store = knowledge_store
        self.gap_detector = gap_detector or GapDetector(knowledge_store)

    async def get_all_gaps(self) -> List[KnowledgeGap]:
        """Get all knowledge gaps."""
        return await self.gap_detector.get_gaps()

    async def get_gaps_summary(self) -> Dict:
        """Get summary of knowledge gaps."""
        return self.gap_detector.get_gap_summary()

    async def get_unanalyzed_areas(self) -> Dict:
        """
        Get list of unanalyzed areas.

        Returns:
            Dict with unanalyzed files and areas
        """
        all_symbols = self.store.get_all_symbols()

        # Find files with no symbols (likely not analyzed deeply)
        files_with_symbols = set()
        for symbol in all_symbols.values():
            files_with_symbols.add(symbol.file_path)

        # Get all known files
        all_files = list(self.store._files.keys())
        unanalyzed_files = [f for f in all_files if f not in files_with_symbols]

        return {
            "unanalyzed_files": unanalyzed_files,
            "files_with_symbols": len(files_with_symbols),
            "total_files": len(all_files),
            "coverage": len(files_with_symbols) / len(all_files) if all_files else 0,
        }

    async def get_low_confidence_symbols(
        self,
        threshold: ConfidenceLevel = ConfidenceLevel.UNCERTAIN,
    ) -> List[Dict]:
        """
        Get symbols with low confidence.

        Args:
            threshold: Maximum confidence to include

        Returns:
            List of symbol info dicts
        """
        all_symbols = self.store.get_all_symbols()
        low_confidence = []

        for symbol in all_symbols.values():
            if symbol.confidence <= threshold:
                low_confidence.append({
                    "name": symbol.qualified_name,
                    "kind": symbol.kind.value,
                    "file": symbol.file_path,
                    "line": symbol.line_start,
                    "confidence": symbol.confidence.value,
                    "last_verified": symbol.last_verified.isoformat() if symbol.last_verified else None,
                })

        return low_confidence

    async def get_stale_knowledge(self) -> List[Dict]:
        """Get knowledge that has become stale."""
        return await self.get_low_confidence_symbols(ConfidenceLevel.STALE)

    async def suggest_investigation(self, limit: int = 10) -> List[Dict]:
        """
        Suggest areas to investigate.

        Returns:
            List of suggestions with priority
        """
        return await self.gap_detector.suggest_investigation_priorities(limit)

    async def get_coverage_report(self) -> Dict:
        """
        Get knowledge coverage report.

        Returns:
            Dict with coverage metrics
        """
        all_symbols = self.store.get_all_symbols()

        # Count by confidence
        confidence_counts = {
            "verified": 0,
            "inferred": 0,
            "uncertain": 0,
            "stale": 0,
            "unknown": 0,
        }

        for symbol in all_symbols.values():
            level = symbol.confidence.value
            if level in confidence_counts:
                confidence_counts[level] += 1

        total = len(all_symbols)

        # Calculate percentages
        percentages = {}
        for level, count in confidence_counts.items():
            percentages[f"{level}_pct"] = (count / total * 100) if total > 0 else 0

        # Coverage metrics
        high_confidence = confidence_counts["verified"] + confidence_counts["inferred"]
        low_confidence = confidence_counts["uncertain"] + confidence_counts["stale"]

        return {
            "total_symbols": total,
            "confidence_counts": confidence_counts,
            "confidence_percentages": percentages,
            "high_confidence_count": high_confidence,
            "high_confidence_pct": (high_confidence / total * 100) if total > 0 else 0,
            "low_confidence_count": low_confidence,
            "coverage_score": ((high_confidence - low_confidence) / total * 100) if total > 0 else 0,
        }

    async def explain_gap(self, gap_id: str) -> Optional[Dict]:
        """
        Get detailed explanation of a specific gap.

        Args:
            gap_id: Gap ID to explain

        Returns:
            Gap details or None
        """
        gaps = await self.get_all_gaps()

        for gap in gaps:
            if gap.id == gap_id:
                return {
                    "id": gap.id,
                    "area": gap.area,
                    "description": gap.description,
                    "severity": gap.severity,
                    "related_files": gap.related_files,
                    "related_symbols": gap.related_symbols,
                    "triggered_by": gap.triggered_by,
                    "suggested_actions": gap.suggested_actions,
                    "resolved": gap.resolved,
                }

        return None

    async def query_what_dont_i_know(self) -> str:
        """
        Generate human-readable response to 'What don't you know?'

        Returns:
            Formatted response string
        """
        lines = ["# Knowledge Gaps\n"]

        # Get summary
        summary = await self.get_gaps_summary()
        lines.append(f"**Total gaps**: {summary['total_gaps']}")
        lines.append(f"**High severity**: {summary['by_severity'].get('high', 0)}")
        lines.append(f"**Medium severity**: {summary['by_severity'].get('medium', 0)}")
        lines.append(f"**Low severity**: {summary['by_severity'].get('low', 0)}\n")

        # Get unanalyzed areas
        unanalyzed = await self.get_unanalyzed_areas()
        if unanalyzed["unanalyzed_files"]:
            lines.append(f"## Unanalyzed Files ({len(unanalyzed['unanalyzed_files'])})")
            for f in unanalyzed["unanalyzed_files"][:10]:
                lines.append(f"- `{f}`")
            if len(unanalyzed["unanalyzed_files"]) > 10:
                lines.append(f"... and {len(unanalyzed['unanalyzed_files']) - 10} more\n")
            else:
                lines.append("")

        # Get low confidence
        low_conf = await self.get_low_confidence_symbols()
        if low_conf:
            lines.append(f"## Low Confidence Symbols ({len(low_conf)})")
            for sym in low_conf[:10]:
                lines.append(f"- `{sym['name']}` ({sym['confidence']})")
            if len(low_conf) > 10:
                lines.append(f"... and {len(low_conf) - 10} more\n")
            else:
                lines.append("")

        # Get suggestions
        suggestions = await self.suggest_investigation(5)
        if suggestions:
            lines.append("## Suggested Investigations")
            for i, sug in enumerate(suggestions, 1):
                lines.append(f"\n### {i}. {sug['area']}")
                lines.append(f"*{sug['description']}*")
                lines.append(f"**Severity**: {sug['severity']}")
                if sug.get("actions"):
                    lines.append("**Actions**:")
                    for action in sug["actions"][:3]:
                        lines.append(f"- {action}")

        # Coverage report
        coverage = await self.get_coverage_report()
        lines.append(f"\n## Coverage Summary")
        lines.append(f"- **Total symbols**: {coverage['total_symbols']}")
        lines.append(f"- **High confidence**: {coverage['high_confidence_pct']:.1f}%")
        lines.append(f"- **Coverage score**: {coverage['coverage_score']:.1f}")

        return "\n".join(lines)
