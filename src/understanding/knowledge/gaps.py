"""Knowledge gap detection and tracking."""

import logging
from typing import List, Dict, Optional, Set
from datetime import datetime

from .store import KnowledgeStore
from ..models import (
    KnowledgeGap,
    Symbol,
    FileInfo,
    ConfidenceLevel,
    CodebaseUnderstanding,
)

logger = logging.getLogger(__name__)


class GapDetector:
    """
    Detect and track knowledge gaps.

    PATTERN: Track what areas haven't been analyzed
    CRITICAL: Identify when queries hit unknown areas
    """

    def __init__(
        self,
        knowledge_store: KnowledgeStore,
    ):
        """
        Initialize gap detector.

        Args:
            knowledge_store: Knowledge store to analyze
        """
        self.store = knowledge_store
        self._gaps: Dict[str, KnowledgeGap] = {}
        self._query_history: List[str] = []

    async def detect_gaps(
        self,
        understanding: CodebaseUnderstanding,
    ) -> List[KnowledgeGap]:
        """
        Detect knowledge gaps in understanding.

        Args:
            understanding: Current codebase understanding

        Returns:
            List of detected gaps
        """
        gaps = []

        # Gap 1: Unanalyzed files
        if understanding.unanalyzed_files:
            gap = KnowledgeGap(
                id=KnowledgeGap.generate_id("files", "unanalyzed_files"),
                area="files",
                description=f"{len(understanding.unanalyzed_files)} files not analyzed",
                severity="high" if len(understanding.unanalyzed_files) > 50 else "medium",
                related_files=understanding.unanalyzed_files[:20],
                suggested_actions=[
                    "Run deep analysis to analyze remaining files",
                    f"Files include: {', '.join(understanding.unanalyzed_files[:5])}",
                ],
            )
            gaps.append(gap)

        # Gap 2: Low confidence symbols
        low_confidence_symbols = [
            s for s in understanding.symbols.values()
            if s.confidence in (ConfidenceLevel.UNCERTAIN, ConfidenceLevel.STALE)
        ]
        if low_confidence_symbols:
            gap = KnowledgeGap(
                id=KnowledgeGap.generate_id("symbols", "low_confidence"),
                area="symbols",
                description=f"{len(low_confidence_symbols)} symbols with low confidence",
                severity="medium",
                related_symbols=[s.id for s in low_confidence_symbols[:10]],
                suggested_actions=[
                    "Re-analyze files with stale symbols",
                    "Refresh analysis for uncertain symbols",
                ],
            )
            gaps.append(gap)

        # Gap 3: Analysis errors
        if understanding.analysis_errors:
            gap = KnowledgeGap(
                id=KnowledgeGap.generate_id("errors", "analysis_errors"),
                area="analysis",
                description=f"{len(understanding.analysis_errors)} files had errors",
                severity="high",
                related_files=list(understanding.analysis_errors.keys())[:10],
                suggested_actions=[
                    "Review error logs",
                    "Check file encoding/syntax",
                ],
            )
            gaps.append(gap)

        # Gap 4: Missing dependency information
        missing_deps = self._find_missing_dependencies(understanding)
        if missing_deps:
            gap = KnowledgeGap(
                id=KnowledgeGap.generate_id("dependencies", "missing_deps"),
                area="dependencies",
                description=f"{len(missing_deps)} modules have unknown dependencies",
                severity="medium",
                suggested_actions=[
                    "Analyze external dependencies",
                    f"Unknown: {', '.join(list(missing_deps)[:5])}",
                ],
            )
            gaps.append(gap)

        # Gap 5: Undocumented symbols
        undocumented = [
            s for s in understanding.symbols.values()
            if s.kind in (SymbolKind.FUNCTION, SymbolKind.CLASS, SymbolKind.METHOD)
            and not s.docstring
        ]
        if undocumented:
            gap = KnowledgeGap(
                id=KnowledgeGap.generate_id("documentation", "undocumented"),
                area="documentation",
                description=f"{len(undocumented)} symbols lack documentation",
                severity="low",
                related_symbols=[s.id for s in undocumented[:10]],
                suggested_actions=[
                    "Consider adding docstrings",
                    "Use AI to generate descriptions",
                ],
            )
            gaps.append(gap)

        # Store gaps
        for gap in gaps:
            self._gaps[gap.id] = gap

        return gaps

    def _find_missing_dependencies(
        self,
        understanding: CodebaseUnderstanding,
    ) -> Set[str]:
        """Find external dependencies not in project."""
        known_modules = set(understanding.dependency_graph.nodes.keys())
        all_imports = set()

        for edge in understanding.dependency_graph.edges:
            all_imports.add(edge["to"])

        return all_imports - known_modules

    async def track_query_gap(
        self,
        query: str,
        found_results: bool,
        confidence: ConfidenceLevel,
    ) -> Optional[KnowledgeGap]:
        """
        Track when a query exposes a gap.

        Args:
            query: The query that was made
            found_results: Whether results were found
            confidence: Confidence in results

        Returns:
            New gap if detected
        """
        self._query_history.append(query)

        if not found_results or confidence < ConfidenceLevel.INFERRED:
            # This query exposed a gap
            gap = KnowledgeGap(
                id=KnowledgeGap.generate_id("query", query),
                area="query_coverage",
                description=f"Query '{query[:50]}...' had insufficient results",
                severity="low",
                triggered_by=query,
                suggested_actions=[
                    "Analyze related code areas",
                    "Check if topic is covered in codebase",
                ],
            )
            self._gaps[gap.id] = gap
            return gap

        return None

    async def get_gaps(
        self,
        severity: Optional[str] = None,
        area: Optional[str] = None,
    ) -> List[KnowledgeGap]:
        """
        Get current knowledge gaps.

        Args:
            severity: Filter by severity
            area: Filter by area

        Returns:
            List of gaps
        """
        gaps = list(self._gaps.values())

        if severity:
            gaps = [g for g in gaps if g.severity == severity]

        if area:
            gaps = [g for g in gaps if g.area == area]

        # Sort by severity
        severity_order = {"high": 0, "medium": 1, "low": 2}
        gaps.sort(key=lambda g: severity_order.get(g.severity, 3))

        return gaps

    async def resolve_gap(self, gap_id: str) -> bool:
        """Mark a gap as resolved."""
        if gap_id in self._gaps:
            self._gaps[gap_id].resolved = True
            self._gaps[gap_id].resolved_at = datetime.now()
            return True
        return False

    async def suggest_investigation_priorities(
        self,
        limit: int = 5,
    ) -> List[Dict]:
        """
        Suggest areas to investigate based on gaps.

        Returns:
            List of suggestions with priority
        """
        gaps = await self.get_gaps()
        suggestions = []

        for gap in gaps[:limit]:
            suggestion = {
                "area": gap.area,
                "description": gap.description,
                "severity": gap.severity,
                "actions": gap.suggested_actions,
            }

            if gap.related_files:
                suggestion["files_to_check"] = gap.related_files[:3]

            suggestions.append(suggestion)

        return suggestions

    def get_gap_summary(self) -> Dict:
        """Get summary of all gaps."""
        gaps = list(self._gaps.values())

        return {
            "total_gaps": len(gaps),
            "by_severity": {
                "high": len([g for g in gaps if g.severity == "high"]),
                "medium": len([g for g in gaps if g.severity == "medium"]),
                "low": len([g for g in gaps if g.severity == "low"]),
            },
            "by_area": {
                area: len([g for g in gaps if g.area == area])
                for area in set(g.area for g in gaps)
            },
            "resolved": len([g for g in gaps if g.resolved]),
            "unresolved": len([g for g in gaps if not g.resolved]),
        }


# Import SymbolKind for use in detect_gaps
from ..models import SymbolKind
