"""Understanding mixin for agents."""

import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class UnderstandingMixin:
    """
    Mixin to add codebase understanding capabilities to agents.

    PATTERN: Add understanding hooks to any agent
    CRITICAL: verify_before_claim() for anti-hallucination
    """

    _understanding_service: Optional[Any] = None

    def set_understanding_service(self, service: Any) -> None:
        """
        Set the understanding service.

        Args:
            service: UnderstandingService instance
        """
        self._understanding_service = service

    @property
    def has_understanding(self) -> bool:
        """Check if understanding service is available and analyzed."""
        return (
            self._understanding_service is not None
            and self._understanding_service.is_analyzed
        )

    async def verify_before_claim(self, claim: str) -> str:
        """
        Verify a claim about the codebase before making it.

        CRITICAL: Anti-hallucination hook
        Use this before making any statements about the codebase.

        Args:
            claim: Claim to verify

        Returns:
            Original claim or modified response with uncertainty
        """
        if not self._understanding_service:
            return claim

        try:
            return await self._understanding_service.verify_before_claim(claim)
        except Exception as e:
            logger.warning(f"Verification failed: {e}")
            return claim

    async def check_for_duplicates(
        self,
        proposed_name: str,
        proposed_description: str,
        proposed_kind: str = "function",
        proposed_signature: Optional[str] = None,
    ) -> List[Dict]:
        """
        Check for potential duplicate functionality before creating code.

        CRITICAL: Call this before creating any new function/class

        Args:
            proposed_name: Name of proposed symbol
            proposed_description: What it should do
            proposed_kind: function, class, method
            proposed_signature: Optional signature

        Returns:
            List of potential duplicates with recommendations
        """
        if not self._understanding_service:
            return []

        try:
            candidates = await self._understanding_service.check_for_duplicates(
                proposed_name=proposed_name,
                proposed_description=proposed_description,
                proposed_kind=proposed_kind,
                proposed_signature=proposed_signature,
            )

            return [
                {
                    "existing_symbol": c.existing_symbol,
                    "similarity": c.similarity_score,
                    "type": c.similarity_type,
                    "recommendation": c.recommendation,
                    "location": c.existing_location,
                    "description": c.existing_description,
                }
                for c in candidates
            ]
        except Exception as e:
            logger.warning(f"Duplicate check failed: {e}")
            return []

    async def get_relevant_context(
        self,
        query: str,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """
        Get relevant codebase context for task planning.

        Args:
            query: What context to retrieve
            limit: Maximum results

        Returns:
            Dict with relevant symbols, files, and documentation
        """
        if not self._understanding_service:
            return {"symbols": [], "files": [], "has_understanding": False}

        try:
            # Query for relevant code
            response = await self._understanding_service.query(query)

            # Find similar code
            similar = await self._understanding_service.find_similar(
                description=query,
                limit=limit,
            )

            return {
                "has_understanding": True,
                "answer": response.answer,
                "confidence": response.confidence.value,
                "sources": response.sources,
                "similar_code": similar,
                "knowledge_gaps": response.knowledge_gaps,
            }
        except Exception as e:
            logger.warning(f"Context retrieval failed: {e}")
            return {"symbols": [], "files": [], "has_understanding": False, "error": str(e)}

    async def flag_uncertainty(
        self,
        area: str,
        description: str,
    ) -> None:
        """
        Flag an area of uncertainty.

        Use when agent is unsure about something.

        Args:
            area: Code area with uncertainty
            description: What is uncertain
        """
        if not self._understanding_service:
            return

        try:
            # Record the uncertainty as a knowledge gap
            from ...understanding.models import KnowledgeGap

            gap = KnowledgeGap(
                id=KnowledgeGap.generate_id(area, description),
                area=area,
                description=description,
                severity="medium",
                triggered_by="agent_uncertainty",
                suggested_actions=["Investigate this area", "Ask user for clarification"],
            )

            # Add to gap detector
            self._understanding_service.gap_detector._gaps[gap.id] = gap
            logger.info(f"Flagged uncertainty: {area} - {description}")

        except Exception as e:
            logger.warning(f"Failed to flag uncertainty: {e}")

    async def get_symbol_info(self, symbol_name: str) -> Optional[str]:
        """
        Get verified information about a symbol.

        Args:
            symbol_name: Symbol to look up

        Returns:
            Information string or None
        """
        if not self._understanding_service:
            return None

        try:
            return await self._understanding_service.get_symbol_info(symbol_name)
        except Exception as e:
            logger.warning(f"Symbol lookup failed: {e}")
            return None

    async def verify_symbol_exists(
        self,
        symbol_name: str,
        expected_kind: Optional[str] = None,
    ) -> Dict:
        """
        Verify a symbol exists in the codebase.

        Args:
            symbol_name: Symbol to verify
            expected_kind: Expected kind (function, class, etc.)

        Returns:
            Verification result dict
        """
        if not self._understanding_service:
            return {"verified": False, "error": "No understanding service"}

        try:
            result = await self._understanding_service.verify_symbol_exists(
                symbol_name, expected_kind
            )
            return {
                "verified": result.verified,
                "confidence": result.confidence.value,
                "supporting_evidence": result.supporting_evidence,
                "correction": result.correction,
            }
        except Exception as e:
            return {"verified": False, "error": str(e)}

    def get_analysis_context(self) -> Dict[str, Any]:
        """
        Get summary of codebase understanding for context.

        Returns:
            Dict with codebase summary
        """
        if not self._understanding_service:
            return {"analyzed": False}

        try:
            stats = self._understanding_service.get_statistics()
            return {
                "analyzed": stats.get("analyzed", False),
                "project_name": stats.get("project_name"),
                "detected_type": stats.get("detected_type"),
                "framework": stats.get("framework"),
                "total_files": stats.get("total_files", 0),
                "total_symbols": stats.get("total_symbols", 0),
                "total_lines": stats.get("total_lines", 0),
            }
        except Exception as e:
            logger.warning(f"Failed to get analysis context: {e}")
            return {"analyzed": False, "error": str(e)}


class UnderstandingAwareAgent(UnderstandingMixin):
    """
    Base class for agents that need understanding capabilities.

    Inherit from this along with BaseAgent to get understanding features.
    """

    async def pre_execute_checks(self) -> Dict[str, Any]:
        """
        Run pre-execution checks using understanding.

        Call this at start of execute() method.

        Returns:
            Dict with context and any warnings
        """
        result = {
            "has_understanding": self.has_understanding,
            "warnings": [],
        }

        if self.has_understanding:
            result["codebase"] = self.get_analysis_context()

            # Check for knowledge gaps
            if self._understanding_service:
                gaps = await self._understanding_service.get_knowledge_gaps()
                high_severity = [g for g in gaps if g.get("severity") == "high"]
                if high_severity:
                    result["warnings"].append(
                        f"Warning: {len(high_severity)} high-severity knowledge gaps"
                    )

        return result
