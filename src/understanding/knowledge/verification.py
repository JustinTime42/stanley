"""Knowledge verification system - Anti-hallucination."""

import logging
import re
from typing import List, Optional, Tuple

from .store import KnowledgeStore
from .confidence import ConfidenceScorer
from ..models import VerificationResult, ConfidenceLevel, Symbol

logger = logging.getLogger(__name__)


class KnowledgeVerifier:
    """
    Verify claims about the codebase before agents make them.

    CRITICAL: This is the anti-hallucination system
    PATTERN: Check symbol table and knowledge store before any claim
    """

    def __init__(
        self,
        knowledge_store: KnowledgeStore,
        confidence_scorer: Optional[ConfidenceScorer] = None,
    ):
        """
        Initialize verifier.

        Args:
            knowledge_store: Knowledge store to verify against
            confidence_scorer: Optional confidence scorer
        """
        self.store = knowledge_store
        self.scorer = confidence_scorer or ConfidenceScorer()

    async def verify_symbol_exists(
        self,
        symbol_name: str,
        expected_kind: Optional[str] = None,
    ) -> VerificationResult:
        """
        Verify a symbol (function, class, etc.) exists.

        Args:
            symbol_name: Name or qualified name of symbol
            expected_kind: Optional expected kind (function, class, etc.)

        Returns:
            VerificationResult with confidence
        """
        # Try exact match first
        symbol = await self.store.get_symbol(symbol_name)

        if symbol:
            # Check kind if specified
            if expected_kind and symbol.kind.value != expected_kind:
                return VerificationResult(
                    claim=f"{symbol_name} is a {expected_kind}",
                    verified=False,
                    confidence=ConfidenceLevel.VERIFIED,
                    contradicting_evidence=[
                        f"{symbol_name} is actually a {symbol.kind.value}"
                    ],
                    correction=f"{symbol_name} is a {symbol.kind.value}, not a {expected_kind}",
                )

            return VerificationResult(
                claim=f"{symbol_name} exists",
                verified=True,
                confidence=symbol.confidence,
                supporting_evidence=[
                    f"Found in {symbol.file_path}:{symbol.line_start}"
                ],
            )

        # Try fuzzy match
        similar = await self.store.find_similar_symbols(symbol_name, limit=3)

        if similar:
            return VerificationResult(
                claim=f"{symbol_name} exists",
                verified=False,
                confidence=ConfidenceLevel.VERIFIED,
                contradicting_evidence=[f"Symbol '{symbol_name}' not found"],
                correction=f"Did you mean: {', '.join(s.name for s in similar)}?",
            )

        return VerificationResult(
            claim=f"{symbol_name} exists",
            verified=False,
            confidence=ConfidenceLevel.UNKNOWN,
            contradicting_evidence=[
                f"Symbol '{symbol_name}' not found in analyzed codebase"
            ],
        )

    async def verify_function_signature(
        self,
        function_name: str,
        expected_params: List[str],
    ) -> VerificationResult:
        """
        Verify a function has the expected parameters.

        Args:
            function_name: Function name
            expected_params: Expected parameter names

        Returns:
            VerificationResult
        """
        symbol = await self.store.get_symbol(function_name)

        if not symbol:
            return VerificationResult(
                claim=f"{function_name} has params {expected_params}",
                verified=False,
                confidence=ConfidenceLevel.UNKNOWN,
                contradicting_evidence=[f"Function '{function_name}' not found"],
            )

        actual_params = [p["name"] for p in symbol.parameters]

        if set(expected_params) == set(actual_params):
            return VerificationResult(
                claim=f"{function_name} has params {expected_params}",
                verified=True,
                confidence=symbol.confidence,
                supporting_evidence=[f"Signature: {symbol.signature}"],
            )

        return VerificationResult(
            claim=f"{function_name} has params {expected_params}",
            verified=False,
            confidence=symbol.confidence,
            contradicting_evidence=[f"Actual params: {actual_params}"],
            correction=f"{function_name} actually has parameters: {actual_params}",
        )

    async def verify_file_exists(self, file_path: str) -> VerificationResult:
        """
        Verify a file exists in the codebase.

        Args:
            file_path: File path to verify

        Returns:
            VerificationResult
        """
        file_info = await self.store.get_file(file_path)

        if file_info:
            return VerificationResult(
                claim=f"File {file_path} exists",
                verified=True,
                confidence=ConfidenceLevel.VERIFIED,
                supporting_evidence=[f"File has {file_info.line_count} lines"],
            )

        # Check if similar file exists
        similar = await self.store.find_similar_files(file_path)

        return VerificationResult(
            claim=f"File {file_path} exists",
            verified=False,
            confidence=ConfidenceLevel.VERIFIED,
            contradicting_evidence=[f"File '{file_path}' not found"],
            correction=f"Similar files: {', '.join(similar)}" if similar else None,
        )

    async def verify_claim(self, claim: str) -> VerificationResult:
        """
        Verify a natural language claim about the codebase.

        PATTERN: Parse claim, extract entities, verify each
        """
        # Extract entities from claim
        entities = self._extract_claim_entities(claim)

        if not entities:
            return VerificationResult(
                claim=claim,
                verified=False,
                confidence=ConfidenceLevel.UNKNOWN,
                contradicting_evidence=["Could not extract verifiable entities from claim"],
            )

        # Verify each entity
        results = []
        for entity_type, entity_name in entities:
            if entity_type == "symbol":
                result = await self.verify_symbol_exists(entity_name)
                results.append(result)
            elif entity_type == "file":
                result = await self.verify_file_exists(entity_name)
                results.append(result)

        if not results:
            return VerificationResult(
                claim=claim,
                verified=False,
                confidence=ConfidenceLevel.UNKNOWN,
                contradicting_evidence=["No entities could be verified"],
            )

        # Aggregate results
        all_verified = all(r.verified for r in results)
        min_confidence = min(r.confidence for r in results)

        return VerificationResult(
            claim=claim,
            verified=all_verified,
            confidence=min_confidence,
            supporting_evidence=[e for r in results for e in r.supporting_evidence],
            contradicting_evidence=[e for r in results for e in r.contradicting_evidence],
        )

    def _extract_claim_entities(
        self,
        claim: str,
    ) -> List[Tuple[str, str]]:
        """
        Extract verifiable entities from a claim.

        Returns:
            List of (entity_type, entity_name) tuples
        """
        entities = []

        # Look for file paths
        file_pattern = re.compile(r"[\w/\\]+\.(py|js|ts|jsx|tsx|java|go)")
        for match in file_pattern.finditer(claim):
            entities.append(("file", match.group()))

        # Look for qualified names (module.class.method)
        qualified_pattern = re.compile(r"[a-zA-Z_][\w]*(?:\.[a-zA-Z_][\w]*)+")
        for match in qualified_pattern.finditer(claim):
            entities.append(("symbol", match.group()))

        # Look for function/class names in specific patterns
        func_pattern = re.compile(r"(?:function|method|class|def)\s+([a-zA-Z_]\w*)")
        for match in func_pattern.finditer(claim.lower()):
            # Get actual case from original
            start = claim.lower().find(match.group())
            if start != -1:
                name_start = start + len(match.group()) - len(match.group(1))
                name = claim[name_start:name_start + len(match.group(1))]
                entities.append(("symbol", name))

        # Look for backtick-quoted names
        backtick_pattern = re.compile(r"`([a-zA-Z_][\w]*(?:\.[a-zA-Z_][\w]*)*)`")
        for match in backtick_pattern.finditer(claim):
            entities.append(("symbol", match.group(1)))

        return entities

    async def verify_before_claim(
        self,
        claim: str,
    ) -> str:
        """
        Verify before making a claim and return appropriate response.

        Use this to wrap agent responses.

        Args:
            claim: Claim about the codebase

        Returns:
            Original claim or modified response with uncertainty
        """
        result = await self.verify_claim(claim)

        if result.verified and result.confidence >= ConfidenceLevel.INFERRED:
            return claim

        if result.confidence == ConfidenceLevel.UNKNOWN:
            return f"I'm not certain about this. Let me check... {claim}"

        if not result.verified and result.correction:
            return result.correction

        return f"[Unverified] {claim}"

    async def get_symbol_info(
        self,
        symbol_name: str,
    ) -> Optional[str]:
        """
        Get verified information about a symbol.

        Args:
            symbol_name: Symbol to look up

        Returns:
            Information string or None
        """
        symbol = await self.store.get_symbol(symbol_name)

        if not symbol:
            return None

        lines = [
            f"**{symbol.kind.value}**: `{symbol.qualified_name}`",
            f"- **Location**: `{symbol.file_path}:{symbol.line_start}`",
        ]

        if symbol.signature:
            lines.append(f"- **Signature**: `{symbol.signature}`")

        if symbol.docstring:
            lines.append(f"- **Documentation**: {symbol.docstring[:200]}...")

        lines.append(f"- **Confidence**: {symbol.confidence.value}")

        return "\n".join(lines)
