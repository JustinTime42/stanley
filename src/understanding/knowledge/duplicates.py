"""Duplicate functionality detection."""

import logging
from difflib import SequenceMatcher
from typing import List, Optional

from .store import KnowledgeStore
from ..models import DuplicateCandidate, Symbol

logger = logging.getLogger(__name__)


class DuplicateDetector:
    """
    Detect potential duplicate functionality before creating new code.

    CRITICAL: Prevents redundant code creation
    PATTERN: Check similarity before every new function/class
    """

    def __init__(
        self,
        knowledge_store: KnowledgeStore,
        similarity_threshold: float = 0.8,
    ):
        """
        Initialize duplicate detector.

        Args:
            knowledge_store: Knowledge store to search
            similarity_threshold: Minimum similarity to report (0.0-1.0)
        """
        self.store = knowledge_store
        self.threshold = similarity_threshold

    async def check_before_create(
        self,
        proposed_name: str,
        proposed_description: str,
        proposed_kind: str = "function",
        proposed_signature: Optional[str] = None,
    ) -> List[DuplicateCandidate]:
        """
        Check for duplicates before creating new symbol.

        Args:
            proposed_name: Name of proposed new symbol
            proposed_description: What it should do
            proposed_kind: function, class, method, etc.
            proposed_signature: Optional function signature

        Returns:
            List of potential duplicate candidates
        """
        candidates = []

        # Check 1: Name similarity
        name_matches = await self.store.find_similar_symbols(
            proposed_name,
            kind=proposed_kind,
            limit=5,
        )

        for match in name_matches:
            similarity = self._name_similarity(proposed_name, match.name)
            if similarity >= self.threshold:
                candidates.append(DuplicateCandidate(
                    new_symbol=proposed_name,
                    existing_symbol=match.qualified_name,
                    similarity_score=similarity,
                    similarity_type="name",
                    new_description=proposed_description,
                    existing_description=match.docstring or match.description or "",
                    existing_location=f"{match.file_path}:{match.line_start}",
                    recommendation=self._recommend(similarity, "name"),
                ))

        # Check 2: Signature similarity (if provided)
        if proposed_signature:
            sig_matches = await self.store.find_by_signature_similarity(
                proposed_signature,
                kind=proposed_kind,
                limit=5,
            )

            for match in sig_matches:
                if match.qualified_name not in [c.existing_symbol for c in candidates]:
                    similarity = self._signature_similarity(
                        proposed_signature, match.signature or ""
                    )
                    if similarity >= self.threshold:
                        candidates.append(DuplicateCandidate(
                            new_symbol=proposed_name,
                            existing_symbol=match.qualified_name,
                            similarity_score=similarity,
                            similarity_type="signature",
                            new_description=proposed_description,
                            existing_description=match.docstring or "",
                            existing_location=f"{match.file_path}:{match.line_start}",
                            recommendation=self._recommend(similarity, "signature"),
                        ))

        # Check 3: Semantic similarity (most important)
        semantic_matches = await self.store.semantic_search(
            proposed_description,
            kind=proposed_kind,
            limit=5,
        )

        for match, score in semantic_matches:
            if match.qualified_name not in [c.existing_symbol for c in candidates]:
                if score >= self.threshold:
                    candidates.append(DuplicateCandidate(
                        new_symbol=proposed_name,
                        existing_symbol=match.qualified_name,
                        similarity_score=score,
                        similarity_type="semantic",
                        new_description=proposed_description,
                        existing_description=match.docstring or match.description or "",
                        existing_location=f"{match.file_path}:{match.line_start}",
                        recommendation=self._recommend(score, "semantic"),
                    ))

        # Sort by similarity score
        candidates.sort(key=lambda c: c.similarity_score, reverse=True)

        return candidates

    def _name_similarity(self, name1: str, name2: str) -> float:
        """Calculate name similarity using SequenceMatcher."""
        # Normalize: convert to lowercase and split on underscores/camelCase
        def normalize(name: str) -> str:
            # Split camelCase
            import re
            words = re.sub(r"([a-z])([A-Z])", r"\1_\2", name)
            return words.lower()

        return SequenceMatcher(None, normalize(name1), normalize(name2)).ratio()

    def _signature_similarity(self, sig1: str, sig2: str) -> float:
        """Calculate signature similarity."""
        # Extract parameter names
        def extract_params(sig: str) -> set:
            import re
            # Find content between parentheses
            match = re.search(r"\(([^)]*)\)", sig)
            if not match:
                return set()
            params = match.group(1)
            # Split by comma and get parameter names
            param_names = set()
            for param in params.split(","):
                param = param.strip()
                if param:
                    # Get first word (parameter name)
                    name = param.split(":")[0].split("=")[0].strip()
                    param_names.add(name.lower())
            return param_names

        params1 = extract_params(sig1)
        params2 = extract_params(sig2)

        if not params1 and not params2:
            return 0.5  # Both empty, moderate similarity

        if not params1 or not params2:
            return 0.0

        # Jaccard similarity
        intersection = len(params1 & params2)
        union = len(params1 | params2)

        return intersection / union if union > 0 else 0.0

    def _recommend(self, similarity: float, match_type: str) -> str:
        """Generate recommendation based on similarity."""
        if similarity >= 0.95:
            return "use_existing"
        elif similarity >= 0.85:
            if match_type == "semantic":
                return "use_existing"
            return "consider_existing"
        elif similarity >= 0.7:
            return "extend_existing"
        return "create_new"

    async def find_similar_implementations(
        self,
        code_snippet: str,
        limit: int = 5,
    ) -> List[DuplicateCandidate]:
        """
        Find similar implementations to a code snippet.

        Args:
            code_snippet: Code to find duplicates of
            limit: Maximum results

        Returns:
            List of similar implementations
        """
        # Use semantic search with code as query
        matches = await self.store.semantic_search(code_snippet, limit=limit)

        candidates = []
        for match, score in matches:
            if score >= self.threshold * 0.8:  # Lower threshold for code
                candidates.append(DuplicateCandidate(
                    new_symbol="<snippet>",
                    existing_symbol=match.qualified_name,
                    similarity_score=score,
                    similarity_type="implementation",
                    new_description=code_snippet[:100],
                    existing_description=match.docstring or "",
                    existing_location=f"{match.file_path}:{match.line_start}",
                    recommendation=self._recommend(score, "implementation"),
                ))

        return candidates

    async def get_duplicate_report(
        self,
        proposed_items: List[dict],
    ) -> dict:
        """
        Generate duplicate report for multiple proposed items.

        Args:
            proposed_items: List of {name, description, kind, signature}

        Returns:
            Report with duplicates found
        """
        report = {
            "total_checked": len(proposed_items),
            "duplicates_found": 0,
            "items": [],
        }

        for item in proposed_items:
            candidates = await self.check_before_create(
                proposed_name=item.get("name", ""),
                proposed_description=item.get("description", ""),
                proposed_kind=item.get("kind", "function"),
                proposed_signature=item.get("signature"),
            )

            if candidates:
                report["duplicates_found"] += 1

            report["items"].append({
                "proposed": item.get("name"),
                "duplicates": [
                    {
                        "existing": c.existing_symbol,
                        "similarity": round(c.similarity_score, 2),
                        "type": c.similarity_type,
                        "recommendation": c.recommendation,
                        "location": c.existing_location,
                    }
                    for c in candidates
                ],
            })

        return report
