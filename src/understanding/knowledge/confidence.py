"""Confidence scoring system."""

import logging
from datetime import datetime, timedelta
from typing import List, Optional

from ..models import Symbol, ConfidenceLevel

logger = logging.getLogger(__name__)


class ConfidenceScorer:
    """
    Score and track confidence in knowledge.

    PATTERN: Score based on analysis method, age, verification
    CRITICAL: Degrade confidence over time without verification
    """

    # Time thresholds for confidence degradation
    STALE_THRESHOLD = timedelta(hours=24)
    UNCERTAIN_THRESHOLD = timedelta(days=7)

    def __init__(
        self,
        stale_hours: int = 24,
        uncertain_days: int = 7,
    ):
        """
        Initialize confidence scorer.

        Args:
            stale_hours: Hours until verified becomes stale
            uncertain_days: Days until stale becomes uncertain
        """
        self.stale_threshold = timedelta(hours=stale_hours)
        self.uncertain_threshold = timedelta(days=uncertain_days)

    def score_symbol(self, symbol: Symbol) -> ConfidenceLevel:
        """
        Score confidence for a symbol.

        Args:
            symbol: Symbol to score

        Returns:
            Updated confidence level
        """
        current_confidence = symbol.confidence
        age = datetime.now() - symbol.last_verified

        # If already unknown, stay unknown
        if current_confidence == ConfidenceLevel.UNKNOWN:
            return ConfidenceLevel.UNKNOWN

        # Degrade based on age
        if current_confidence == ConfidenceLevel.VERIFIED:
            if age > self.stale_threshold:
                return ConfidenceLevel.STALE
            return ConfidenceLevel.VERIFIED

        if current_confidence == ConfidenceLevel.INFERRED:
            if age > self.stale_threshold:
                return ConfidenceLevel.UNCERTAIN
            return ConfidenceLevel.INFERRED

        if current_confidence == ConfidenceLevel.STALE:
            if age > self.uncertain_threshold:
                return ConfidenceLevel.UNCERTAIN
            return ConfidenceLevel.STALE

        return current_confidence

    def boost_confidence(
        self,
        symbol: Symbol,
        reason: str = "verification",
    ) -> ConfidenceLevel:
        """
        Boost confidence when a claim is confirmed.

        Args:
            symbol: Symbol to boost
            reason: Reason for boost

        Returns:
            New confidence level
        """
        current = symbol.confidence

        # Can't boost unknown
        if current == ConfidenceLevel.UNKNOWN:
            return ConfidenceLevel.UNCERTAIN

        # Boost stale to inferred
        if current == ConfidenceLevel.STALE:
            return ConfidenceLevel.INFERRED

        # Boost uncertain to inferred
        if current == ConfidenceLevel.UNCERTAIN:
            return ConfidenceLevel.INFERRED

        # Boost inferred to verified
        if current == ConfidenceLevel.INFERRED:
            return ConfidenceLevel.VERIFIED

        return current

    def degrade_confidence(
        self,
        symbol: Symbol,
        reason: str = "file_changed",
    ) -> ConfidenceLevel:
        """
        Degrade confidence when source changes.

        Args:
            symbol: Symbol to degrade
            reason: Reason for degradation

        Returns:
            New confidence level
        """
        current = symbol.confidence

        # Can't degrade unknown
        if current == ConfidenceLevel.UNKNOWN:
            return ConfidenceLevel.UNKNOWN

        # Degrade verified to stale
        if current == ConfidenceLevel.VERIFIED:
            return ConfidenceLevel.STALE

        # Degrade inferred to uncertain
        if current == ConfidenceLevel.INFERRED:
            return ConfidenceLevel.UNCERTAIN

        # Degrade stale to uncertain
        if current == ConfidenceLevel.STALE:
            return ConfidenceLevel.UNCERTAIN

        return ConfidenceLevel.UNKNOWN

    def compute_aggregate_confidence(
        self,
        symbols: List[Symbol],
    ) -> ConfidenceLevel:
        """
        Compute aggregate confidence from multiple symbols.

        Args:
            symbols: List of symbols

        Returns:
            Minimum confidence level
        """
        if not symbols:
            return ConfidenceLevel.UNKNOWN

        # Return minimum confidence
        confidences = [s.confidence for s in symbols]
        return min(confidences)

    def confidence_to_score(self, confidence: ConfidenceLevel) -> float:
        """Convert confidence level to numeric score."""
        scores = {
            ConfidenceLevel.VERIFIED: 1.0,
            ConfidenceLevel.INFERRED: 0.75,
            ConfidenceLevel.UNCERTAIN: 0.5,
            ConfidenceLevel.STALE: 0.25,
            ConfidenceLevel.UNKNOWN: 0.0,
        }
        return scores.get(confidence, 0.0)

    def score_to_confidence(self, score: float) -> ConfidenceLevel:
        """Convert numeric score to confidence level."""
        if score >= 0.9:
            return ConfidenceLevel.VERIFIED
        if score >= 0.7:
            return ConfidenceLevel.INFERRED
        if score >= 0.4:
            return ConfidenceLevel.UNCERTAIN
        if score >= 0.2:
            return ConfidenceLevel.STALE
        return ConfidenceLevel.UNKNOWN

    def should_reverify(self, symbol: Symbol) -> bool:
        """Check if symbol should be re-verified."""
        age = datetime.now() - symbol.last_verified

        if symbol.confidence == ConfidenceLevel.STALE:
            return True

        if symbol.confidence == ConfidenceLevel.VERIFIED:
            return age > self.stale_threshold

        if symbol.confidence == ConfidenceLevel.INFERRED:
            return age > timedelta(hours=12)

        return True

    def get_confidence_summary(
        self,
        symbols: List[Symbol],
    ) -> dict:
        """Get confidence distribution summary."""
        summary = {
            "verified": 0,
            "inferred": 0,
            "uncertain": 0,
            "stale": 0,
            "unknown": 0,
            "total": len(symbols),
        }

        for symbol in symbols:
            level = symbol.confidence.value
            if level in summary:
                summary[level] += 1

        # Calculate percentages
        if summary["total"] > 0:
            summary["verified_pct"] = summary["verified"] / summary["total"] * 100
            summary["high_confidence_pct"] = (
                (summary["verified"] + summary["inferred"]) / summary["total"] * 100
            )

        return summary
