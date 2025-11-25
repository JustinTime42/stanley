"""Knowledge query and response models."""

from pydantic import BaseModel, Field
from typing import List, Optional
from .understanding_models import ConfidenceLevel


class KnowledgeQuery(BaseModel):
    """Query about codebase knowledge."""

    query: str = Field(description="Natural language or symbol query")
    query_type: str = Field(
        default="auto",
        description="Query type: symbol, file, concept, relationship, auto",
    )
    context: Optional[str] = Field(
        default=None, description="Additional context for the query"
    )
    max_results: int = Field(default=10, ge=1, le=100)
    min_confidence: ConfidenceLevel = Field(
        default=ConfidenceLevel.UNCERTAIN,
        description="Minimum confidence level for results",
    )


class KnowledgeResponse(BaseModel):
    """Response to knowledge query with confidence."""

    query: str = Field(description="Original query")
    answer: str = Field(description="Answer to the query")
    confidence: ConfidenceLevel = Field(description="Overall confidence")

    # Evidence
    sources: List[str] = Field(
        default_factory=list, description="Files/symbols supporting answer"
    )
    verified_claims: List[str] = Field(
        default_factory=list, description="Verified claims in answer"
    )
    uncertain_claims: List[str] = Field(
        default_factory=list, description="Uncertain claims in answer"
    )

    # Gaps
    knowledge_gaps: List[str] = Field(
        default_factory=list, description="What we don't know"
    )
    suggested_investigation: List[str] = Field(
        default_factory=list, description="Suggested areas to investigate"
    )


class VerificationResult(BaseModel):
    """Result of verifying a claim about the codebase."""

    claim: str = Field(description="The claim being verified")
    verified: bool = Field(description="Whether claim is verified")
    confidence: ConfidenceLevel = Field(description="Confidence in verification")

    # Evidence
    supporting_evidence: List[str] = Field(
        default_factory=list, description="Evidence supporting claim"
    )
    contradicting_evidence: List[str] = Field(
        default_factory=list, description="Evidence contradicting claim"
    )

    # Correction
    correction: Optional[str] = Field(
        default=None, description="Corrected claim if wrong"
    )

    def is_certain(self) -> bool:
        """Check if verification is certain."""
        return self.confidence in (ConfidenceLevel.VERIFIED, ConfidenceLevel.INFERRED)

    def get_summary(self) -> str:
        """Get summary of verification result."""
        if self.verified:
            return f"Verified ({self.confidence.value}): {self.claim}"
        else:
            correction = f" -> {self.correction}" if self.correction else ""
            return f"Not verified ({self.confidence.value}): {self.claim}{correction}"
