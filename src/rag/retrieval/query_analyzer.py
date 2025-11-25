"""Query analyzer for intent detection and feature extraction."""

import logging
import re
from typing import List, Set

from ...models.rag_models import QueryAnalysis, QueryIntent

logger = logging.getLogger(__name__)


class QueryAnalyzer:
    """
    Analyze search queries to detect intent and extract features.

    PATTERN: Multi-signal classification using keywords and patterns
    CRITICAL: Balance accuracy vs speed for real-time analysis
    """

    # Programming-related keywords
    PROGRAMMING_KEYWORDS: Set[str] = {
        "function",
        "class",
        "method",
        "variable",
        "import",
        "export",
        "async",
        "await",
        "return",
        "def",
        "const",
        "let",
        "var",
        "interface",
        "type",
        "struct",
        "enum",
        "trait",
        "impl",
        "python",
        "javascript",
        "typescript",
        "java",
        "go",
        "rust",
        "api",
        "endpoint",
        "route",
        "handler",
        "middleware",
    }

    # Intent pattern keywords
    INTENT_PATTERNS = {
        QueryIntent.EXAMPLE_SEARCH: [
            "how to",
            "implement",
            "create",
            "build",
            "make",
            "example",
            "sample",
            "tutorial",
            "guide",
        ],
        QueryIntent.DEFINITION: [
            "what is",
            "define",
            "definition",
            "meaning",
            "explain what",
        ],
        QueryIntent.EXPLANATION: [
            "why",
            "explain",
            "understand",
            "how does",
            "reason",
            "purpose",
            "difference between",
        ],
        QueryIntent.DEBUGGING: [
            "error",
            "bug",
            "fix",
            "debug",
            "issue",
            "problem",
            "not working",
            "fails",
            "crash",
            "exception",
        ],
        QueryIntent.CODE_SEARCH: [
            "find code",
            "search for",
            "locate",
            "where is",
            "show me the",
            "implementation of",
        ],
        QueryIntent.DOCUMENTATION: [
            "docs",
            "documentation",
            "manual",
            "reference",
            "readme",
            "guide",
            "specification",
        ],
    }

    def __init__(self):
        """Initialize query analyzer."""
        self.logger = logger

    def analyze_query(self, query: str) -> QueryAnalysis:
        """
        Analyze query to detect intent and extract features.

        PATTERN: Keyword-based classification with confidence scoring
        GOTCHA: Edge cases with ambiguous queries

        Args:
            query: Search query string

        Returns:
            QueryAnalysis with detected intent and features
        """
        query_lower = query.lower()

        # Extract keywords
        keywords = self._extract_keywords(query)

        # Detect programming terms
        programming_terms = self._detect_programming_terms(query_lower, keywords)

        # Extract entities (quoted strings, identifiers)
        entities = self._extract_entities(query)

        # Detect intent
        intent, confidence = self._detect_intent(
            query_lower, keywords, programming_terms
        )

        # Determine content requirements
        requires_code = self._requires_code(query_lower, programming_terms)
        requires_docs = self._requires_documentation(query_lower)

        analysis = QueryAnalysis(
            query=query,
            intent=intent,
            keywords=keywords,
            entities=entities,
            programming_terms=programming_terms,
            requires_code=requires_code,
            requires_docs=requires_docs,
            confidence=confidence,
        )

        self.logger.debug(
            f"Query analysis: intent={intent.value}, confidence={confidence:.2f}"
        )

        return analysis

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract significant keywords from query."""
        # Remove common stop words
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "have",
            "has",
            "had",
        }

        # Extract words
        words = re.findall(r"\b\w+\b", query.lower())

        # Filter stop words and short words
        keywords = [word for word in words if word not in stop_words and len(word) > 2]

        return keywords

    def _detect_programming_terms(
        self,
        query_lower: str,
        keywords: List[str],
    ) -> List[str]:
        """Detect programming-related terms."""
        programming_terms = []

        # Check against programming keyword set
        for keyword in keywords:
            if keyword in self.PROGRAMMING_KEYWORDS:
                programming_terms.append(keyword)

        # Check for camelCase or snake_case identifiers
        identifier_pattern = re.compile(r"\b([a-z]+[A-Z][a-zA-Z]*|[a-z]+_[a-z]+)\b")
        for match in identifier_pattern.finditer(query_lower):
            programming_terms.append(match.group(1))

        return programming_terms

    def _extract_entities(self, query: str) -> List[str]:
        """Extract named entities (quoted strings, identifiers)."""
        entities = []

        # Extract quoted strings
        quoted_pattern = re.compile(r'["\']([^"\']+)["\']')
        for match in quoted_pattern.finditer(query):
            entities.append(match.group(1))

        # Extract capitalized words (potential class/type names)
        capitalized_pattern = re.compile(r"\b([A-Z][a-z]+(?:[A-Z][a-z]+)*)\b")
        for match in capitalized_pattern.finditer(query):
            entities.append(match.group(1))

        return entities

    def _detect_intent(
        self,
        query_lower: str,
        keywords: List[str],
        programming_terms: List[str],
    ) -> tuple[QueryIntent, float]:
        """
        Detect primary query intent with confidence.

        Returns:
            Tuple of (intent, confidence)
        """
        scores = {intent: 0.0 for intent in QueryIntent}

        # Score each intent based on pattern matches
        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if pattern in query_lower:
                    scores[intent] += 1.0

        # Adjust scores based on context
        if len(programming_terms) > 2:
            scores[QueryIntent.CODE_SEARCH] += 0.5

        # Find highest scoring intent
        max_score = max(scores.values())

        if max_score > 0:
            # Get intent with highest score
            best_intent = max(scores.items(), key=lambda x: x[1])[0]
            confidence = min(1.0, max_score / 3.0)  # Normalize to 0-1
        else:
            # Default to QUESTION_ANSWER
            best_intent = QueryIntent.QUESTION_ANSWER
            confidence = 0.5

        return best_intent, confidence

    def _requires_code(self, query_lower: str, programming_terms: List[str]) -> bool:
        """Determine if query requires code context."""
        code_indicators = [
            "function",
            "class",
            "implement",
            "code",
            "example",
            "syntax",
            "method",
            "api",
            "library",
        ]

        # Check for code indicators
        for indicator in code_indicators:
            if indicator in query_lower:
                return True

        # Check programming terms
        if len(programming_terms) > 1:
            return True

        return False

    def _requires_documentation(self, query_lower: str) -> bool:
        """Determine if query requires documentation."""
        doc_indicators = [
            "how",
            "what",
            "why",
            "explain",
            "documentation",
            "guide",
            "tutorial",
            "manual",
            "docs",
        ]

        for indicator in doc_indicators:
            if indicator in query_lower:
                return True

        return False
