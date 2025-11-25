"""Task complexity analyzer for intelligent model routing."""

import logging
import uuid
from typing import List, Dict
from ..models.llm_models import TaskComplexity, ModelCapability
from ..models.routing_models import TaskAnalysis

logger = logging.getLogger(__name__)


class TaskComplexityAnalyzer:
    """
    Analyzes task complexity to enable intelligent model routing.

    PATTERN: Multi-factor heuristic analysis balancing accuracy and speed
    CRITICAL: Must complete in <100ms for routing decision
    """

    def __init__(self):
        """Initialize task complexity analyzer."""
        self.logger = logging.getLogger(__name__)

        # Keywords indicating complex tasks
        self.complex_keywords = {
            "architecture",
            "design",
            "refactor",
            "optimize",
            "complex",
            "advanced",
            "sophisticated",
            "comprehensive",
            "intricate",
            "multi-step",
            "system",
            "scalable",
            "performance",
            "security",
        }

        # Keywords indicating simple tasks
        self.simple_keywords = {
            "fix",
            "typo",
            "rename",
            "format",
            "simple",
            "basic",
            "quick",
            "minor",
            "small",
            "trivial",
            "obvious",
            "straightforward",
        }

        # Capability keywords mapping
        self.capability_keywords = {
            ModelCapability.CODE_GENERATION: {
                "implement",
                "create",
                "write",
                "generate",
                "code",
                "function",
                "class",
                "method",
            },
            ModelCapability.CODE_REVIEW: {
                "review",
                "check",
                "analyze",
                "validate",
                "inspect",
                "audit",
            },
            ModelCapability.PLANNING: {
                "plan",
                "design",
                "architecture",
                "strategy",
                "approach",
                "outline",
            },
            ModelCapability.DEBUGGING: {
                "debug",
                "fix",
                "error",
                "bug",
                "issue",
                "problem",
                "troubleshoot",
            },
            ModelCapability.TESTING: {
                "test",
                "verify",
                "validate",
                "qa",
                "coverage",
                "unittest",
            },
            ModelCapability.DOCUMENTATION: {
                "document",
                "explain",
                "describe",
                "comment",
                "readme",
                "docs",
            },
        }

    def analyze_task(
        self,
        task_description: str,
        agent_role: str = "general",
        message_history: List[Dict] = None,
    ) -> TaskAnalysis:
        """
        Analyze task complexity and requirements.

        PATTERN: Combine multiple signals for robust classification
        CRITICAL: Balance accuracy with <100ms latency requirement

        Args:
            task_description: Description of the task
            agent_role: Role of the requesting agent
            message_history: Optional message history for context

        Returns:
            TaskAnalysis with complexity and requirements
        """
        message_history = message_history or []

        # Estimate tokens (simple word-based)
        estimated_tokens = self._estimate_tokens(task_description, message_history)

        # Calculate complexity score
        complexity_score = self._calculate_complexity_score(
            task_description,
            agent_role,
            estimated_tokens,
        )

        # Map score to complexity level
        if complexity_score < 0.3:
            complexity = TaskComplexity.SIMPLE
        elif complexity_score < 0.7:
            complexity = TaskComplexity.MEDIUM
        else:
            complexity = TaskComplexity.COMPLEX

        # Identify required capabilities
        required_capabilities = self._identify_capabilities(
            task_description,
            agent_role,
        )

        # Generate reasoning
        reasoning = self._generate_reasoning(
            task_description,
            complexity_score,
            estimated_tokens,
        )

        # Determine confidence
        confidence = self._calculate_confidence(
            task_description,
            complexity_score,
        )

        return TaskAnalysis(
            task_id=str(uuid.uuid4()),
            complexity=complexity,
            estimated_tokens=estimated_tokens,
            required_capabilities=required_capabilities,
            requires_functions=self._requires_functions(task_description),
            requires_vision=self._requires_vision(task_description),
            confidence=confidence,
            reasoning=reasoning,
        )

    def _estimate_tokens(
        self,
        task_description: str,
        message_history: List[Dict],
    ) -> int:
        """
        Estimate token usage for the task.

        PATTERN: Simple word-based estimation for speed
        GOTCHA: ~1.3 tokens per word is a reasonable approximation

        Args:
            task_description: Task description
            message_history: Message history

        Returns:
            Estimated token count
        """
        # Count words in task description
        task_words = len(task_description.split())

        # Count words in message history
        history_words = 0
        for msg in message_history:
            content = msg.get("content", "")
            history_words += len(content.split())

        # Apply token multiplier
        total_tokens = int((task_words + history_words) * 1.3)

        # Add overhead for message formatting
        total_tokens += len(message_history) * 4

        return total_tokens

    def _calculate_complexity_score(
        self,
        task_description: str,
        agent_role: str,
        estimated_tokens: int,
    ) -> float:
        """
        Calculate complexity score (0-1).

        PATTERN: Multi-factor scoring combining keywords, length, and role
        CRITICAL: Score must be normalized to 0-1 range

        Args:
            task_description: Task description
            agent_role: Agent role
            estimated_tokens: Estimated tokens

        Returns:
            Complexity score (0-1)
        """
        score = 0.0
        lower_desc = task_description.lower()

        # Factor 1: Keyword analysis (40% weight)
        complex_count = sum(
            1 for kw in self.complex_keywords
            if kw in lower_desc
        )
        simple_count = sum(
            1 for kw in self.simple_keywords
            if kw in lower_desc
        )

        keyword_score = 0.5  # Default neutral
        if complex_count > simple_count:
            keyword_score = 0.6 + min(complex_count * 0.1, 0.4)
        elif simple_count > complex_count:
            keyword_score = 0.4 - min(simple_count * 0.1, 0.4)

        score += keyword_score * 0.4

        # Factor 2: Token count (30% weight)
        # More tokens generally means more complex
        if estimated_tokens < 500:
            token_score = 0.2
        elif estimated_tokens < 2000:
            token_score = 0.5
        else:
            token_score = 0.8

        score += token_score * 0.3

        # Factor 3: Agent role (30% weight)
        # Some roles inherently handle more complex tasks
        role_complexity = {
            "architect": 0.8,
            "planner": 0.7,
            "debugger": 0.6,
            "implementer": 0.6,
            "tester": 0.4,
            "validator": 0.5,
            "coordinator": 0.5,
        }
        role_score = role_complexity.get(agent_role.lower(), 0.5)
        score += role_score * 0.3

        # Normalize to 0-1 range
        return max(0.0, min(1.0, score))

    def _identify_capabilities(
        self,
        task_description: str,
        agent_role: str,
    ) -> List[ModelCapability]:
        """
        Identify required model capabilities.

        PATTERN: Keyword matching with agent role context
        GOTCHA: Multiple capabilities may be required

        Args:
            task_description: Task description
            agent_role: Agent role

        Returns:
            List of required capabilities
        """
        lower_desc = task_description.lower()
        capabilities = []

        # Check each capability
        for capability, keywords in self.capability_keywords.items():
            if any(kw in lower_desc for kw in keywords):
                capabilities.append(capability)

        # Default based on agent role if no capabilities detected
        if not capabilities:
            role_default_capability = {
                "planner": ModelCapability.PLANNING,
                "architect": ModelCapability.CODE_GENERATION,
                "implementer": ModelCapability.CODE_GENERATION,
                "tester": ModelCapability.TESTING,
                "validator": ModelCapability.CODE_REVIEW,
                "debugger": ModelCapability.DEBUGGING,
            }
            default = role_default_capability.get(
                agent_role.lower(),
                ModelCapability.GENERAL
            )
            capabilities.append(default)

        return capabilities

    def _requires_functions(self, task_description: str) -> bool:
        """
        Check if task requires function calling.

        Args:
            task_description: Task description

        Returns:
            True if function calling needed
        """
        function_keywords = {
            "api call",
            "function call",
            "tool use",
            "execute command",
        }
        lower_desc = task_description.lower()
        return any(kw in lower_desc for kw in function_keywords)

    def _requires_vision(self, task_description: str) -> bool:
        """
        Check if task requires vision capabilities.

        Args:
            task_description: Task description

        Returns:
            True if vision needed
        """
        vision_keywords = {
            "image",
            "screenshot",
            "diagram",
            "visual",
            "chart",
        }
        lower_desc = task_description.lower()
        return any(kw in lower_desc for kw in vision_keywords)

    def _calculate_confidence(
        self,
        task_description: str,
        complexity_score: float,
    ) -> float:
        """
        Calculate confidence in the analysis.

        PATTERN: Higher confidence with more signals
        CRITICAL: Return realistic confidence levels

        Args:
            task_description: Task description
            complexity_score: Calculated complexity score

        Returns:
            Confidence score (0-1)
        """
        confidence = 0.7  # Base confidence

        # More text = higher confidence
        word_count = len(task_description.split())
        if word_count > 50:
            confidence += 0.1
        elif word_count < 10:
            confidence -= 0.1

        # Extreme scores are less confident
        if 0.3 < complexity_score < 0.7:
            confidence -= 0.05  # Mid-range is ambiguous

        # Normalize
        return max(0.5, min(1.0, confidence))

    def _generate_reasoning(
        self,
        task_description: str,
        complexity_score: float,
        estimated_tokens: int,
    ) -> str:
        """
        Generate human-readable reasoning for the classification.

        Args:
            task_description: Task description
            complexity_score: Complexity score
            estimated_tokens: Estimated tokens

        Returns:
            Reasoning string
        """
        reasons = []

        # Complexity reasoning
        if complexity_score < 0.3:
            reasons.append("simple task with basic requirements")
        elif complexity_score < 0.7:
            reasons.append("moderate complexity requiring capable model")
        else:
            reasons.append("complex task requiring premium model")

        # Token reasoning
        if estimated_tokens > 2000:
            reasons.append(f"large context ({estimated_tokens} tokens)")
        elif estimated_tokens < 500:
            reasons.append("compact task")

        # Keyword analysis
        lower_desc = task_description.lower()
        complex_found = [
            kw for kw in self.complex_keywords
            if kw in lower_desc
        ]
        if complex_found:
            reasons.append(f"complexity indicators: {', '.join(complex_found[:3])}")

        return "; ".join(reasons)
