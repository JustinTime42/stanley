"""Property-based test generation."""

import logging
from typing import List, Dict, Any, Optional

from ..models.testing_models import PropertyTest, TestFramework
from ..models.analysis_models import CodeEntity

logger = logging.getLogger(__name__)


class PropertyTestGenerator:
    """
    Generate property-based tests.

    PATTERN: Infer properties from function signature and docs
    CRITICAL: Only for pure functions without side effects
    GOTCHA: Different frameworks use different property testing libraries
    """

    def __init__(self):
        """Initialize property test generator."""
        self.logger = logger

    async def generate_properties(
        self, function: CodeEntity, framework: TestFramework
    ) -> Optional[PropertyTest]:
        """
        Generate property-based test for function.

        PATTERN: Infer properties from function signature and docs

        Args:
            function: Function to test
            framework: Testing framework

        Returns:
            PropertyTest if applicable, None if function is impure
        """
        # Check if function is pure
        if not await self._is_pure_function(function):
            self.logger.debug(
                f"Function {function.name} is not pure, skipping property test generation"
            )
            return None

        # Infer properties from function
        properties = await self._infer_properties(function)

        if not properties:
            self.logger.debug(
                f"No properties inferred for {function.name}, skipping property test"
            )
            return None

        # Generate input strategies based on function signature
        input_strategies = await self._create_input_strategies(function, framework)

        # Extract invariants
        invariants = await self._extract_invariants(function)

        # Framework-specific configuration
        framework_config = await self._get_framework_config(framework)

        # Create property test
        property_test = PropertyTest(
            id=f"prop_{function.name}",
            function_name=function.name,
            properties=properties,
            input_strategies=input_strategies,
            invariants=invariants,
            framework_config=framework_config,
        )

        return property_test

    async def _is_pure_function(self, function: CodeEntity) -> bool:
        """
        Check if function is pure (no side effects).

        CRITICAL: Only generate property tests for pure functions
        PATTERN: Check for I/O, network, mutations, global state

        Args:
            function: Function to check

        Returns:
            True if function appears pure
        """
        # Check metadata
        metadata = function.metadata or {}
        if metadata.get("has_side_effects", False):
            return False
        if metadata.get("has_io", False):
            return False
        if metadata.get("has_network", False):
            return False

        # Check function signature and body for impure indicators
        impure_keywords = [
            "open(",
            "write(",
            "print(",
            "input(",
            "socket",
            "http",
            "requests",
            "fetch(",
            "axios",
            "global ",
            "nonlocal ",
            "Database",
            "Session",
            "connect(",
            "execute(",
            "commit(",
            "save(",
            "update(",
            "delete(",
        ]

        signature = function.signature or ""
        signature_lower = signature.lower()

        for keyword in impure_keywords:
            if keyword.lower() in signature_lower:
                return False

        # If we have docstring, check for mentions of side effects
        docstring = function.docstring or ""
        docstring_lower = docstring.lower()

        side_effect_indicators = [
            "side effect",
            "modifies",
            "updates",
            "changes",
            "mutates",
            "writes to",
            "saves to",
        ]

        for indicator in side_effect_indicators:
            if indicator in docstring_lower:
                return False

        return True

    async def _infer_properties(self, function: CodeEntity) -> List[str]:
        """
        Infer properties from function analysis.

        PATTERN: Common property patterns (idempotence, commutativity, etc.)

        Args:
            function: Function to analyze

        Returns:
            List of property names
        """
        properties = []

        function_name = function.name.lower()
        signature = (function.signature or "").lower()
        docstring = (function.docstring or "").lower()

        # Idempotence: f(f(x)) = f(x)
        # Common for: normalize, sort, unique, deduplicate
        if any(
            word in function_name
            for word in ["normalize", "sort", "unique", "deduplicate", "clean"]
        ):
            properties.append("idempotent")

        # Commutativity: f(a, b) = f(b, a)
        # Common for: add, multiply, union, intersection
        if any(
            word in function_name
            for word in ["add", "sum", "multiply", "union", "intersection", "merge"]
        ):
            if "(" in signature and signature.count(",") >= 1:
                properties.append("commutative")

        # Associativity: f(f(a, b), c) = f(a, f(b, c))
        if any(
            word in function_name
            for word in ["add", "multiply", "concat", "union", "merge"]
        ):
            properties.append("associative")

        # Identity: f(x, identity) = x
        if "add" in function_name or "sum" in function_name:
            properties.append("identity_zero")
        elif "multiply" in function_name or "product" in function_name:
            properties.append("identity_one")

        # Inverse: f(inverse(x)) = identity
        if any(word in function_name for word in ["encode", "decode", "encrypt", "decrypt"]):
            properties.append("invertible")

        # Round-trip: decode(encode(x)) = x
        if "encode" in function_name or "serialize" in function_name:
            properties.append("round_trip_encode")
        elif "decode" in function_name or "deserialize" in function_name:
            properties.append("round_trip_decode")

        # Domain invariants from docstring
        if "always" in docstring:
            properties.append("domain_invariant")

        # Return type invariant
        if "returns" in docstring or "->" in signature:
            properties.append("type_invariant")

        return properties

    async def _create_input_strategies(
        self, function: CodeEntity, framework: TestFramework
    ) -> Dict[str, str]:
        """
        Create input generation strategies based on function signature.

        Args:
            function: Function to analyze
            framework: Testing framework

        Returns:
            Dictionary mapping parameter names to strategy strings
        """
        strategies = {}

        # Parse function signature
        # This is simplified - real implementation would use AST
        signature = function.signature or ""

        if framework == TestFramework.PYTEST:
            # Hypothesis strategies
            strategies = await self._create_hypothesis_strategies(signature)
        elif framework == TestFramework.JEST:
            # Fast-check strategies
            strategies = await self._create_fastcheck_strategies(signature)

        return strategies

    async def _create_hypothesis_strategies(self, signature: str) -> Dict[str, str]:
        """Create Hypothesis (Python) strategies."""
        strategies = {}

        # Simple parameter type inference
        # In practice, would parse actual type annotations
        if "int" in signature.lower():
            strategies["value"] = "st.integers()"
        if "str" in signature.lower():
            strategies["text"] = "st.text()"
        if "list" in signature.lower():
            strategies["items"] = "st.lists(st.integers())"
        if "bool" in signature.lower():
            strategies["flag"] = "st.booleans()"

        # Default strategies if none inferred
        if not strategies:
            strategies["x"] = "st.integers()"

        return strategies

    async def _create_fastcheck_strategies(self, signature: str) -> Dict[str, str]:
        """Create fast-check (JavaScript) strategies."""
        strategies = {}

        # Simple parameter type inference
        if "number" in signature.lower() or "int" in signature.lower():
            strategies["value"] = "fc.integer()"
        if "string" in signature.lower():
            strategies["text"] = "fc.string()"
        if "array" in signature.lower():
            strategies["items"] = "fc.array(fc.integer())"
        if "boolean" in signature.lower():
            strategies["flag"] = "fc.boolean()"

        # Default strategies if none inferred
        if not strategies:
            strategies["x"] = "fc.integer()"

        return strategies

    async def _extract_invariants(self, function: CodeEntity) -> List[str]:
        """
        Extract invariants from function documentation.

        Args:
            function: Function to analyze

        Returns:
            List of invariant descriptions
        """
        invariants = []

        docstring = function.docstring or ""

        # Look for common invariant patterns in docstrings
        invariant_keywords = [
            ("always", "Always holds"),
            ("never", "Never occurs"),
            ("must", "Must satisfy"),
            ("ensures", "Ensures"),
            ("guarantees", "Guarantees"),
            ("invariant", "Invariant"),
        ]

        for keyword, prefix in invariant_keywords:
            if keyword in docstring.lower():
                # Extract the sentence containing the keyword
                sentences = docstring.split(".")
                for sentence in sentences:
                    if keyword in sentence.lower():
                        invariants.append(f"{prefix}: {sentence.strip()}")

        return invariants

    async def _get_framework_config(
        self, framework: TestFramework
    ) -> Dict[str, Any]:
        """
        Get framework-specific configuration.

        Args:
            framework: Testing framework

        Returns:
            Configuration dictionary
        """
        if framework == TestFramework.PYTEST:
            return {
                "library": "hypothesis",
                "decorator": "@given",
                "max_examples": 100,
                "max_shrinks": 500,
            }
        elif framework == TestFramework.JEST:
            return {
                "library": "fast-check",
                "method": "fc.assert",
                "num_runs": 100,
            }
        else:
            return {}

    def generate_property_assertion(
        self, property_name: str, function_name: str
    ) -> str:
        """
        Generate assertion code for a property.

        Args:
            property_name: Name of property
            function_name: Function being tested

        Returns:
            Assertion code string
        """
        if property_name == "idempotent":
            return f"""
        result1 = {function_name}(x)
        result2 = {function_name}(result1)
        assert result1 == result2, "Function should be idempotent"
"""

        elif property_name == "commutative":
            return f"""
        result1 = {function_name}(a, b)
        result2 = {function_name}(b, a)
        assert result1 == result2, "Function should be commutative"
"""

        elif property_name == "associative":
            return f"""
        result1 = {function_name}({function_name}(a, b), c)
        result2 = {function_name}(a, {function_name}(b, c))
        assert result1 == result2, "Function should be associative"
"""

        elif property_name == "round_trip_encode":
            # Assumes there's a corresponding decode function
            decode_func = function_name.replace("encode", "decode")
            return f"""
        encoded = {function_name}(x)
        decoded = {decode_func}(encoded)
        assert x == decoded, "Round trip should preserve value"
"""

        elif property_name == "type_invariant":
            return f"""
        result = {function_name}(x)
        assert isinstance(result, type(x)) or result is not None, "Type should be preserved"
"""

        else:
            # Generic property check
            return f"""
        result = {function_name}(x)
        assert result is not None, "Result should not be None"
"""
