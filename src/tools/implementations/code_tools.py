"""Code generation and manipulation tools."""

import logging
from datetime import datetime

from ..base import BaseTool
from ...models.tool_models import (
    ToolCategory,
    ToolSchema,
    ToolParameter,
    ToolResult,
)

logger = logging.getLogger(__name__)


class GenerateCodeTool(BaseTool):
    """
    Tool for generating code using LLM.

    PATTERN: Tool with LLM integration
    GOTCHA: Requires LLM service dependency
    """

    def __init__(self, llm_service=None):
        """
        Initialize code generation tool.

        Args:
            llm_service: Optional LLM service for generation
        """
        super().__init__(
            name="generate_code",
            category=ToolCategory.CODE_GENERATION,
        )
        self.llm_service = llm_service

    def _build_schema(self) -> ToolSchema:
        """Build tool schema."""
        return ToolSchema(
            name=self.name,
            description="Generate code based on description",
            category=self.category,
            parameters=[
                ToolParameter(
                    name="description",
                    type="string",
                    description="Description of code to generate",
                    required=True,
                ),
                ToolParameter(
                    name="language",
                    type="string",
                    description="Programming language",
                    required=True,
                    enum=["python", "javascript", "typescript", "java", "go"],
                ),
                ToolParameter(
                    name="context",
                    type="string",
                    description="Additional context",
                    required=False,
                    default="",
                ),
            ],
            returns="Generated code",
            timeout_seconds=300,  # 5 minutes for slow local LLMs
            max_retries=2,
        )

    async def execute(
        self, description: str, language: str, context: str = "", **kwargs
    ) -> ToolResult:
        """
        Execute code generation.

        Args:
            description: Code description
            language: Programming language
            context: Additional context

        Returns:
            ToolResult with generated code
        """
        start_time = datetime.now()

        try:
            # Use LLM service if available
            if self.llm_service:
                code = await self._generate_with_llm(description, language, context)
            else:
                # Fallback to template if no LLM service
                self.logger.warning("No LLM service available, using template fallback")
                code = self._get_template(language, description)

            execution_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )

            return self._create_success_result(
                result={
                    "code": code,
                    "language": language,
                    "description": description,
                    "used_llm": self.llm_service is not None,
                },
                execution_time_ms=execution_time_ms,
            )

        except Exception as e:
            execution_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )
            return self._create_error_result(
                error=f"Code generation failed: {str(e)}",
                execution_time_ms=execution_time_ms,
            )

    async def _generate_with_llm(
        self, description: str, language: str, context: str
    ) -> str:
        """
        Generate code using LLM service.

        Args:
            description: What to implement
            language: Programming language
            context: Additional context

        Returns:
            Generated code
        """
        from ...models.llm_models import LLMRequest

        # Build prompt for code generation
        prompt = f"""Generate production-ready {language} code for the following task:

Task: {description}

Context: {context}

Requirements:
- Write complete, working code (not stubs or placeholders)
- Include proper error handling
- Add docstrings/comments
- Follow {language} best practices and idioms
- Make it production-ready

Return ONLY the code, no explanations or markdown formatting."""

        # Call LLM service using LLMRequest
        messages = [{"role": "user", "content": prompt}]

        request = LLMRequest(
            messages=messages,
            agent_role="code_generator",
            task_description=f"Generate {language} code: {description}",
            temperature=0.3,  # Lower temperature for more deterministic code
            max_tokens=2000,
            use_cache=True,
        )

        response = await self.llm_service.generate_response(request)

        code = response.content.strip()

        # Remove markdown code blocks if present
        if code.startswith("```"):
            # Remove first line (```python or similar)
            lines = code.split("\n")
            code = "\n".join(lines[1:-1]) if len(lines) > 2 else code
            code = code.strip()

        return code

    def _get_template(self, language: str, description: str) -> str:
        """Get code template (placeholder)."""
        templates = {
            "python": f"# {description}\n\ndef main():\n    pass\n",
            "javascript": f"// {description}\n\nfunction main() {{\n  // TODO\n}}\n",
            "typescript": f"// {description}\n\nfunction main(): void {{\n  // TODO\n}}\n",
        }
        return templates.get(language, f"// {description}\n")


class RefactorCodeTool(BaseTool):
    """Tool for refactoring code."""

    def __init__(self, llm_service=None):
        """Initialize refactor tool."""
        super().__init__(
            name="refactor_code",
            category=ToolCategory.CODE_GENERATION,
        )
        self.llm_service = llm_service

    def _build_schema(self) -> ToolSchema:
        """Build tool schema."""
        return ToolSchema(
            name=self.name,
            description="Refactor existing code",
            category=self.category,
            parameters=[
                ToolParameter(
                    name="code",
                    type="string",
                    description="Code to refactor",
                    required=True,
                ),
                ToolParameter(
                    name="instructions",
                    type="string",
                    description="Refactoring instructions",
                    required=True,
                ),
                ToolParameter(
                    name="language",
                    type="string",
                    description="Programming language",
                    required=True,
                ),
            ],
            returns="Refactored code",
            timeout_seconds=60,
        )

    async def execute(
        self, code: str, instructions: str, language: str, **kwargs
    ) -> ToolResult:
        """Execute code refactoring using LLM."""
        start_time = datetime.now()

        try:
            # Use LLM service if available
            if self.llm_service:
                refactored = await self._refactor_with_llm(code, instructions, language)
            else:
                # Fallback: return original code
                self.logger.warning("No LLM service available, returning original code")
                refactored = code

            execution_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )

            return self._create_success_result(
                result={
                    "code": refactored,
                    "language": language,
                    "instructions": instructions,
                    "used_llm": self.llm_service is not None,
                },
                execution_time_ms=execution_time_ms,
            )

        except Exception as e:
            execution_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )
            return self._create_error_result(
                error=f"Refactoring failed: {str(e)}",
                execution_time_ms=execution_time_ms,
            )

    async def _refactor_with_llm(
        self, code: str, instructions: str, language: str
    ) -> str:
        """
        Refactor code using LLM service.

        Args:
            code: Original code
            instructions: Refactoring instructions
            language: Programming language

        Returns:
            Refactored code
        """
        from ...models.llm_models import LLMRequest

        prompt = f"""Refactor the following {language} code according to these instructions:

Instructions: {instructions}

Original Code:
```{language}
{code}
```

Requirements:
- Apply the refactoring instructions
- Maintain functionality
- Improve code quality
- Follow {language} best practices

Return ONLY the refactored code, no explanations."""

        messages = [{"role": "user", "content": prompt}]

        request = LLMRequest(
            messages=messages,
            agent_role="code_refactorer",
            task_description=f"Refactor {language} code: {instructions}",
            temperature=0.3,
            max_tokens=2000,
            use_cache=True,
        )

        response = await self.llm_service.generate_response(request)

        refactored = response.content.strip()

        # Remove markdown code blocks if present
        if refactored.startswith("```"):
            lines = refactored.split("\n")
            refactored = "\n".join(lines[1:-1]) if len(lines) > 2 else refactored
            refactored = refactored.strip()

        return refactored


class AddTestsTool(BaseTool):
    """Tool for adding tests to code."""

    def __init__(self, llm_service=None):
        """Initialize add tests tool."""
        super().__init__(
            name="add_tests",
            category=ToolCategory.CODE_GENERATION,
        )
        self.llm_service = llm_service

    def _build_schema(self) -> ToolSchema:
        """Build tool schema."""
        return ToolSchema(
            name=self.name,
            description="Generate tests for code",
            category=self.category,
            parameters=[
                ToolParameter(
                    name="code",
                    type="string",
                    description="Code to test",
                    required=True,
                ),
                ToolParameter(
                    name="language",
                    type="string",
                    description="Programming language",
                    required=True,
                ),
                ToolParameter(
                    name="test_framework",
                    type="string",
                    description="Testing framework",
                    required=False,
                    default="pytest",
                ),
            ],
            returns="Generated test code",
            timeout_seconds=60,
        )

    async def execute(
        self, code: str, language: str, test_framework: str = "pytest", **kwargs
    ) -> ToolResult:
        """Execute test generation using LLM."""
        start_time = datetime.now()

        try:
            # Use LLM service if available
            if self.llm_service:
                tests = await self._generate_tests_with_llm(code, language, test_framework)
            else:
                # Fallback to minimal test
                self.logger.warning("No LLM service available, using minimal test template")
                tests = f"# Tests for code\nimport {test_framework}\n\ndef test_example():\n    pass\n"

            execution_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )

            return self._create_success_result(
                result={
                    "tests": tests,
                    "language": language,
                    "framework": test_framework,
                    "used_llm": self.llm_service is not None,
                },
                execution_time_ms=execution_time_ms,
            )

        except Exception as e:
            execution_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )
            return self._create_error_result(
                error=f"Test generation failed: {str(e)}",
                execution_time_ms=execution_time_ms,
            )

    async def _generate_tests_with_llm(
        self, code: str, language: str, test_framework: str
    ) -> str:
        """
        Generate tests using LLM service.

        Args:
            code: Code to test
            language: Programming language
            test_framework: Testing framework

        Returns:
            Generated test code
        """
        from ...models.llm_models import LLMRequest

        prompt = f"""Generate comprehensive unit tests for the following {language} code using {test_framework}:

Code to test:
```{language}
{code}
```

Requirements:
- Use {test_framework} framework
- Test all functions/methods
- Include edge cases and error handling tests
- Write clear, descriptive test names
- Make tests production-ready

Return ONLY the test code, no explanations."""

        messages = [{"role": "user", "content": prompt}]

        request = LLMRequest(
            messages=messages,
            agent_role="test_generator",
            task_description=f"Generate {test_framework} tests for {language} code",
            temperature=0.3,
            max_tokens=2000,
            use_cache=True,
        )

        response = await self.llm_service.generate_response(request)

        tests = response.content.strip()

        # Remove markdown code blocks if present
        if tests.startswith("```"):
            lines = tests.split("\n")
            tests = "\n".join(lines[1:-1]) if len(lines) > 2 else tests
            tests = tests.strip()

        return tests
