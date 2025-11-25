"""Mock and stub generation for tests."""

import logging
from typing import Dict, Any, List, Optional

from ..models.testing_models import MockSpecification, MockType, TestFramework
from ..models.analysis_models import CodeEntity

logger = logging.getLogger(__name__)


class MockGenerator:
    """
    Generate appropriate test doubles (mocks, stubs, spies, fakes).

    PATTERN: Framework-specific mock syntax
    CRITICAL: Respect dependency interfaces
    GOTCHA: Handle circular dependencies
    """

    def __init__(self):
        """Initialize mock generator."""
        self.logger = logger

    async def generate_mock(
        self,
        dependency: str,
        mock_type: MockType,
        framework: TestFramework,
        interface: Optional[Dict[str, Any]] = None,
    ) -> MockSpecification:
        """
        Generate mock for dependency.

        PATTERN: Framework-specific mock syntax

        Args:
            dependency: Name of dependency to mock
            mock_type: Type of test double
            framework: Testing framework
            interface: Optional interface specification

        Returns:
            MockSpecification with framework-specific syntax
        """
        # Extract or create interface if not provided
        if interface is None:
            interface = await self._extract_interface(dependency)

        # Generate framework-specific mock
        if framework == TestFramework.PYTEST:
            return await self._generate_pytest_mock(dependency, interface, mock_type)
        elif framework == TestFramework.JEST:
            return await self._generate_jest_mock(dependency, interface, mock_type)
        elif framework == TestFramework.JUNIT:
            return await self._generate_junit_mock(dependency, interface, mock_type)
        else:
            # Default to pytest-style
            return await self._generate_pytest_mock(dependency, interface, mock_type)

    async def _extract_interface(self, dependency: str) -> Dict[str, Any]:
        """
        Extract interface from dependency.

        CRITICAL: Analyze dependency to understand methods and signatures

        Args:
            dependency: Dependency identifier

        Returns:
            Interface dictionary
        """
        # Simplified interface extraction
        # In production, would use AST analysis to extract actual interface

        interface = {
            "name": dependency,
            "methods": [],
            "default_return": None,
            "is_class": True,
            "is_function": False,
        }

        # Common return types by naming convention
        if "service" in dependency.lower():
            interface["default_return"] = "Mock()"
        elif "repository" in dependency.lower() or "dao" in dependency.lower():
            interface["default_return"] = "{}"
        elif "client" in dependency.lower():
            interface["default_return"] = "Mock()"
        else:
            interface["default_return"] = "None"

        return interface

    async def _generate_pytest_mock(
        self, dependency: str, interface: Dict[str, Any], mock_type: MockType
    ) -> MockSpecification:
        """
        Generate pytest mock.

        PATTERN: pytest-mock or unittest.mock

        Args:
            dependency: Dependency to mock
            interface: Interface specification
            mock_type: Type of test double

        Returns:
            MockSpecification with pytest syntax
        """
        default_return = interface.get("default_return", "None")
        dependency_name = dependency.split(".")[-1].lower()

        if mock_type == MockType.MOCK:
            # Full mock with expectations
            syntax = f"""
@pytest.fixture
def mock_{dependency_name}(mocker):
    \"\"\"Mock {dependency}.\"\"\"
    mock = mocker.patch('{dependency}')
    mock.return_value = {default_return}
    return mock
"""

        elif mock_type == MockType.STUB:
            # Simple stub
            syntax = f"""
@pytest.fixture
def stub_{dependency_name}():
    \"\"\"Stub {dependency}.\"\"\"
    class Stub{dependency.split('.')[-1]}:
        def __call__(self, *args, **kwargs):
            return {default_return}
    return Stub{dependency.split('.')[-1]}()
"""

        elif mock_type == MockType.SPY:
            # Spy that records calls
            syntax = f"""
@pytest.fixture
def spy_{dependency_name}(mocker):
    \"\"\"Spy on {dependency}.\"\"\"
    spy = mocker.spy({dependency.rsplit('.', 1)[0]}, '{dependency.split('.')[-1]}')
    return spy
"""

        elif mock_type == MockType.FAKE:
            # Working fake implementation
            syntax = f"""
@pytest.fixture
def fake_{dependency_name}():
    \"\"\"Fake {dependency}.\"\"\"
    class Fake{dependency.split('.')[-1]}:
        def __init__(self):
            self.calls = []

        def __call__(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return {default_return}

    return Fake{dependency.split('.')[-1]}()
"""

        else:  # DUMMY
            # Placeholder
            syntax = f"""
@pytest.fixture
def dummy_{dependency_name}():
    \"\"\"Dummy {dependency}.\"\"\"
    return None
"""

        return MockSpecification(
            target=dependency,
            mock_type=mock_type,
            framework_syntax=syntax,
            return_value=default_return,
        )

    async def _generate_jest_mock(
        self, dependency: str, interface: Dict[str, Any], mock_type: MockType
    ) -> MockSpecification:
        """
        Generate Jest mock.

        PATTERN: jest.mock() and jest.fn()

        Args:
            dependency: Dependency to mock
            interface: Interface specification
            mock_type: Type of test double

        Returns:
            MockSpecification with Jest syntax
        """
        default_return = interface.get("default_return", "null")
        dependency_name = dependency.split("/")[-1].replace(".js", "").replace(".ts", "")

        if mock_type == MockType.MOCK:
            # Jest mock
            syntax = f"""
jest.mock('{dependency}', () => ({{
    default: jest.fn().mockReturnValue({default_return}),
}}));
"""

        elif mock_type == MockType.STUB:
            # Simple stub
            syntax = f"""
const stub{dependency_name.title()} = () => {default_return};
"""

        elif mock_type == MockType.SPY:
            # Jest spy
            syntax = f"""
const spy{dependency_name.title()} = jest.spyOn(require('{dependency}'), 'default');
"""

        elif mock_type == MockType.FAKE:
            # Fake implementation
            syntax = f"""
const fake{dependency_name.title()} = {{
    calls: [],
    execute: function(...args) {{
        this.calls.push(args);
        return {default_return};
    }}
}};
"""

        else:  # DUMMY
            # Placeholder
            syntax = f"""
const dummy{dependency_name.title()} = null;
"""

        return MockSpecification(
            target=dependency,
            mock_type=mock_type,
            framework_syntax=syntax,
            return_value=default_return,
        )

    async def _generate_junit_mock(
        self, dependency: str, interface: Dict[str, Any], mock_type: MockType
    ) -> MockSpecification:
        """
        Generate JUnit mock (Mockito).

        Args:
            dependency: Dependency to mock
            interface: Interface specification
            mock_type: Type of test double

        Returns:
            MockSpecification with Mockito syntax
        """
        class_name = dependency.split(".")[-1]
        default_return = interface.get("default_return", "null")

        if mock_type == MockType.MOCK:
            # Mockito mock
            syntax = f"""
@Mock
private {class_name} mock{class_name};

// In test method:
when(mock{class_name}.execute(any())).thenReturn({default_return});
"""

        elif mock_type == MockType.STUB:
            # Mockito stub
            syntax = f"""
private {class_name} stub{class_name} = new {class_name}() {{
    @Override
    public Object execute(Object... args) {{
        return {default_return};
    }}
}};
"""

        elif mock_type == MockType.SPY:
            # Mockito spy
            syntax = f"""
@Spy
private {class_name} spy{class_name};
"""

        else:
            # Default mock
            syntax = f"""
@Mock
private {class_name} mock{class_name};
"""

        return MockSpecification(
            target=dependency,
            mock_type=mock_type,
            framework_syntax=syntax,
            return_value=default_return,
        )

    async def generate_mocks_for_function(
        self,
        function: CodeEntity,
        framework: TestFramework,
        mock_all: bool = False,
    ) -> List[MockSpecification]:
        """
        Generate mocks for all dependencies of a function.

        Args:
            function: Function to generate mocks for
            framework: Testing framework
            mock_all: Whether to mock all dependencies or be selective

        Returns:
            List of mock specifications
        """
        mocks = []

        # Get function dependencies
        dependencies = function.dependencies or []

        for dependency in dependencies:
            # Decide if we should mock this dependency
            should_mock = mock_all or await self._should_mock_dependency(dependency)

            if should_mock:
                # Determine mock type
                mock_type = await self._determine_mock_type(dependency)

                # Generate mock
                mock_spec = await self.generate_mock(
                    dependency=dependency,
                    mock_type=mock_type,
                    framework=framework,
                )
                mocks.append(mock_spec)

        return mocks

    async def _should_mock_dependency(self, dependency: str) -> bool:
        """
        Determine if dependency should be mocked.

        PATTERN: Mock external services, databases, network calls
        GOTCHA: Don't mock simple utilities or pure functions

        Args:
            dependency: Dependency name

        Returns:
            True if should be mocked
        """
        # Mock if it's an external service
        external_indicators = [
            "service",
            "client",
            "repository",
            "dao",
            "api",
            "http",
            "database",
            "db",
            "cache",
            "queue",
        ]

        dependency_lower = dependency.lower()
        for indicator in external_indicators:
            if indicator in dependency_lower:
                return True

        # Don't mock utilities or built-ins
        utility_indicators = ["util", "helper", "math", "string", "array"]
        for indicator in utility_indicators:
            if indicator in dependency_lower:
                return False

        # Default: mock if it's not a standard library
        return not dependency.startswith(("builtins.", "std.", "java.lang.", "System."))

    async def _determine_mock_type(self, dependency: str) -> MockType:
        """
        Determine appropriate mock type for dependency.

        Args:
            dependency: Dependency name

        Returns:
            Recommended MockType
        """
        dependency_lower = dependency.lower()

        # Use spy for logging/monitoring
        if "logger" in dependency_lower or "monitor" in dependency_lower:
            return MockType.SPY

        # Use fake for simple data structures
        if "cache" in dependency_lower or "store" in dependency_lower:
            return MockType.FAKE

        # Use stub for simple dependencies
        if "config" in dependency_lower or "settings" in dependency_lower:
            return MockType.STUB

        # Default to mock
        return MockType.MOCK
