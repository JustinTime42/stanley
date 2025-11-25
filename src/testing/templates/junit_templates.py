"""JUnit test templates."""

# JUnit 5 unit test template
JUNIT_UNIT_TEMPLATE = """package {package_name};

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.AfterEach;
import static org.junit.jupiter.api.Assertions.*;
{imports}

public class {class_name}Test {{

    private {class_name} {instance_name};

    @BeforeEach
    void setUp() {{
{setup_code}
    }}

    @AfterEach
    void tearDown() {{
{teardown_code}
    }}

    @Test
    void test{test_name}() {{
        // Arrange
{arrange_code}

        // Act
{act_code}

        // Assert
{assertions}
    }}
}}
"""

# JUnit parametrized test template
JUNIT_PARAMETRIZED_TEMPLATE = """
    @ParameterizedTest
    @ValueSource({value_type} = {{ {values} }})
    void test{test_name}Parametrized({param_type} {param_name}) {{
        // Arrange
{arrange_code}

        // Act
{act_code}

        // Assert
{assertions}
    }}
"""

# JUnit mock template (Mockito)
JUNIT_MOCK_TEMPLATE = """package {package_name};

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
{imports}

@ExtendWith(MockitoExtension.class)
public class {class_name}Test {{

    @Mock
    private {mock_type} {mock_name};

    @Test
    void test{test_name}WithMock() {{
        // Arrange
        when({mock_name}.{mock_method}({mock_params})).thenReturn({return_value});

        // Act
{act_code}

        // Assert
{assertions}
        verify({mock_name}).{mock_method}({expected_params});
    }}
}}
"""
