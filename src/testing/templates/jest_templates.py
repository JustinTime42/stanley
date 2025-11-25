"""Jest test templates."""

# Jest unit test template
JEST_UNIT_TEMPLATE = """import {{ {imports} }} from '{module_path}';

describe('{suite_name}', () => {{
{setup_code}

{teardown_code}

    it('{test_description}', () => {{
        // Arrange
{arrange_code}

        // Act
{act_code}

        // Assert
{assertions}
    }});
}});
"""

# Jest async test template
JEST_ASYNC_TEMPLATE = """import {{ {imports} }} from '{module_path}';

describe('{suite_name}', () => {{
    it('{test_description}', async () => {{
        // Arrange
{arrange_code}

        // Act
{act_code}

        // Assert
{assertions}
    }});
}});
"""

# Jest mock template
JEST_MOCK_TEMPLATE = """import {{ {imports} }} from '{module_path}';

// Mock {mock_target}
jest.mock('{mock_target}', () => ({{
{mock_implementation}
}}));

describe('{suite_name}', () => {{
    it('{test_description}', () => {{
        // Arrange
{arrange_code}

        // Act
{act_code}

        // Assert
{assertions}
        // Verify mock calls
{mock_assertions}
    }});
}});
"""

# Jest spy template
JEST_SPY_TEMPLATE = """import {{ {imports} }} from '{module_path}';

describe('{suite_name}', () => {{
    it('{test_description}', () => {{
        // Arrange
        const spy{spy_name} = jest.spyOn({spy_target}, '{spy_method}');
{arrange_code}

        // Act
{act_code}

        // Assert
{assertions}
        // Verify spy
        expect(spy{spy_name}).toHaveBeenCalledWith({expected_args});

        // Cleanup
        spy{spy_name}.mockRestore();
    }});
}});
"""

# Jest integration test template
JEST_INTEGRATION_TEMPLATE = """import {{ {imports} }} from '{module_path}';

describe('{suite_name} Integration', () => {{
    beforeAll(async () => {{
{setup_code}
    }});

    afterAll(async () => {{
{teardown_code}
    }});

    it('{test_description}', async () => {{
        // Arrange
{arrange_code}

        // Act
{act_code}

        // Assert
{assertions}
    }});
}});
"""

# Jest parametrized test template (test.each)
JEST_PARAMETRIZED_TEMPLATE = """import {{ {imports} }} from '{module_path}';

describe('{suite_name}', () => {{
    it.each([
{test_cases}
    ])('{test_description}', ({param_names}) => {{
        // Arrange
{arrange_code}

        // Act
{act_code}

        // Assert
{assertions}
    }});
}});
"""

# Jest property-based test template (fast-check)
JEST_PROPERTY_TEMPLATE = """import {{ {imports} }} from '{module_path}';
import fc from 'fast-check';

describe('{suite_name} Properties', () => {{
    it('{property_description}', () => {{
        fc.assert(
            fc.property(
{strategies}
                ({param_names}) => {{
                    // Act
                    const result = {function_call};

                    // Assert property
{property_assertions}
                }}
            )
        );
    }});
}});
"""

# Jest snapshot test template
JEST_SNAPSHOT_TEMPLATE = """import {{ {imports} }} from '{module_path}';

describe('{suite_name} Snapshots', () => {{
    it('{test_description}', () => {{
        // Arrange
{arrange_code}

        // Act
{act_code}

        // Assert
        expect(result).toMatchSnapshot();
    }});
}});
"""

# Jest edge case template
JEST_EDGE_CASE_TEMPLATE = """
    it('{test_description} - edge case: {case_name}', () => {{
        // Arrange edge case
{arrange_code}

        // Act
{act_code}

        // Assert
{assertions}
    }});
"""
