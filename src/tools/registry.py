"""Tool registry for discovery and management."""

import logging
from typing import Dict, List, Optional

from .base import BaseTool
from ..models.tool_models import ToolCategory, ToolSchema

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Tool registry for discovery and management.

    PATTERN: Singleton registry pattern
    CRITICAL: Thread-safe registration and retrieval
    GOTCHA: Tools must be registered before use
    """

    _instance: Optional["ToolRegistry"] = None
    _tools: Dict[str, BaseTool] = {}

    def __new__(cls):
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._tools = {}
        return cls._instance

    def register_tool(self, tool: BaseTool) -> None:
        """
        Register a tool.

        Args:
            tool: Tool instance to register

        Raises:
            ValueError: If tool name already registered
        """
        if tool.name in self._tools:
            raise ValueError(
                f"Tool '{tool.name}' is already registered. "
                "Use a unique name for each tool."
            )

        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name} (category: {tool.category.value})")

    def unregister_tool(self, tool_name: str) -> None:
        """
        Unregister a tool.

        Args:
            tool_name: Name of tool to unregister
        """
        if tool_name in self._tools:
            del self._tools[tool_name]
            logger.info(f"Unregistered tool: {tool_name}")
        else:
            logger.warning(f"Tool '{tool_name}' not found in registry")

    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """
        Get a tool by name.

        Args:
            tool_name: Name of tool to retrieve

        Returns:
            Tool instance or None if not found
        """
        return self._tools.get(tool_name)

    def list_tools(self, category: Optional[ToolCategory] = None) -> List[BaseTool]:
        """
        List all registered tools.

        Args:
            category: Optional category filter

        Returns:
            List of tools
        """
        tools = list(self._tools.values())

        if category:
            tools = [t for t in tools if t.category == category]

        return tools

    def list_tool_names(self, category: Optional[ToolCategory] = None) -> List[str]:
        """
        List all registered tool names.

        Args:
            category: Optional category filter

        Returns:
            List of tool names
        """
        return [tool.name for tool in self.list_tools(category)]

    def get_tool_schemas(
        self, category: Optional[ToolCategory] = None
    ) -> List[ToolSchema]:
        """
        Get schemas for all tools.

        Args:
            category: Optional category filter

        Returns:
            List of tool schemas
        """
        tools = self.list_tools(category)
        return [tool.get_schema() for tool in tools]

    def get_tool_count(self) -> int:
        """
        Get total number of registered tools.

        Returns:
            Number of tools
        """
        return len(self._tools)

    def clear(self) -> None:
        """Clear all registered tools (for testing)."""
        self._tools.clear()
        logger.warning("Cleared all tools from registry")

    def has_tool(self, tool_name: str) -> bool:
        """
        Check if a tool is registered.

        Args:
            tool_name: Name of tool to check

        Returns:
            True if tool is registered
        """
        return tool_name in self._tools

    def get_tools_by_category(self, category: ToolCategory) -> Dict[str, BaseTool]:
        """
        Get all tools in a category.

        Args:
            category: Tool category

        Returns:
            Dictionary of tool name -> tool instance
        """
        return {
            name: tool
            for name, tool in self._tools.items()
            if tool.category == category
        }
