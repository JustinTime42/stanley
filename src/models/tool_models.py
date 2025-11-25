"""Tool-related data models for the tool abstraction layer."""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from enum import Enum
from datetime import datetime


class ToolCategory(str, Enum):
    """Tool categories for organization and discovery."""

    FILE_SYSTEM = "file_system"
    CODE_GENERATION = "code_generation"
    VERSION_CONTROL = "version_control"
    TESTING = "testing"
    VALIDATION = "validation"
    BUILD = "build"
    CONTAINER = "container"
    EXTERNAL_API = "external_api"
    SEARCH = "search"
    ANALYSIS = "analysis"


class ToolStatus(str, Enum):
    """Tool execution status."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    RETRYING = "retrying"


class ToolParameter(BaseModel):
    """Tool parameter definition for schema."""

    name: str = Field(description="Parameter name")
    type: str = Field(description="Parameter type (string, integer, etc.)")
    description: str = Field(description="Parameter description")
    required: bool = Field(default=True)
    default: Optional[Any] = Field(default=None)
    enum: Optional[List[Any]] = Field(default=None, description="Allowed values")


class ToolSchema(BaseModel):
    """Tool schema for function calling and validation."""

    name: str = Field(description="Tool name")
    description: str = Field(description="Tool description")
    category: ToolCategory
    parameters: List[ToolParameter] = Field(default_factory=list)
    returns: str = Field(description="Return type description")
    examples: List[Dict[str, Any]] = Field(default_factory=list)
    timeout_seconds: int = Field(default=30)
    max_retries: int = Field(default=3)
    requires_confirmation: bool = Field(default=False)


class ToolRequest(BaseModel):
    """Request to execute a tool."""

    tool_name: str = Field(description="Name of tool to execute")
    parameters: Dict[str, Any] = Field(default_factory=dict)
    agent_id: str = Field(description="Agent making the request")
    workflow_id: str = Field(description="Workflow context")
    timeout_override: Optional[int] = Field(default=None)
    async_execution: bool = Field(default=False)
    priority: int = Field(default=5, ge=1, le=10)


class ToolResult(BaseModel):
    """Result from tool execution."""

    tool_name: str
    status: ToolStatus
    result: Optional[Any] = Field(default=None, description="Tool output")
    error: Optional[str] = Field(default=None)
    execution_time_ms: int = Field(description="Execution duration")
    resource_usage: Dict[str, float] = Field(default_factory=dict)
    retry_count: int = Field(default=0)
    timestamp: datetime = Field(default_factory=datetime.now)
