"""CLI session state models."""

from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List

from pydantic import BaseModel, Field


class CLIMode(str, Enum):
    """CLI operating modes."""

    CHAT = "chat"  # Single-agent conversational
    TASK = "task"  # Multi-agent autonomous workflow


class Message(BaseModel):
    """Chat message model."""

    role: str = Field(description="Message role: user, assistant, or system")
    content: str = Field(description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for LLM API."""
        return {"role": self.role, "content": self.content}

    class Config:
        """Pydantic configuration."""

        json_encoders = {datetime: lambda v: v.isoformat()}


class SessionState(BaseModel):
    """
    CLI session state - must be JSON-serializable.

    CRITICAL: All fields must be JSON-serializable for persistence.
    """

    session_id: str = Field(description="Unique session identifier")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    working_directory: str = Field(description="Directory session was started in")

    # Conversation state
    mode: CLIMode = Field(default=CLIMode.CHAT)
    messages: List[Message] = Field(default_factory=list)

    # Model configuration
    model: Optional[str] = Field(default=None, description="Override model")
    temperature: float = Field(default=0.7)
    system_prompt: Optional[str] = Field(
        default=None,
        description="Custom system prompt",
    )

    # Statistics
    total_tokens: int = Field(default=0)
    total_cost: float = Field(default=0.0)
    turn_count: int = Field(default=0)

    # Task mode state
    active_workflow_id: Optional[str] = Field(default=None)
    workflow_status: Optional[str] = Field(default=None)

    # Session metadata
    name: Optional[str] = Field(default=None, description="Optional session name")
    tags: List[str] = Field(default_factory=list)

    class Config:
        """Pydantic configuration."""

        use_enum_values = True
        json_encoders = {datetime: lambda v: v.isoformat()}

    def add_message(self, role: str, content: str, **metadata: Any) -> None:
        """
        Add a message to the conversation.

        Args:
            role: Message role (user, assistant, system)
            content: Message content
            **metadata: Additional metadata
        """
        message = Message(
            role=role,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata,
        )
        self.messages.append(message)
        self.updated_at = datetime.now()

        if role in ("user", "assistant"):
            self.turn_count += 1

    def clear_messages(self) -> None:
        """Clear all messages from the conversation."""
        self.messages = []
        self.turn_count = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self.updated_at = datetime.now()

    def get_messages_for_llm(self) -> List[Dict[str, str]]:
        """
        Get messages formatted for LLM API.

        Returns:
            List of message dictionaries
        """
        return [msg.to_dict() for msg in self.messages]

    def update_stats(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost: float = 0.0,
    ) -> None:
        """
        Update session statistics.

        Args:
            input_tokens: Input tokens used
            output_tokens: Output tokens used
            cost: Cost in USD
        """
        self.total_tokens += input_tokens + output_tokens
        self.total_cost += cost
        self.updated_at = datetime.now()

    def to_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the session.

        Returns:
            Summary dictionary
        """
        return {
            "session_id": self.session_id,
            "name": self.name,
            "mode": self.mode,
            "turn_count": self.turn_count,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "working_directory": self.working_directory,
        }
