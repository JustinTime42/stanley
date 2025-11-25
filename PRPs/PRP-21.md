# PRP-21: Interactive CLI Interface

## Goal

**Feature Goal**: Implement a Claude Code-style interactive CLI interface with REPL functionality, session management, slash commands, and streaming output that provides a conversational interface to the agent swarm system.

**Deliverable**: Full-featured CLI application supporting both chat mode (single-agent conversation) and task mode (multi-agent autonomous workflows), with session persistence, rich terminal output, and extensible command system.

**Success Definition**: 
- Interactive REPL with streaming responses and rich formatting
- Session persistence with continue/resume capabilities
- Built-in slash commands (/help, /clear, /cost, /model, /config, /memory, /task)
- Custom command support via markdown files
- Sub-200ms startup time, real-time streaming display
- Seamless integration with existing LLM, Tool, and Memory services

## Why

- Current system requires writing Python scripts to execute tasks - high friction for daily use
- No conversational interface for iterative development and exploration
- Cannot easily switch between quick questions and complex autonomous tasks
- No session continuity - each run starts fresh without context
- Missing the intuitive UX that makes tools like Claude Code accessible
- Power users need keyboard shortcuts, vim mode, and customization

## What

Implement an interactive CLI that serves as the primary user interface for the agent swarm, supporting both conversational chat and autonomous task execution modes.

### Success Criteria

- [ ] REPL starts in <200ms and handles Ctrl+C/Ctrl+D gracefully
- [ ] Streaming responses display in real-time with proper formatting
- [ ] Sessions persist and can be resumed with `--continue` and `--resume <id>`
- [ ] All built-in slash commands functional (/help, /clear, /cost, /model, /config, /task, /chat)
- [ ] Custom commands loadable from `.agent-swarm/commands/*.md`
- [ ] Multi-line input works via `\` + Enter and paste detection
- [ ] Command history persists across sessions
- [ ] Rich output with syntax highlighting for code blocks
- [ ] Both chat mode and task mode fully operational

## All Needed Context

### Context Completeness Check

_If someone knew nothing about this codebase, would they have everything needed to implement this successfully?_ YES - This PRP includes complete CLI patterns, integration points with existing services, and reference implementations.

### Documentation & References

```yaml
- file: PRPs/ai_docs/cc_cli.md
  why: Claude Code CLI reference - target UX patterns
  pattern: Command structure, flags, keyboard shortcuts
  critical: Follow established conventions for familiarity

- file: PRPs/ai_docs/cc_commands.md
  why: Slash command system design reference
  pattern: Built-in commands, custom commands, frontmatter
  critical: Support namespaced commands and argument passing

- file: src/services/llm_service.py
  why: LLM orchestrator with streaming support
  pattern: stream_response() method for real-time output
  critical: Use existing routing and caching

- file: src/services/workflow_service.py
  why: WorkflowOrchestrator for task mode
  pattern: start_workflow(), resume_workflow()
  critical: Integrate existing multi-agent system

- file: src/services/checkpoint_service.py
  why: Session persistence infrastructure
  pattern: CheckpointManager for state storage
  critical: Reuse for CLI session management

- url: https://python-prompt-toolkit.readthedocs.io/
  why: Primary library for REPL implementation
  critical: Async support, history, key bindings, vim mode

- url: https://rich.readthedocs.io/
  why: Rich terminal output and formatting
  critical: Markdown rendering, syntax highlighting, Live display
```

### Current Codebase Tree (Relevant Sections)

```bash
agent-swarm/
├── src/
│   ├── services/
│   │   ├── llm_service.py       # LLM with streaming
│   │   ├── tool_service.py      # Tool orchestration
│   │   ├── memory_service.py    # Memory system
│   │   ├── workflow_service.py  # Multi-agent workflows
│   │   └── checkpoint_service.py # State persistence
│   ├── models/
│   │   ├── llm_models.py        # LLMRequest, LLMResponse
│   │   └── workflow_models.py   # WorkflowConfig
│   └── config/
│       ├── llm_config.py        # LLM configuration
│       └── memory_config.py     # Memory configuration
├── PRPs/
│   └── ai_docs/
│       ├── cc_cli.md            # Claude Code CLI reference
│       └── cc_commands.md       # Slash commands reference
└── requirements.txt
```

### Desired Codebase Tree with Files to be Added

```bash
agent-swarm/
├── src/
│   ├── cli/                      # NEW: CLI subsystem
│   │   ├── __init__.py          # CLI entry point exports
│   │   ├── app.py               # Main CLI application class
│   │   ├── repl.py              # REPL loop implementation
│   │   ├── modes/
│   │   │   ├── __init__.py
│   │   │   ├── base.py          # Base mode class
│   │   │   ├── chat.py          # Chat mode (single-agent)
│   │   │   └── task.py          # Task mode (multi-agent workflow)
│   │   ├── commands/
│   │   │   ├── __init__.py      # Command registry
│   │   │   ├── base.py          # BaseCommand class
│   │   │   ├── builtin.py       # Built-in commands
│   │   │   ├── config_cmd.py    # /config command
│   │   │   ├── memory_cmd.py    # /memory command
│   │   │   └── loader.py        # Custom command loader
│   │   ├── input/
│   │   │   ├── __init__.py
│   │   │   ├── parser.py        # Input parsing (message vs command)
│   │   │   ├── multiline.py     # Multi-line input handling
│   │   │   └── completer.py     # Tab completion
│   │   ├── output/
│   │   │   ├── __init__.py
│   │   │   ├── renderer.py      # Rich output rendering
│   │   │   ├── streaming.py     # Streaming display
│   │   │   └── themes.py        # Color themes
│   │   ├── session/
│   │   │   ├── __init__.py
│   │   │   ├── manager.py       # Session persistence
│   │   │   ├── history.py       # Command history
│   │   │   └── state.py         # Session state model
│   │   └── config/
│   │       ├── __init__.py
│   │       └── cli_config.py    # CLI-specific configuration
│   └── tests/
│       └── cli/                  # NEW: CLI tests
│           ├── test_repl.py
│           ├── test_commands.py
│           ├── test_session.py
│           └── test_modes.py
├── cli.py                        # NEW: CLI entry point script
├── .agent-swarm/                 # NEW: User config directory
│   ├── config.yaml              # User configuration
│   ├── commands/                # Custom commands
│   │   └── example.md
│   └── sessions/                # Session storage (SQLite)
└── requirements.txt              # MODIFY: Add prompt_toolkit, rich
```

### Known Gotchas of our Codebase & Library Quirks

```python
# CRITICAL: prompt_toolkit requires async for non-blocking REPL
# Must use prompt_toolkit.PromptSession with prompt_async()
session = PromptSession()
result = await session.prompt_async("You: ")  # NOT session.prompt()

# CRITICAL: Rich Live display conflicts with prompt_toolkit input
# Must stop Live context before prompting for input
with Live(spinner) as live:
    # ... display streaming
    pass  # Live exits here
# Now safe to prompt for input

# CRITICAL: LLM streaming yields chunks, not full messages
# Must accumulate chunks for history storage
full_response = ""
async for chunk in llm_service.stream_response(request):
    full_response += chunk.content
    display(chunk.content)
messages.append({"role": "assistant", "content": full_response})

# CRITICAL: Keyboard interrupt during streaming
# Must handle Ctrl+C gracefully without crashing
try:
    async for chunk in stream:
        ...
except KeyboardInterrupt:
    # Cancel generation, keep partial response
    pass

# CRITICAL: Session state must be JSON-serializable
# Avoid storing non-serializable objects in session
session_state = {
    "messages": [...],  # List of dicts, not Message objects
    "config": {...},    # Plain dict, not Pydantic model
}

# CRITICAL: Custom commands with frontmatter
# Use python-frontmatter library, handle missing frontmatter gracefully
import frontmatter
post = frontmatter.load(command_file)
metadata = post.metadata or {}
content = post.content

# CRITICAL: Working directory awareness
# CLI should operate in current directory, not installation directory
import os
working_dir = os.getcwd()
project_commands_dir = os.path.join(working_dir, ".agent-swarm", "commands")
```

## Implementation Blueprint

### Data Models and Structure

```python
# src/cli/session/state.py
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum

class CLIMode(str, Enum):
    CHAT = "chat"      # Single-agent conversational
    TASK = "task"      # Multi-agent autonomous workflow

class Message(BaseModel):
    """Chat message model"""
    role: str = Field(description="user, assistant, or system")
    content: str = Field(description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SessionState(BaseModel):
    """CLI session state - must be JSON-serializable"""
    session_id: str = Field(description="Unique session identifier")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    working_directory: str = Field(description="Directory session was started in")
    
    # Conversation state
    mode: CLIMode = Field(default=CLIMode.CHAT)
    messages: List[Message] = Field(default_factory=list)
    
    # Configuration
    model: Optional[str] = Field(default=None, description="Override model")
    temperature: float = Field(default=0.7)
    
    # Statistics
    total_tokens: int = Field(default=0)
    total_cost: float = Field(default=0.0)
    turn_count: int = Field(default=0)
    
    # Task mode state
    active_workflow_id: Optional[str] = Field(default=None)
    workflow_status: Optional[str] = Field(default=None)

class CLIConfig(BaseModel):
    """CLI configuration"""
    # Display
    theme: str = Field(default="monokai")
    show_tokens: bool = Field(default=True)
    show_cost: bool = Field(default=True)
    stream_output: bool = Field(default=True)
    
    # Behavior
    default_mode: CLIMode = Field(default=CLIMode.CHAT)
    auto_save_sessions: bool = Field(default=True)
    max_history_size: int = Field(default=1000)
    
    # Vim mode
    vim_mode: bool = Field(default=False)
    
    # Directories
    commands_dir: str = Field(default=".agent-swarm/commands")
    sessions_dir: str = Field(default=".agent-swarm/sessions")


# src/cli/commands/base.py
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any

class BaseCommand(ABC):
    """Base class for slash commands"""
    
    name: str                           # Command name (without /)
    description: str                    # Help text
    aliases: List[str] = []            # Alternative names
    arguments: str = ""                 # Argument hint for help
    
    @abstractmethod
    async def execute(
        self,
        args: str,
        context: "CommandContext",
    ) -> Optional[str]:
        """
        Execute the command.
        
        Args:
            args: Arguments passed to command
            context: Execution context with services
            
        Returns:
            Optional response to display
        """
        pass

class CommandContext:
    """Context passed to command execution"""
    def __init__(
        self,
        session: SessionState,
        llm_service: "LLMOrchestrator",
        tool_service: "ToolOrchestrator",
        memory_service: "MemoryOrchestrator",
        workflow_service: "WorkflowOrchestrator",
        console: "Console",
        config: CLIConfig,
    ):
        self.session = session
        self.llm = llm_service
        self.tools = tool_service
        self.memory = memory_service
        self.workflow = workflow_service
        self.console = console
        self.config = config
```

### Implementation Tasks (ordered by dependencies)

```yaml
Task 1: CREATE src/cli/config/cli_config.py
  description: CLI configuration management
  depends_on: []
  estimated_time: 2 hours
  validation: |
    python -c "from src.cli.config import CLIConfig; c = CLIConfig(); print(c.model_dump())"
  implementation_notes: |
    - Load from ~/.agent-swarm/config.yaml and ./.agent-swarm/config.yaml
    - Merge with defaults (user config overrides project config)
    - Environment variable overrides (AGENT_SWARM_*)
    - Validate paths exist or create them

Task 2: CREATE src/cli/session/state.py
  description: Session state models
  depends_on: [Task 1]
  estimated_time: 1 hour
  validation: |
    python -c "from src.cli.session import SessionState; s = SessionState(session_id='test', working_directory='.'); print(s.model_dump_json())"

Task 3: CREATE src/cli/session/manager.py
  description: Session persistence and management
  depends_on: [Task 2]
  estimated_time: 4 hours
  validation: |
    pytest src/tests/cli/test_session.py -v
  implementation_notes: |
    - Use SQLite for session storage (portable, no server needed)
    - Store in ~/.agent-swarm/sessions/sessions.db
    - Support: create, save, load, list, delete sessions
    - Auto-save on each turn if configured
    - Session ID format: {timestamp}_{short_uuid}

Task 4: CREATE src/cli/session/history.py
  description: Command history management
  depends_on: [Task 3]
  estimated_time: 2 hours
  validation: |
    python -c "from src.cli.session import HistoryManager; h = HistoryManager(); print('OK')"
  implementation_notes: |
    - Integrate with prompt_toolkit FileHistory
    - Per-directory history files
    - Configurable max history size

Task 5: CREATE src/cli/commands/base.py
  description: Base command infrastructure
  depends_on: [Task 2]
  estimated_time: 2 hours
  validation: |
    python -c "from src.cli.commands import BaseCommand, CommandContext; print('OK')"

Task 6: CREATE src/cli/commands/builtin.py
  description: Built-in slash commands
  depends_on: [Task 5]
  estimated_time: 4 hours
  validation: |
    pytest src/tests/cli/test_commands.py -v
  implementation_notes: |
    Commands to implement:
    - /help [command] - Show help
    - /clear - Clear conversation
    - /cost - Show token usage and cost
    - /model [name] - Show or change model
    - /mode [chat|task] - Switch modes
    - /quit or /exit - Exit CLI
    - /sessions - List saved sessions
    - /save [name] - Save current session
    - /load <id> - Load session
    - /status - Show current status

Task 7: CREATE src/cli/commands/config_cmd.py
  description: /config command for settings
  depends_on: [Task 6]
  estimated_time: 2 hours
  validation: |
    # Manual test in REPL
  implementation_notes: |
    - /config - Show current config
    - /config set <key> <value> - Set config value
    - /config reset - Reset to defaults

Task 8: CREATE src/cli/commands/memory_cmd.py
  description: /memory command for context management
  depends_on: [Task 6]
  estimated_time: 3 hours
  validation: |
    # Manual test in REPL
  implementation_notes: |
    - /memory - Show memory stats
    - /memory search <query> - Search memories
    - /memory add <content> - Add to project memory
    - /memory clear - Clear working memory

Task 9: CREATE src/cli/commands/loader.py
  description: Custom command loader from markdown files
  depends_on: [Task 5]
  estimated_time: 3 hours
  validation: |
    pytest src/tests/cli/test_commands.py::test_custom_commands -v
  implementation_notes: |
    - Load from .agent-swarm/commands/*.md
    - Load from ~/.agent-swarm/commands/*.md
    - Support frontmatter (description, allowed-tools, model)
    - Support $ARGUMENTS placeholder
    - Support !`bash command` execution
    - Support @file references

Task 10: CREATE src/cli/input/parser.py
  description: Input parsing (message vs command)
  depends_on: []
  estimated_time: 2 hours
  validation: |
    python -c "from src.cli.input import InputParser; p = InputParser(); print(p.parse('/help'))"
  implementation_notes: |
    - Detect slash commands (starts with /)
    - Detect special prefixes (# for memory)
    - Handle multi-line input detection
    - Return ParsedInput with type and content

Task 11: CREATE src/cli/input/multiline.py
  description: Multi-line input handling
  depends_on: [Task 10]
  estimated_time: 2 hours
  implementation_notes: |
    - Detect \ at end of line
    - Detect paste (rapid input)
    - Detect code blocks (```)
    - Return complete input when ready

Task 12: CREATE src/cli/input/completer.py
  description: Tab completion
  depends_on: [Task 6, Task 9]
  estimated_time: 3 hours
  implementation_notes: |
    - Complete slash commands
    - Complete command arguments
    - Complete file paths for @ references
    - Integrate with prompt_toolkit Completer

Task 13: CREATE src/cli/output/themes.py
  description: Color themes
  depends_on: []
  estimated_time: 1 hour
  implementation_notes: |
    - Define theme colors for different elements
    - Support: monokai, dracula, github, light
    - User role, assistant role, system, error, code

Task 14: CREATE src/cli/output/renderer.py
  description: Rich output rendering
  depends_on: [Task 13]
  estimated_time: 3 hours
  validation: |
    python -c "from src.cli.output import OutputRenderer; r = OutputRenderer(); r.render_message('Hello **world**')"
  implementation_notes: |
    - Render markdown with rich.Markdown
    - Syntax highlight code blocks
    - Format tool calls and results
    - Format errors with traceback
    - Status indicators (spinners, progress)

Task 15: CREATE src/cli/output/streaming.py
  description: Streaming display
  depends_on: [Task 14]
  estimated_time: 3 hours
  implementation_notes: |
    - Display tokens as they arrive
    - Handle partial markdown (don't break formatting)
    - Show typing indicator while waiting
    - Accumulate for final render and history

Task 16: CREATE src/cli/modes/base.py
  description: Base mode class
  depends_on: [Task 2, Task 14]
  estimated_time: 2 hours
  implementation_notes: |
    - Abstract base for chat and task modes
    - Common methods: process_input, display_response
    - Mode-specific prompt prefix

Task 17: CREATE src/cli/modes/chat.py
  description: Chat mode (single-agent conversation)
  depends_on: [Task 16]
  estimated_time: 4 hours
  validation: |
    pytest src/tests/cli/test_modes.py::test_chat_mode -v
  implementation_notes: |
    - Direct conversation with LLM
    - Tool use during conversation
    - Message history management
    - System prompt configuration

Task 18: CREATE src/cli/modes/task.py
  description: Task mode (multi-agent workflow)
  depends_on: [Task 16]
  estimated_time: 4 hours
  validation: |
    pytest src/tests/cli/test_modes.py::test_task_mode -v
  implementation_notes: |
    - Integrate with WorkflowOrchestrator
    - Display agent progress
    - Support human-in-the-loop approval
    - Show workflow status and results

Task 19: CREATE src/cli/repl.py
  description: Main REPL loop
  depends_on: [Task 10, Task 11, Task 12, Task 15, Task 17, Task 18]
  estimated_time: 4 hours
  validation: |
    python -m src.cli --help
  implementation_notes: |
    - Initialize prompt_toolkit PromptSession
    - Configure key bindings (Ctrl+C, Ctrl+D, etc)
    - Main async loop with prompt_async
    - Dispatch to parser, commands, or modes
    - Handle vim mode toggle

Task 20: CREATE src/cli/app.py
  description: Main CLI application
  depends_on: [Task 19, Task 3, Task 6, Task 9]
  estimated_time: 4 hours
  validation: |
    python cli.py --help
    python cli.py
    python cli.py "Hello world"
    python cli.py -c
    python cli.py --model qwen2.5-coder:14b
  implementation_notes: |
    - CLI argument parsing with click
    - Initialize all services
    - Handle --continue, --resume, --model flags
    - One-shot mode with -p flag
    - Entry point for the CLI

Task 21: CREATE cli.py
  description: CLI entry point script
  depends_on: [Task 20]
  estimated_time: 30 minutes
  implementation_notes: |
    - Simple entry point that imports and runs app
    - Add to pyproject.toml as console script

Task 22: UPDATE requirements.txt
  description: Add CLI dependencies
  depends_on: []
  estimated_time: 15 minutes
  implementation_notes: |
    Add:
    - prompt_toolkit>=3.0.0
    - rich>=13.0.0
    - click>=8.0.0
    - python-frontmatter>=1.0.0

Task 23: CREATE src/tests/cli/
  description: CLI test suite
  depends_on: [Task 20]
  estimated_time: 4 hours
  validation: |
    pytest src/tests/cli/ -v --cov=src/cli
  implementation_notes: |
    - test_repl.py - REPL behavior tests
    - test_commands.py - Command tests
    - test_session.py - Session management tests
    - test_modes.py - Chat and task mode tests
    - Use pytest-asyncio for async tests
```

### Core Implementation Patterns

```python
# src/cli/app.py - Main application structure
import click
import asyncio
from pathlib import Path

from .repl import REPL
from .session import SessionManager, SessionState
from .config import CLIConfig, load_config
from .commands import CommandRegistry, load_builtin_commands, load_custom_commands
from .output import OutputRenderer
from ..services.llm_service import LLMOrchestrator
from ..services.tool_service import ToolOrchestrator
from ..services.memory_service import MemoryOrchestrator
from ..services.workflow_service import WorkflowOrchestrator
from ..services.checkpoint_service import CheckpointManager


@click.command()
@click.argument("prompt", required=False)
@click.option("-p", "--print", "print_mode", is_flag=True, help="Print response and exit")
@click.option("-c", "--continue", "continue_session", is_flag=True, help="Continue last session")
@click.option("-r", "--resume", "resume_id", help="Resume session by ID")
@click.option("--model", help="Override model for this session")
@click.option("--mode", type=click.Choice(["chat", "task"]), help="Start in specific mode")
@click.option("--verbose", is_flag=True, help="Enable verbose logging")
@click.option("--config", "config_path", type=click.Path(), help="Config file path")
def main(
    prompt: str,
    print_mode: bool,
    continue_session: bool,
    resume_id: str,
    model: str,
    mode: str,
    verbose: bool,
    config_path: str,
):
    """Agent Swarm CLI - Interactive AI coding assistant"""
    asyncio.run(run_cli(
        prompt=prompt,
        print_mode=print_mode,
        continue_session=continue_session,
        resume_id=resume_id,
        model_override=model,
        mode_override=mode,
        verbose=verbose,
        config_path=config_path,
    ))


async def run_cli(
    prompt: str = None,
    print_mode: bool = False,
    continue_session: bool = False,
    resume_id: str = None,
    model_override: str = None,
    mode_override: str = None,
    verbose: bool = False,
    config_path: str = None,
):
    """Async CLI runner"""
    
    # Load configuration
    config = load_config(config_path)
    
    # Initialize services
    llm_service = LLMOrchestrator()
    checkpoint_manager = CheckpointManager()
    memory_service = MemoryOrchestrator(...)
    tool_service = ToolOrchestrator(llm_service=llm_service)
    workflow_service = WorkflowOrchestrator(
        checkpoint_manager=checkpoint_manager,
        memory_service=memory_service,
        llm_service=llm_service,
        tool_service=tool_service,
    )
    
    # Initialize session manager
    session_manager = SessionManager(config)
    
    # Load or create session
    if continue_session:
        session = await session_manager.load_latest()
    elif resume_id:
        session = await session_manager.load(resume_id)
    else:
        session = SessionState(
            session_id=session_manager.generate_id(),
            working_directory=str(Path.cwd()),
        )
    
    # Apply overrides
    if model_override:
        session.model = model_override
    if mode_override:
        session.mode = CLIMode(mode_override)
    
    # Initialize output renderer
    renderer = OutputRenderer(config)
    
    # Initialize command registry
    commands = CommandRegistry()
    load_builtin_commands(commands)
    load_custom_commands(commands, config)
    
    # Create REPL
    repl = REPL(
        session=session,
        session_manager=session_manager,
        config=config,
        commands=commands,
        renderer=renderer,
        llm_service=llm_service,
        tool_service=tool_service,
        memory_service=memory_service,
        workflow_service=workflow_service,
    )
    
    # Run in appropriate mode
    if print_mode and prompt:
        # One-shot mode
        response = await repl.process_once(prompt)
        print(response)
    elif prompt:
        # Start REPL with initial prompt
        await repl.run(initial_prompt=prompt)
    else:
        # Interactive REPL
        await repl.run()
    
    # Cleanup
    await llm_service.cleanup()


# src/cli/repl.py - REPL implementation
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console

class REPL:
    """Main REPL loop"""
    
    def __init__(
        self,
        session: SessionState,
        session_manager: SessionManager,
        config: CLIConfig,
        commands: CommandRegistry,
        renderer: OutputRenderer,
        llm_service: LLMOrchestrator,
        tool_service: ToolOrchestrator,
        memory_service: MemoryOrchestrator,
        workflow_service: WorkflowOrchestrator,
    ):
        self.session = session
        self.session_manager = session_manager
        self.config = config
        self.commands = commands
        self.renderer = renderer
        self.llm = llm_service
        self.tools = tool_service
        self.memory = memory_service
        self.workflow = workflow_service
        self.console = Console()
        
        # Initialize prompt session
        history_file = Path.home() / ".agent-swarm" / "history"
        history_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.prompt_session = PromptSession(
            history=FileHistory(str(history_file)),
            auto_suggest=AutoSuggestFromHistory(),
            key_bindings=self._create_key_bindings(),
            vi_mode=config.vim_mode,
        )
        
        # Initialize modes
        self.chat_mode = ChatMode(self)
        self.task_mode = TaskMode(self)
    
    def _create_key_bindings(self) -> KeyBindings:
        """Create custom key bindings"""
        kb = KeyBindings()
        
        @kb.add("c-c")
        def _(event):
            """Handle Ctrl+C"""
            event.app.current_buffer.reset()
        
        @kb.add("c-d")
        def _(event):
            """Handle Ctrl+D"""
            event.app.exit()
        
        return kb
    
    async def run(self, initial_prompt: str = None):
        """Main REPL loop"""
        self._display_welcome()
        
        # Process initial prompt if provided
        if initial_prompt:
            await self._process_input(initial_prompt)
        
        while True:
            try:
                # Get prompt based on mode
                prompt_text = self._get_prompt()
                
                # Get user input
                user_input = await self.prompt_session.prompt_async(prompt_text)
                
                if not user_input.strip():
                    continue
                
                # Process input
                await self._process_input(user_input)
                
            except KeyboardInterrupt:
                continue
            except EOFError:
                break
            except Exception as e:
                self.renderer.render_error(e)
        
        self._display_goodbye()
    
    async def _process_input(self, user_input: str):
        """Process user input"""
        # Check for slash command
        if user_input.startswith("/"):
            await self._handle_command(user_input)
        else:
            # Process with current mode
            await self._handle_message(user_input)
    
    async def _handle_command(self, input_text: str):
        """Handle slash command"""
        parts = input_text[1:].split(maxsplit=1)
        cmd_name = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        command = self.commands.get(cmd_name)
        if command:
            context = CommandContext(
                session=self.session,
                llm_service=self.llm,
                tool_service=self.tools,
                memory_service=self.memory,
                workflow_service=self.workflow,
                console=self.console,
                config=self.config,
            )
            result = await command.execute(args, context)
            if result:
                self.renderer.render(result)
        else:
            self.renderer.render_error(f"Unknown command: /{cmd_name}")
            self.renderer.render("Type /help for available commands")
    
    async def _handle_message(self, message: str):
        """Handle chat/task message"""
        if self.session.mode == CLIMode.CHAT:
            await self.chat_mode.process(message)
        else:
            await self.task_mode.process(message)
        
        # Auto-save session
        if self.config.auto_save_sessions:
            await self.session_manager.save(self.session)
    
    def _get_prompt(self) -> str:
        """Get prompt string based on mode"""
        mode_indicator = "💬" if self.session.mode == CLIMode.CHAT else "🤖"
        return f"{mode_indicator} You: "
    
    def _display_welcome(self):
        """Display welcome message"""
        self.console.print()
        self.console.print("[bold green]Agent Swarm CLI[/bold green]")
        self.console.print(f"Mode: {self.session.mode.value} | Model: {self.session.model or 'auto'}")
        self.console.print("Type [bold]/help[/bold] for commands, [bold]/quit[/bold] to exit")
        self.console.print()
    
    def _display_goodbye(self):
        """Display goodbye message"""
        self.console.print()
        self.console.print("[dim]Session saved. Goodbye![/dim]")


# src/cli/modes/chat.py - Chat mode
class ChatMode:
    """Single-agent conversational mode"""
    
    def __init__(self, repl: "REPL"):
        self.repl = repl
    
    async def process(self, message: str):
        """Process a chat message"""
        # Add user message to history
        self.repl.session.messages.append(Message(role="user", content=message))
        
        # Prepare request
        messages = [
            {"role": m.role, "content": m.content}
            for m in self.repl.session.messages
        ]
        
        # Add system prompt
        system_prompt = self._get_system_prompt()
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        
        # Display assistant prefix
        self.repl.console.print("\n[bold blue]Assistant:[/bold blue]")
        
        # Stream response
        full_response = ""
        try:
            request = LLMRequest(
                messages=messages,
                agent_role="cli_assistant",
                task_description="Respond to user in CLI",
                temperature=self.repl.session.temperature,
            )
            
            async for chunk in self.repl.llm.stream_response(request):
                content = chunk.content if hasattr(chunk, "content") else str(chunk)
                self.repl.console.print(content, end="")
                full_response += content
                
        except KeyboardInterrupt:
            self.repl.console.print("\n[dim](interrupted)[/dim]")
        
        self.repl.console.print("\n")
        
        # Add assistant response to history
        if full_response:
            self.repl.session.messages.append(
                Message(role="assistant", content=full_response)
            )
            self.repl.session.turn_count += 1
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for chat mode"""
        return """You are a helpful AI coding assistant running in a CLI environment.
You have access to tools for reading/writing files, running commands, and more.
Be concise but thorough. Use markdown formatting for code blocks."""


# src/cli/modes/task.py - Task mode
class TaskMode:
    """Multi-agent autonomous workflow mode"""
    
    def __init__(self, repl: "REPL"):
        self.repl = repl
    
    async def process(self, message: str):
        """Process a task request"""
        self.repl.console.print("\n[bold yellow]Starting autonomous task...[/bold yellow]")
        
        # Create task spec
        task = {
            "id": f"cli_task_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "description": message,
            "requirements": [],  # Could parse from message
        }
        
        # Create workflow config
        config = WorkflowConfig(
            project_id=f"cli_{self.repl.session.session_id}",
            enable_human_approval=True,  # Ask before executing
            max_retries=3,
            save_artifacts=True,
        )
        
        try:
            # Start workflow
            execution = await self.repl.workflow.start_workflow(
                task=task,
                config=config,
            )
            
            # Update session
            self.repl.session.active_workflow_id = execution.workflow_id
            self.repl.session.workflow_status = execution.status.value
            
            # Display results
            self.repl.console.print(f"\n[bold green]✓ Task complete![/bold green]")
            self.repl.console.print(f"Status: {execution.status.value}")
            self.repl.console.print(f"Duration: {execution.elapsed_time_seconds:.1f}s")
            self.repl.console.print(f"Cost: ${execution.total_cost_usd:.4f}")
            
        except Exception as e:
            self.repl.console.print(f"\n[bold red]✗ Task failed: {e}[/bold red]")
        
        self.repl.console.print()
```

### Integration Points

```yaml
LLM_SERVICE:
  - Use existing LLMOrchestrator
  - stream_response() for real-time output
  - Respect session.model override if set
  - Track tokens and cost in session state

TOOL_SERVICE:
  - Pass through to LLM for tool calls in chat mode
  - Agents handle tools in task mode
  - /tools command to list available tools

MEMORY_SERVICE:
  - /memory command for interaction
  - Load project memory on startup if exists
  - Store conversation summaries periodically

WORKFLOW_SERVICE:
  - /task mode uses full agent swarm
  - Display agent progress updates
  - Handle human-in-the-loop approval prompts

CHECKPOINT_SERVICE:
  - Session manager uses for persistence
  - Support workflow resume in task mode

CONFIG:
  - Add to: .agent-swarm/config.yaml, ~/.agent-swarm/config.yaml
  - Variables: |
      theme: monokai
      default_mode: chat
      vim_mode: false
      show_tokens: true
      show_cost: true
      auto_save: true
      model: null  # Use auto-routing
```

## Validation Loop

### Level 1: Syntax & Style (Immediate Feedback)

```bash
# After creating each Python file
ruff check src/cli/ --fix
mypy src/cli/ --strict
ruff format src/cli/

# Validate imports
python -c "from src.cli import REPL, CLIConfig; print('Imports OK')"

# Expected: Zero errors, all imports resolve
```

### Level 2: Unit Tests (Component Validation)

```bash
# Test individual components
pytest src/tests/cli/test_session.py -v --cov=src/cli/session
pytest src/tests/cli/test_commands.py -v --cov=src/cli/commands
pytest src/tests/cli/test_modes.py -v --cov=src/cli/modes

# Run all CLI tests
pytest src/tests/cli/ -v --cov=src/cli --cov-report=term-missing

# Expected: 80%+ coverage, all tests pass
```

### Level 3: Integration Testing (System Validation)

```bash
# Test CLI startup
python cli.py --help
# Expected: Help message displays

# Test one-shot mode
python cli.py -p "What is 2+2?"
# Expected: Response prints and exits

# Test interactive mode (manual)
python cli.py
# Expected: REPL starts, can type commands

# Test session persistence
python cli.py
> /save test_session
> /quit
python cli.py --resume test_session
# Expected: Previous context available

# Test slash commands
python cli.py
> /help
> /cost
> /model
> /mode task
> /clear
# Expected: All commands work
```

### Level 4: Creative & Domain-Specific Validation

```bash
# Startup Performance
time python -c "from src.cli.app import main"
# Expected: <200ms

# Streaming Display Test
python cli.py
> Write a haiku about coding
# Expected: Text appears incrementally, not all at once

# Multi-line Input Test
python cli.py
> Explain this code:\
> def foo():\
>     return 42
# Expected: Multi-line input accepted

# Task Mode Workflow
python cli.py --mode task
> Create a hello world Python script
# Expected: Full agent swarm executes, shows progress

# Custom Command Test
mkdir -p .agent-swarm/commands
echo "Explain the following code:" > .agent-swarm/commands/explain.md
python cli.py
> /explain
# Expected: Custom command loads and executes

# Session Resume Test
python cli.py
> Tell me about Python
> /quit
python cli.py -c
> What did we discuss?
# Expected: Context from previous session available

# Vim Mode Test
python cli.py
> /config set vim_mode true
> /vim
# Expected: Vim keybindings active
```

## Final Validation Checklist

### Technical Validation

- [ ] All 4 validation levels completed successfully
- [ ] CLI tests achieve 80%+ coverage: `pytest src/tests/cli/ --cov=src/cli`
- [ ] No linting errors: `ruff check src/cli/`
- [ ] No type errors: `mypy src/cli/`
- [ ] CLI starts in <200ms

### Feature Validation

- [ ] Interactive REPL works with streaming responses
- [ ] All built-in slash commands functional
- [ ] Custom commands load from .agent-swarm/commands/
- [ ] Session persist/resume works (--continue, --resume)
- [ ] Chat mode provides conversational AI
- [ ] Task mode triggers full agent swarm
- [ ] Command history persists across sessions
- [ ] Multi-line input works (\ + Enter)
- [ ] Ctrl+C and Ctrl+D handled gracefully

### UX Validation

- [ ] Output is readable with proper formatting
- [ ] Code blocks have syntax highlighting
- [ ] Errors are displayed clearly
- [ ] Token/cost display is accurate
- [ ] Mode switching (/chat, /task) works smoothly

### Code Quality Validation

- [ ] Follows existing agent swarm patterns
- [ ] Proper async/await throughout
- [ ] Graceful error handling
- [ ] Session state is JSON-serializable
- [ ] No blocking operations in REPL loop

---

## Anti-Patterns to Avoid

- ❌ Don't use synchronous input in async REPL (blocks event loop)
- ❌ Don't store non-serializable objects in session state
- ❌ Don't mix Rich Live context with prompt_toolkit input
- ❌ Don't hardcode paths (use config and Path objects)
- ❌ Don't ignore KeyboardInterrupt (handle gracefully)
- ❌ Don't accumulate unbounded history (enforce max size)
- ❌ Don't block on LLM response (always stream or show progress)
- ❌ Don't forget to save session on exit
- ❌ Don't assume terminal supports all features (graceful degradation)

---

## Future Enhancements (Out of Scope for This PRP)

- MCP (Model Context Protocol) server integration
- Web UI alongside CLI
- Voice input/output
- IDE plugin (VS Code extension)
- Collaborative sessions (multiple users)
- Plugin system for custom modes
