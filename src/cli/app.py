"""Main CLI application entry point."""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

import click
from dotenv import load_dotenv

# Load .env file from project root or current directory
# This ensures API keys are available before LLM services initialize
_env_paths = [
    Path(__file__).parent.parent.parent / ".env",  # Project root
    Path.cwd() / ".env",  # Current working directory
]
for env_path in _env_paths:
    if env_path.exists():
        load_dotenv(env_path)
        break

from .config.cli_config import load_config, ensure_directories
from .session.state import SessionState, CLIMode
from .session.manager import SessionManager
from .repl import create_repl

logger = logging.getLogger(__name__)


@click.command()
@click.argument("prompt", required=False)
@click.option(
    "-p", "--print", "print_mode",
    is_flag=True,
    help="Print response and exit (non-interactive)",
)
@click.option(
    "-c", "--continue", "continue_session",
    is_flag=True,
    help="Continue most recent session",
)
@click.option(
    "-r", "--resume", "resume_id",
    help="Resume session by ID",
)
@click.option(
    "--model",
    help="Override model for this session",
)
@click.option(
    "--mode",
    type=click.Choice(["chat", "task"]),
    help="Start in specific mode",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Enable verbose logging",
)
@click.option(
    "--config", "config_path",
    type=click.Path(),
    help="Config file path",
)
@click.option(
    "--vim",
    is_flag=True,
    help="Enable vim mode",
)
def main(
    prompt: Optional[str],
    print_mode: bool,
    continue_session: bool,
    resume_id: Optional[str],
    model: Optional[str],
    mode: Optional[str],
    verbose: bool,
    config_path: Optional[str],
    vim: bool,
) -> None:
    """
    Agent Swarm CLI - Interactive AI coding assistant.

    Start interactive mode:
        agent-swarm

    Start with initial prompt:
        agent-swarm "explain this project"

    One-shot query:
        agent-swarm -p "what is 2+2"

    Continue previous session:
        agent-swarm -c

    Resume specific session:
        agent-swarm -r <session-id>
    """
    # Configure logging
    log_level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Suppress noisy loggers in non-verbose mode
    if not verbose:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        # Suppress LLM fallback noise - these are expected when primary fails
        logging.getLogger("src.llm").setLevel(logging.CRITICAL)
        logging.getLogger("src.services").setLevel(logging.CRITICAL)
        logging.getLogger("src.tools").setLevel(logging.CRITICAL)
        logging.getLogger("src.memory").setLevel(logging.CRITICAL)

    # Run async main
    try:
        asyncio.run(run_cli(
            prompt=prompt,
            print_mode=print_mode,
            continue_session=continue_session,
            resume_id=resume_id,
            model_override=model,
            mode_override=mode,
            verbose=verbose,
            config_path=config_path,
            vim_mode=vim,
        ))
    except KeyboardInterrupt:
        pass
    except Exception as e:
        if verbose:
            logger.exception("CLI error")
        else:
            click.echo(f"Error: {e}", err=True)
        sys.exit(1)


async def run_cli(
    prompt: Optional[str] = None,
    print_mode: bool = False,
    continue_session: bool = False,
    resume_id: Optional[str] = None,
    model_override: Optional[str] = None,
    mode_override: Optional[str] = None,
    verbose: bool = False,
    config_path: Optional[str] = None,
    vim_mode: bool = False,
) -> None:
    """
    Async CLI runner.

    Args:
        prompt: Initial prompt
        print_mode: Non-interactive mode
        continue_session: Continue last session
        resume_id: Session ID to resume
        model_override: Model override
        mode_override: Mode override
        verbose: Verbose logging
        config_path: Config file path
        vim_mode: Enable vim mode
    """
    # Load configuration
    config = load_config(config_path)

    # Apply vim mode override
    if vim_mode:
        config.vim_mode = True

    # Ensure directories exist
    ensure_directories(config)

    # Initialize services
    llm_service = None
    tool_service = None
    memory_service = None
    workflow_service = None

    try:
        # Try to import and initialize services
        from ..services.llm_service import LLMOrchestrator
        from ..config.llm_config import LLMConfig

        llm_config = LLMConfig()
        llm_service = LLMOrchestrator(config=llm_config)
        logger.info("LLM service initialized")

    except ImportError as e:
        logger.warning(f"Could not import LLM service: {e}")
    except Exception as e:
        logger.warning(f"Could not initialize LLM service: {e}")

    try:
        from ..services.tool_service import ToolOrchestrator
        tool_service = ToolOrchestrator(llm_service=llm_service)
        logger.info("Tool service initialized")
    except Exception as e:
        logger.debug(f"Could not initialize tool service: {e}")

    try:
        from ..services.memory_service import MemoryOrchestrator
        from ..config.memory_config import MemoryConfig
        memory_config = MemoryConfig()
        memory_service = MemoryOrchestrator(config=memory_config)
        logger.info("Memory service initialized")
    except Exception as e:
        logger.debug(f"Could not initialize memory service: {e}")

    try:
        from ..services.workflow_service import WorkflowOrchestrator
        from ..services.checkpoint_service import CheckpointManager
        from ..config.memory_config import MemoryConfig

        checkpoint_config = MemoryConfig()
        checkpoint_manager = CheckpointManager(checkpoint_config)
        workflow_service = WorkflowOrchestrator(
            checkpoint_manager=checkpoint_manager,
            memory_service=memory_service,
            llm_service=llm_service,
            tool_service=tool_service,
        )
        logger.info("Workflow service initialized")
    except Exception as e:
        logger.debug(f"Could not initialize workflow service: {e}")

    # Initialize session manager
    session_manager = SessionManager(config)

    # Load or create session
    session = None

    if continue_session:
        session = await session_manager.load_latest(str(Path.cwd()))
        if session:
            logger.info(f"Continuing session {session.session_id}")
        else:
            logger.info("No previous session found, creating new one")

    elif resume_id:
        session = await session_manager.load(resume_id)
        if session:
            logger.info(f"Resumed session {session.session_id}")
        else:
            click.echo(f"Session not found: {resume_id}", err=True)
            return

    if not session:
        session = SessionState(
            session_id=session_manager.generate_id(),
            working_directory=str(Path.cwd()),
        )
        logger.info(f"Created new session {session.session_id}")

    # Apply overrides
    if model_override:
        session.model = model_override

    if mode_override:
        session.mode = CLIMode(mode_override)

    # Create REPL
    repl = create_repl(
        session=session,
        session_manager=session_manager,
        config=config,
        llm_service=llm_service,
        tool_service=tool_service,
        memory_service=memory_service,
        workflow_service=workflow_service,
    )

    try:
        # Run in appropriate mode
        if print_mode and prompt:
            # One-shot mode - print response and exit
            await repl.process_once(prompt)
            # Response is already printed by the mode
        elif prompt:
            # Start REPL with initial prompt
            await repl.run(initial_prompt=prompt)
        else:
            # Interactive REPL
            await repl.run()

    finally:
        # Cleanup
        if llm_service:
            await llm_service.cleanup()
        if memory_service:
            await memory_service.cleanup()


if __name__ == "__main__":
    main()
