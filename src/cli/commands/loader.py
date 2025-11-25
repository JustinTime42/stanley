"""Custom command loader from markdown files."""

import logging
from pathlib import Path
from typing import List, Optional

import frontmatter

from .base import CommandRegistry, CustomCommand
from ..config.cli_config import CLIConfig, get_commands_dir

logger = logging.getLogger(__name__)


def load_custom_commands(
    registry: CommandRegistry,
    config: CLIConfig,
) -> int:
    """
    Load custom commands from markdown files.

    Commands are loaded from:
    1. ~/.agent-swarm/commands/ (user commands)
    2. ./.agent-swarm/commands/ (project commands)

    Args:
        registry: Command registry to populate
        config: CLI configuration

    Returns:
        Number of commands loaded
    """
    loaded = 0

    # Load user commands (lower priority)
    user_dir = get_commands_dir(config, "user")
    loaded += _load_commands_from_dir(registry, user_dir, "user")

    # Load project commands (higher priority)
    project_dir = get_commands_dir(config, "project")
    loaded += _load_commands_from_dir(registry, project_dir, "project")

    logger.info(f"Loaded {loaded} custom commands")
    return loaded


def _load_commands_from_dir(
    registry: CommandRegistry,
    directory: Path,
    scope: str,
) -> int:
    """
    Load commands from a directory.

    Args:
        registry: Command registry
        directory: Directory to load from
        scope: Command scope (user or project)

    Returns:
        Number of commands loaded
    """
    if not directory.exists():
        return 0

    loaded = 0

    # Find all markdown files, including in subdirectories
    for md_file in directory.rglob("*.md"):
        try:
            command = _load_command_file(md_file, directory, scope)
            if command:
                registry.register(command)
                loaded += 1
                logger.debug(f"Loaded command /{command.name} from {md_file}")
        except Exception as e:
            logger.warning(f"Failed to load command from {md_file}: {e}")

    return loaded


def _load_command_file(
    file_path: Path,
    base_dir: Path,
    scope: str,
) -> Optional[CustomCommand]:
    """
    Load a command from a markdown file.

    Args:
        file_path: Path to markdown file
        base_dir: Base commands directory
        scope: Command scope

    Returns:
        CustomCommand instance or None
    """
    try:
        post = frontmatter.load(file_path)
    except Exception as e:
        logger.warning(f"Failed to parse frontmatter in {file_path}: {e}")
        # Try loading as plain markdown
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        post = frontmatter.Post(content)

    # Determine command name from path
    # e.g., commands/frontend/component.md -> frontend:component
    rel_path = file_path.relative_to(base_dir)
    parts = list(rel_path.parts)
    parts[-1] = parts[-1].replace(".md", "")  # Remove .md extension

    if len(parts) > 1:
        # Namespaced command: dir1/dir2/cmd.md -> dir1:dir2:cmd
        name = ":".join(parts)
    else:
        name = parts[0]

    # Extract metadata from frontmatter
    metadata = post.metadata or {}

    description = metadata.get("description", "")
    if not description and post.content:
        # Use first line as description
        first_line = post.content.strip().split("\n")[0]
        description = first_line[:50] + "..." if len(first_line) > 50 else first_line

    allowed_tools = metadata.get("allowed-tools", [])
    if isinstance(allowed_tools, str):
        allowed_tools = [t.strip() for t in allowed_tools.split(",")]

    model = metadata.get("model")
    argument_hint = metadata.get("argument-hint", "")

    return CustomCommand(
        name=name,
        content=post.content,
        description=f"{description} ({scope})",
        allowed_tools=allowed_tools,
        model=model,
        argument_hint=argument_hint,
        source_path=str(file_path),
        scope=scope,
    )


def create_example_command(config: CLIConfig) -> None:
    """
    Create an example custom command file.

    Args:
        config: CLI configuration
    """
    project_dir = get_commands_dir(config, "project")
    project_dir.mkdir(parents=True, exist_ok=True)

    example_path = project_dir / "example.md"
    if example_path.exists():
        return

    example_content = '''---
description: Example custom command
argument-hint: [topic]
---

# Example Command

This is an example custom command. The user asked about: $ARGUMENTS

Please provide helpful information about the topic above.

You can also reference files with @filename.txt syntax.
'''

    with open(example_path, "w", encoding="utf-8") as f:
        f.write(example_content)

    logger.info(f"Created example command at {example_path}")


def list_custom_commands(config: CLIConfig) -> List[dict]:
    """
    List all custom commands without loading them.

    Args:
        config: CLI configuration

    Returns:
        List of command info dictionaries
    """
    commands = []

    for scope in ["user", "project"]:
        directory = get_commands_dir(config, scope)
        if not directory.exists():
            continue

        for md_file in directory.rglob("*.md"):
            rel_path = md_file.relative_to(directory)
            parts = list(rel_path.parts)
            parts[-1] = parts[-1].replace(".md", "")
            name = ":".join(parts) if len(parts) > 1 else parts[0]

            commands.append({
                "name": name,
                "scope": scope,
                "path": str(md_file),
            })

    return commands
