"""CLI configuration management."""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CLIMode:
    """CLI mode constants."""

    CHAT = "chat"  # Single-agent conversational
    TASK = "task"  # Multi-agent autonomous workflow


class CLIConfig(BaseModel):
    """CLI configuration model."""

    # Display settings
    theme: str = Field(default="monokai", description="Color theme")
    show_tokens: bool = Field(default=True, description="Show token count")
    show_cost: bool = Field(default=True, description="Show cost information")
    stream_output: bool = Field(default=True, description="Stream responses")

    # Behavior settings
    default_mode: str = Field(default=CLIMode.CHAT, description="Default CLI mode")
    auto_save_sessions: bool = Field(default=True, description="Auto-save sessions")
    max_history_size: int = Field(default=1000, description="Max command history")

    # Vim mode
    vim_mode: bool = Field(default=False, description="Enable vim keybindings")

    # Directories (relative to home or working directory)
    commands_dir: str = Field(
        default=".agent-swarm/commands",
        description="Custom commands directory",
    )
    sessions_dir: str = Field(
        default=".agent-swarm/sessions",
        description="Sessions storage directory",
    )
    config_dir: str = Field(
        default=".agent-swarm",
        description="Configuration directory",
    )

    # Model settings
    default_model: Optional[str] = Field(
        default=None,
        description="Default model (None = auto-routing)",
    )
    default_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Default temperature",
    )

    class Config:
        """Pydantic configuration."""

        extra = "allow"  # Allow extra fields from config files


def get_config_paths() -> list[Path]:
    """
    Get configuration file paths in priority order.

    Returns:
        List of paths, highest priority last
    """
    paths = []

    # Global user config
    home = Path.home()
    global_config = home / ".agent-swarm" / "config.yaml"
    paths.append(global_config)

    # Project-specific config
    cwd = Path.cwd()
    project_config = cwd / ".agent-swarm" / "config.yaml"
    paths.append(project_config)

    return paths


def load_config(config_path: Optional[str] = None) -> CLIConfig:
    """
    Load CLI configuration from files.

    Configuration is merged in this order (later overrides earlier):
    1. Default values
    2. Global config (~/.agent-swarm/config.yaml)
    3. Project config (./.agent-swarm/config.yaml)
    4. Explicit config_path if provided
    5. Environment variables (AGENT_SWARM_*)

    Args:
        config_path: Optional explicit config file path

    Returns:
        Merged CLIConfig instance
    """
    merged_config: Dict[str, Any] = {}

    # Load configs in priority order
    config_paths = get_config_paths()
    if config_path:
        config_paths.append(Path(config_path))

    for path in config_paths:
        if path.exists():
            try:
                with open(path) as f:
                    file_config = yaml.safe_load(f) or {}
                    # Only merge 'cli' section if present, otherwise use whole file
                    if "cli" in file_config:
                        merged_config.update(file_config["cli"])
                    else:
                        merged_config.update(file_config)
                    logger.debug(f"Loaded config from {path}")
            except Exception as e:
                logger.warning(f"Failed to load config from {path}: {e}")

    # Apply environment variable overrides
    env_overrides = _get_env_overrides()
    merged_config.update(env_overrides)

    # Create config instance
    return CLIConfig(**merged_config)


def _get_env_overrides() -> Dict[str, Any]:
    """
    Get configuration overrides from environment variables.

    Environment variables are prefixed with AGENT_SWARM_CLI_.
    Boolean values: "true", "1", "yes" are True; others are False.
    Numeric values are converted automatically.

    Returns:
        Dictionary of overrides
    """
    prefix = "AGENT_SWARM_CLI_"
    overrides: Dict[str, Any] = {}

    for key, value in os.environ.items():
        if key.startswith(prefix):
            # Convert key: AGENT_SWARM_CLI_VIM_MODE -> vim_mode
            config_key = key[len(prefix) :].lower()

            # Convert value types
            if value.lower() in ("true", "1", "yes"):
                overrides[config_key] = True
            elif value.lower() in ("false", "0", "no"):
                overrides[config_key] = False
            else:
                try:
                    overrides[config_key] = int(value)
                except ValueError:
                    try:
                        overrides[config_key] = float(value)
                    except ValueError:
                        overrides[config_key] = value

    return overrides


def save_config(config: CLIConfig, path: Optional[Path] = None) -> None:
    """
    Save configuration to file.

    Args:
        config: Configuration to save
        path: Target path (default: project config)
    """
    if path is None:
        path = Path.cwd() / ".agent-swarm" / "config.yaml"

    # Ensure directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing config to preserve other sections
    existing: Dict[str, Any] = {}
    if path.exists():
        try:
            with open(path) as f:
                existing = yaml.safe_load(f) or {}
        except Exception:
            pass

    # Update CLI section
    existing["cli"] = config.model_dump(exclude_none=True)

    # Write config
    with open(path, "w") as f:
        yaml.dump(existing, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Saved config to {path}")


def get_commands_dir(config: CLIConfig, scope: str = "project") -> Path:
    """
    Get the commands directory path.

    Args:
        config: CLI configuration
        scope: 'project' for current directory, 'user' for home directory

    Returns:
        Path to commands directory
    """
    if scope == "user":
        base = Path.home()
    else:
        base = Path.cwd()

    return base / config.commands_dir


def get_sessions_dir(config: CLIConfig) -> Path:
    """
    Get the sessions directory path.

    Args:
        config: CLI configuration

    Returns:
        Path to sessions directory
    """
    # Sessions are stored in user's home directory
    return Path.home() / config.sessions_dir


def ensure_directories(config: CLIConfig) -> None:
    """
    Ensure all required directories exist.

    Args:
        config: CLI configuration
    """
    dirs = [
        get_commands_dir(config, "user"),
        get_commands_dir(config, "project"),
        get_sessions_dir(config),
    ]

    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
