"""Color themes for CLI output."""

from dataclasses import dataclass
from typing import Dict


@dataclass
class Theme:
    """Color theme definition."""

    name: str

    # Message roles
    user_color: str
    assistant_color: str
    system_color: str

    # UI elements
    prompt_color: str
    info_color: str
    success_color: str
    warning_color: str
    error_color: str

    # Code
    code_style: str  # Rich syntax highlighting theme

    # Status
    dim_color: str
    highlight_color: str


# Available themes
THEMES: Dict[str, Theme] = {
    "monokai": Theme(
        name="monokai",
        user_color="green",
        assistant_color="blue",
        system_color="yellow",
        prompt_color="cyan",
        info_color="cyan",
        success_color="green",
        warning_color="yellow",
        error_color="red",
        code_style="monokai",
        dim_color="dim",
        highlight_color="bold white",
    ),
    "dracula": Theme(
        name="dracula",
        user_color="#50fa7b",  # Dracula green
        assistant_color="#8be9fd",  # Dracula cyan
        system_color="#f1fa8c",  # Dracula yellow
        prompt_color="#bd93f9",  # Dracula purple
        info_color="#8be9fd",
        success_color="#50fa7b",
        warning_color="#ffb86c",  # Dracula orange
        error_color="#ff5555",  # Dracula red
        code_style="dracula",
        dim_color="#6272a4",  # Dracula comment
        highlight_color="#f8f8f2",  # Dracula foreground
    ),
    "github": Theme(
        name="github",
        user_color="#22863a",  # GitHub green
        assistant_color="#0366d6",  # GitHub blue
        system_color="#6f42c1",  # GitHub purple
        prompt_color="#24292e",
        info_color="#0366d6",
        success_color="#22863a",
        warning_color="#b08800",
        error_color="#cb2431",
        code_style="github-dark",
        dim_color="#6a737d",
        highlight_color="#24292e",
    ),
    "light": Theme(
        name="light",
        user_color="dark_green",
        assistant_color="dark_blue",
        system_color="dark_magenta",
        prompt_color="black",
        info_color="blue",
        success_color="green",
        warning_color="dark_orange",
        error_color="red",
        code_style="default",
        dim_color="grey50",
        highlight_color="black",
    ),
    "nord": Theme(
        name="nord",
        user_color="#a3be8c",  # Nord green
        assistant_color="#88c0d0",  # Nord frost
        system_color="#ebcb8b",  # Nord yellow
        prompt_color="#81a1c1",  # Nord blue
        info_color="#88c0d0",
        success_color="#a3be8c",
        warning_color="#ebcb8b",
        error_color="#bf616a",  # Nord red
        code_style="nord",
        dim_color="#4c566a",  # Nord comment
        highlight_color="#eceff4",  # Nord snow
    ),
    "solarized": Theme(
        name="solarized",
        user_color="#859900",  # Solarized green
        assistant_color="#268bd2",  # Solarized blue
        system_color="#b58900",  # Solarized yellow
        prompt_color="#2aa198",  # Solarized cyan
        info_color="#268bd2",
        success_color="#859900",
        warning_color="#cb4b16",  # Solarized orange
        error_color="#dc322f",  # Solarized red
        code_style="solarized-dark",
        dim_color="#586e75",  # Solarized base01
        highlight_color="#fdf6e3",  # Solarized base3
    ),
}


def get_theme(name: str) -> Theme:
    """
    Get a theme by name.

    Args:
        name: Theme name

    Returns:
        Theme instance (falls back to monokai if not found)
    """
    return THEMES.get(name.lower(), THEMES["monokai"])


def list_themes() -> list[str]:
    """
    List available theme names.

    Returns:
        List of theme names
    """
    return list(THEMES.keys())
