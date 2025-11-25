"""Chat mode for single-agent conversation."""

import logging
import os
from pathlib import Path

from .base import BaseMode
from ...models.llm_models import LLMRequest

logger = logging.getLogger(__name__)


class ChatMode(BaseMode):
    """
    Single-agent conversational mode.

    Direct conversation with LLM including tool use.
    """

    name = "chat"
    description = "Single-agent conversational mode"
    prompt_prefix = "You: "

    # Default system prompt template for chat mode
    DEFAULT_SYSTEM_PROMPT = """You are a helpful AI coding assistant running in a CLI environment.
You are working in the following directory: {working_directory}

{codebase_context}

Be concise but thorough. Use markdown formatting for code blocks.
When asked about the codebase, reference the files and structure shown above.
When asked to perform tasks, break them down into clear steps."""

    async def process(self, message: str) -> None:
        """
        Process a chat message.

        PATTERN: Add user message -> Stream LLM response -> Add to history
        CRITICAL: Must accumulate streaming chunks for history storage

        Args:
            message: User input message
        """
        # Add user message to history
        self.session.add_message("user", message)

        # Prepare messages for LLM
        messages = self.session.get_messages_for_llm()

        # Add system prompt
        system_prompt = self._get_system_prompt()
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        # Display assistant prefix
        self.console.print("\n[bold blue]Assistant:[/bold blue]")

        if not self.llm:
            self.console.print("[red]LLM service not available[/red]")
            return

        # Prepare request
        request = LLMRequest(
            messages=messages,
            agent_role="cli_assistant",
            task_description="Respond to user in CLI",
            temperature=self.session.temperature,
            stream=self.config.stream_output,
        )

        # Apply model override if set
        if self.session.model:
            # Model override is handled by routing, but we can pass a hint
            request.complexity_override = None  # Let the model override take effect

        try:
            if self.config.stream_output:
                # Stream response
                full_response = await self._stream_response(request)
            else:
                # Non-streaming response
                full_response = await self._get_response(request)

            # Add assistant response to history
            if full_response:
                self.session.add_message("assistant", full_response)

            # Show status if enabled
            if self.config.show_tokens or self.config.show_cost:
                self.renderer.render_status(
                    tokens=self.session.total_tokens,
                    cost=self.session.total_cost,
                    model=self.session.model,
                )

        except KeyboardInterrupt:
            self.console.print("\n[dim](interrupted)[/dim]")

        except Exception as e:
            logger.error(f"Chat error: {e}")
            self.renderer.render_error(e)

        self.console.print()  # Newline after response

    async def _stream_response(self, request: LLMRequest) -> str:
        """
        Stream LLM response.

        Args:
            request: LLM request

        Returns:
            Complete response text
        """
        accumulated = ""

        try:
            async for chunk in self.llm.stream_response(request):
                # Extract content from chunk
                if hasattr(chunk, "content"):
                    content = chunk.content
                elif isinstance(chunk, str):
                    content = chunk
                elif isinstance(chunk, dict):
                    content = chunk.get("content", "")
                else:
                    content = str(chunk)

                if content:
                    accumulated += content
                    self.console.print(content, end="", highlight=False)

                # Update stats from chunk if available
                if hasattr(chunk, "input_tokens"):
                    self.session.update_stats(
                        input_tokens=getattr(chunk, "input_tokens", 0),
                        output_tokens=getattr(chunk, "output_tokens", 0),
                        cost=getattr(chunk, "total_cost", 0),
                    )

        except KeyboardInterrupt:
            self.console.print("\n[dim](interrupted)[/dim]")

        self.console.print()  # Final newline
        return accumulated

    async def _get_response(self, request: LLMRequest) -> str:
        """
        Get non-streaming LLM response.

        Args:
            request: LLM request

        Returns:
            Response text
        """
        response = await self.llm.generate_response(request)

        # Update stats
        self.session.update_stats(
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost=response.total_cost,
        )

        # Display response
        self.renderer.render_markdown(response.content)

        return response.content

    def _get_system_prompt(self) -> str:
        """
        Get system prompt for chat mode with codebase context.

        Returns:
            System prompt string
        """
        # Use session override if set
        if self.session.system_prompt:
            return self.session.system_prompt

        # Build codebase context
        codebase_context = self._build_codebase_context()

        return self.DEFAULT_SYSTEM_PROMPT.format(
            working_directory=self.session.working_directory,
            codebase_context=codebase_context,
        )

    def _build_codebase_context(self) -> str:
        """
        Build context about the codebase in the working directory.

        Returns:
            String describing the codebase structure and key files
        """
        working_dir = Path(self.session.working_directory)
        context_parts = []

        # Get directory structure (top-level only to avoid overwhelming)
        try:
            entries = list(working_dir.iterdir())
            dirs = sorted([e.name for e in entries if e.is_dir() and not e.name.startswith('.')])
            files = sorted([e.name for e in entries if e.is_file() and not e.name.startswith('.')])

            if dirs or files:
                context_parts.append("## Directory Structure")
                if dirs:
                    context_parts.append("Directories: " + ", ".join(dirs[:15]))
                if files:
                    context_parts.append("Files: " + ", ".join(files[:15]))
        except Exception:
            pass

        # Read key files for context
        key_files = [
            ("README.md", "Project description"),
            ("README.rst", "Project description"),
            ("pyproject.toml", "Python project config"),
            ("package.json", "Node.js project config"),
            ("Cargo.toml", "Rust project config"),
            ("go.mod", "Go project config"),
            ("requirements.txt", "Python dependencies"),
        ]

        for filename, description in key_files:
            file_path = working_dir / filename
            if file_path.exists():
                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                    # Truncate long files
                    if len(content) > 2000:
                        content = content[:2000] + "\n... (truncated)"
                    context_parts.append(f"\n## {filename} ({description})\n```\n{content}\n```")
                except Exception:
                    pass

        # Check for common source directories
        src_dirs = ['src', 'lib', 'app', 'pkg', 'cmd']
        found_src = None
        for src_dir in src_dirs:
            src_path = working_dir / src_dir
            if src_path.is_dir():
                found_src = src_dir
                try:
                    # List contents of source directory
                    src_entries = list(src_path.iterdir())[:20]
                    src_items = [e.name + ('/' if e.is_dir() else '') for e in src_entries]
                    context_parts.append(f"\n## {src_dir}/ contents\n" + ", ".join(src_items))
                except Exception:
                    pass
                break

        if not context_parts:
            return "No codebase context available in this directory."

        return "\n".join(context_parts)

    def get_prompt(self) -> str:
        """
        Get the prompt string for chat mode.

        Returns:
            Prompt with emoji indicator
        """
        return "💬 You: "

    async def on_enter(self) -> None:
        """Called when entering chat mode."""
        self.renderer.render_info("Entered chat mode (conversational)")

    async def on_exit(self) -> None:
        """Called when exiting chat mode."""
        pass
