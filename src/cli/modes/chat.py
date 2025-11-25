"""Chat mode for single-agent conversation."""

import logging

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

    # Default system prompt for chat mode
    DEFAULT_SYSTEM_PROMPT = """You are a helpful AI coding assistant running in a CLI environment.
You have access to tools for reading/writing files, running commands, and more.
Be concise but thorough. Use markdown formatting for code blocks.
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
        Get system prompt for chat mode.

        Returns:
            System prompt string
        """
        # Use session override if set
        if self.session.system_prompt:
            return self.session.system_prompt

        return self.DEFAULT_SYSTEM_PROMPT

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
