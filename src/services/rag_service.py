"""RAG (Retrieval-Augmented Generation) service."""

import logging
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime

from ..models.memory_models import MemoryType, MemorySearchResult
from ..models.rag_models import ContextOptimization, RetrievalResult
from ..models.document_models import Chunk
from .memory_service import MemoryOrchestrator
from ..rag.context.context_builder import ContextBuilder
from ..rag.context.window_optimizer import WindowOptimizer

logger = logging.getLogger(__name__)


class RAGService:
    """
    RAG service for retrieval-augmented generation.

    Combines memory retrieval with LLM generation to provide
    context-aware responses based on stored memories.
    """

    def __init__(
        self,
        memory_orchestrator: MemoryOrchestrator,
        llm_function: Optional[Callable] = None,
        max_context_tokens: int = 4000,
        use_advanced_context: bool = True,
    ):
        """
        Initialize RAG service.

        Args:
            memory_orchestrator: Memory orchestration service
            llm_function: Async function to call LLM
            max_context_tokens: Maximum tokens for context
            use_advanced_context: Use advanced context builder (with optimization)
        """
        self.memory = memory_orchestrator
        self.llm_function = llm_function
        self.max_context_tokens = max_context_tokens
        self.use_advanced_context = use_advanced_context

        # Initialize advanced context components if enabled
        if use_advanced_context:
            self.context_builder = ContextBuilder()
            self.window_optimizer = WindowOptimizer()
        else:
            self.context_builder = None
            self.window_optimizer = None

    async def generate_with_context(
        self,
        query: str,
        agent_id: str,
        memory_types: Optional[List[MemoryType]] = None,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        **llm_kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Generate response with retrieved context.

        Args:
            query: User query
            agent_id: Agent identifier
            memory_types: Memory tiers to search
            k: Number of memories to retrieve
            filters: Metadata filters
            system_prompt: Optional system prompt
            **llm_kwargs: Additional LLM parameters

        Returns:
            Dictionary with response and metadata
        """
        # Retrieve relevant memories
        memories = await self.memory.retrieve_relevant_memories(
            query=query,
            memory_types=memory_types or [MemoryType.PROJECT],
            k=k,
            filters=filters or {"agent_id": agent_id},
            use_hybrid=True,
            use_cache=True,
        )

        # Build context from memories
        context = self._build_context(memories)

        # Prepare prompt
        prompt = self._prepare_prompt(
            query=query,
            context=context,
            system_prompt=system_prompt,
        )

        # Generate response
        if self.llm_function:
            response = await self._generate_response(prompt, **llm_kwargs)
        else:
            response = {
                "content": "No LLM function provided",
                "model": "none",
            }

        # Store interaction in memory
        await self._store_interaction(
            query=query,
            response=response.get("content", ""),
            agent_id=agent_id,
            memories_used=memories,
        )

        return {
            "response": response,
            "context": context,
            "memories_used": len(memories),
            "memory_sources": [m.source for m in memories],
            "timestamp": datetime.now().isoformat(),
        }

    async def retrieve_and_generate(
        self,
        query: str,
        agent_id: str,
        project_id: Optional[str] = None,
        include_global: bool = False,
        **kwargs: Any,
    ) -> str:
        """
        Simplified retrieval and generation.

        Args:
            query: User query
            agent_id: Agent identifier
            project_id: Project identifier
            include_global: Include global memory
            **kwargs: Additional parameters

        Returns:
            Generated response text
        """
        # Determine memory types
        memory_types = [MemoryType.PROJECT]
        if include_global:
            memory_types.append(MemoryType.GLOBAL)

        # Prepare filters
        filters = {"agent_id": agent_id}
        if project_id:
            filters["project_id"] = project_id

        # Generate with context
        result = await self.generate_with_context(
            query=query,
            agent_id=agent_id,
            memory_types=memory_types,
            filters=filters,
            **kwargs,
        )

        return result["response"].get("content", "")

    def _build_context(
        self,
        memories: List[MemorySearchResult],
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Build context string from retrieved memories.

        Args:
            memories: Retrieved memory results
            max_tokens: Maximum tokens (rough estimate)

        Returns:
            Context string
        """
        max_tokens = max_tokens or self.max_context_tokens

        # Use advanced context builder if available
        if self.use_advanced_context and self.context_builder:
            # Convert MemorySearchResults to RetrievalResults
            retrieval_results = []
            for result in memories:
                # Create a Chunk from memory
                chunk = Chunk(
                    id=result.memory.id,
                    document_id=result.memory.agent_id,
                    content=result.memory.content,
                    start_index=0,
                    end_index=len(result.memory.content),
                    chunk_index=0,
                    metadata=result.memory.metadata,
                    keywords=result.memory.tags,
                )

                retrieval_result = RetrievalResult(
                    chunk=chunk,
                    score=result.score,
                    search_type=result.source,
                )
                retrieval_results.append(retrieval_result)

            # Build optimized context
            optimization = ContextOptimization(
                max_tokens=max_tokens,
                optimization_strategy="compress",
            )

            return self.context_builder.build_context(
                retrieval_results,
                max_tokens,
                optimization,
            )

        # Fall back to simple context building
        context_parts = []
        total_tokens = 0  # Rough estimate: 1 token ~= 4 characters

        for i, result in enumerate(memories, 1):
            memory = result.memory

            # Format memory entry
            entry = f"""
[Memory {i}] (Score: {result.score:.3f}, Source: {result.source})
Content: {memory.content}
Importance: {memory.importance}
Tags: {", ".join(memory.tags) if memory.tags else "none"}
Timestamp: {memory.timestamp.isoformat()}
"""
            # Estimate tokens
            entry_tokens = len(entry) // 4

            if total_tokens + entry_tokens > max_tokens:
                logger.warning(
                    f"Context truncated at {i - 1} memories due to token limit"
                )
                break

            context_parts.append(entry)
            total_tokens += entry_tokens

        return "\n---\n".join(context_parts)

    def _prepare_prompt(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Prepare prompt with context.

        Args:
            query: User query
            context: Retrieved context
            system_prompt: Optional system prompt

        Returns:
            Formatted prompt
        """
        system = (
            system_prompt
            or "You are a helpful AI assistant with access to relevant context from memory."
        )

        prompt = f"""{system}

CONTEXT FROM MEMORY:
{context}

USER QUERY:
{query}

Please provide a helpful response based on the context above. If the context is not relevant, you can still provide a general response.
"""
        return prompt

    async def _generate_response(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Generate response using LLM.

        Args:
            prompt: Formatted prompt
            **kwargs: LLM parameters

        Returns:
            LLM response dictionary
        """
        if not self.llm_function:
            return {"content": "No LLM configured", "model": "none"}

        try:
            response = await self.llm_function(prompt, **kwargs)
            return response

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "content": f"Error: {str(e)}",
                "model": "error",
            }

    async def _store_interaction(
        self,
        query: str,
        response: str,
        agent_id: str,
        memories_used: List[MemorySearchResult],
    ) -> None:
        """
        Store interaction in memory for future reference.

        Args:
            query: User query
            response: Generated response
            agent_id: Agent identifier
            memories_used: Memories used for context
        """
        try:
            # Store query-response pair
            interaction_content = f"Query: {query}\nResponse: {response}"

            memory_ids = [m.memory.id for m in memories_used]

            await self.memory.store_memory(
                content=interaction_content,
                agent_id=agent_id,
                memory_type=MemoryType.PROJECT,
                importance=0.6,  # Medium importance
                tags=["rag_interaction", "query_response"],
                metadata={
                    "query": query,
                    "response_length": len(response),
                    "memories_used": memory_ids,
                    "memory_count": len(memories_used),
                },
            )

            logger.debug("Stored RAG interaction in memory")

        except Exception as e:
            logger.error(f"Failed to store interaction: {e}")

    async def update_memory_importance(
        self,
        memory_id: str,
        importance_delta: float,
        memory_type: MemoryType = MemoryType.PROJECT,
    ) -> None:
        """
        Update memory importance based on usage.

        Args:
            memory_id: Memory identifier
            importance_delta: Change in importance (-1 to 1)
            memory_type: Memory tier
        """
        try:
            # Get appropriate memory store
            if memory_type == MemoryType.PROJECT:
                memory_store = self.memory.project_memory
            elif memory_type == MemoryType.GLOBAL:
                memory_store = self.memory.global_memory
            else:
                logger.warning(f"Cannot update importance for {memory_type}")
                return

            # Get current memory
            memory = await memory_store.get_memory(memory_id)
            if not memory:
                logger.warning(f"Memory {memory_id} not found")
                return

            # Update importance
            new_importance = max(0.0, min(1.0, memory.importance + importance_delta))

            await memory_store.update_memory(
                memory_id=memory_id,
                updates={"importance": new_importance},
            )

            logger.debug(
                f"Updated memory {memory_id} importance: {memory.importance} -> {new_importance}"
            )

        except Exception as e:
            logger.error(f"Failed to update memory importance: {e}")
