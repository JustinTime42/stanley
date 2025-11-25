"""Simple integration test for model routing system."""

import asyncio
import logging
from src.config.llm_config import LLMConfig
from src.services.llm_service import LLMOrchestrator
from src.models.llm_models import LLMRequest

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_routing():
    """Test model routing with different task complexities."""

    # Initialize LLM orchestrator
    config = LLMConfig()
    orchestrator = LLMOrchestrator(config)

    logger.info(f"Available models: {orchestrator.get_available_models()}")

    # Test 1: Simple task (should route to local if available)
    logger.info("\n=== Test 1: Simple Task ===")
    simple_request = LLMRequest(
        messages=[
            {"role": "user", "content": "Say hello world"}
        ],
        agent_role="test",
        task_description="Simple hello world task",
        temperature=0.7,
    )

    try:
        routing_decision = orchestrator.router.route_request(simple_request)
        logger.info(f"Selected model: {routing_decision.selected_model.model_name}")
        logger.info(f"Estimated cost: ${routing_decision.estimated_cost:.4f}")
        logger.info(f"Routing reason: {routing_decision.routing_reason}")
        logger.info(f"Fallback models: {[m.model_name for m in routing_decision.fallback_models]}")
    except Exception as e:
        logger.error(f"Routing failed: {e}")

    # Test 2: Cache statistics
    logger.info("\n=== Test 2: Cache Statistics ===")
    cache_stats = orchestrator.get_cache_stats()
    logger.info(f"Cache stats: {cache_stats}")

    # Cleanup
    await orchestrator.cleanup()
    logger.info("\n=== Tests Complete ===")


if __name__ == "__main__":
    asyncio.run(test_routing())
