"""Tool usage analytics and tracking service."""

import logging
import json
from typing import Dict, List, Optional, Any

try:
    import redis.asyncio as redis
except ImportError:
    redis = None

from ..models.tool_models import ToolResult, ToolStatus
from ..models.execution_models import ExecutionMetrics

logger = logging.getLogger(__name__)


class ToolAnalytics:
    """
    Tool usage analytics and reporting.

    PATTERN: Metrics aggregation and storage
    CRITICAL: Uses Redis for persistence
    GOTCHA: In-memory fallback if Redis unavailable
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        use_redis: bool = True,
    ):
        """
        Initialize analytics service.

        Args:
            redis_url: Redis connection URL
            use_redis: Use Redis for persistence
        """
        self.redis_url = redis_url or "redis://localhost:6379/0"
        self.use_redis = use_redis and redis is not None
        self.redis_client: Optional[redis.Redis] = None
        self.logger = logging.getLogger(__name__)

        # In-memory storage (fallback or cache)
        self.metrics_cache: Dict[str, ExecutionMetrics] = {}

        if self.use_redis:
            try:
                # Initialize Redis connection (lazy)
                self.logger.info(f"Analytics configured with Redis: {self.redis_url}")
            except Exception as e:
                self.logger.warning(f"Redis initialization failed: {e}")
                self.use_redis = False
        else:
            self.logger.info("Analytics using in-memory storage only")

    async def _get_redis(self) -> Optional[redis.Redis]:
        """Get or create Redis client."""
        if not self.use_redis:
            return None

        if self.redis_client is None:
            try:
                self.redis_client = await redis.from_url(self.redis_url)
            except Exception as e:
                self.logger.error(f"Failed to connect to Redis: {e}")
                self.use_redis = False
                return None

        return self.redis_client

    async def record_execution(
        self,
        result: ToolResult,
        agent_id: str = "unknown",
        workflow_id: str = "unknown",
    ) -> None:
        """
        Record tool execution for analytics.

        Args:
            result: Tool execution result
            agent_id: Agent that executed tool
            workflow_id: Workflow context
        """
        try:
            # Get or create metrics for this tool
            metrics = await self.get_metrics(result.tool_name)

            # Update metrics
            metrics.total_executions += 1

            if result.status == ToolStatus.SUCCESS:
                metrics.successful_executions += 1
            elif result.status == ToolStatus.FAILED:
                metrics.failed_executions += 1

            metrics.total_retries += result.retry_count
            metrics.last_execution = result.timestamp

            # Update average execution time
            total_time = (
                metrics.average_execution_time_ms * (metrics.total_executions - 1)
                + result.execution_time_ms
            )
            metrics.average_execution_time_ms = total_time / metrics.total_executions

            # Update resource usage
            if result.resource_usage:
                metrics.total_cpu_seconds += (
                    result.resource_usage.get("cpu_percent", 0) / 100
                )
                metrics.total_memory_mb += result.resource_usage.get("memory_mb", 0)

            # Track error types
            if result.error:
                error_type = (
                    type(result.error).__name__
                    if hasattr(result.error, "__name__")
                    else "Unknown"
                )
                metrics.error_types[error_type] = (
                    metrics.error_types.get(error_type, 0) + 1
                )

            # Store metrics
            await self._store_metrics(result.tool_name, metrics)

            # Store execution record
            await self._store_execution_record(result, agent_id, workflow_id)

        except Exception as e:
            self.logger.error(f"Failed to record execution: {e}")

    async def get_metrics(self, tool_name: str) -> ExecutionMetrics:
        """
        Get metrics for a tool.

        Args:
            tool_name: Tool name

        Returns:
            ExecutionMetrics
        """
        # Try cache first
        if tool_name in self.metrics_cache:
            return self.metrics_cache[tool_name]

        # Try Redis
        if self.use_redis:
            redis_client = await self._get_redis()
            if redis_client:
                try:
                    key = f"tool:metrics:{tool_name}"
                    data = await redis_client.get(key)
                    if data:
                        metrics_dict = json.loads(data)
                        metrics = ExecutionMetrics(**metrics_dict)
                        self.metrics_cache[tool_name] = metrics
                        return metrics
                except Exception as e:
                    self.logger.error(f"Failed to get metrics from Redis: {e}")

        # Return new metrics
        metrics = ExecutionMetrics(tool_name=tool_name)
        self.metrics_cache[tool_name] = metrics
        return metrics

    async def get_all_metrics(self) -> Dict[str, ExecutionMetrics]:
        """
        Get metrics for all tools.

        Returns:
            Dictionary of tool_name -> ExecutionMetrics
        """
        all_metrics = {}

        # Get from cache
        all_metrics.update(self.metrics_cache)

        # Get from Redis if available
        if self.use_redis:
            redis_client = await self._get_redis()
            if redis_client:
                try:
                    pattern = "tool:metrics:*"
                    keys = await redis_client.keys(pattern)

                    for key in keys:
                        data = await redis_client.get(key)
                        if data:
                            metrics_dict = json.loads(data)
                            metrics = ExecutionMetrics(**metrics_dict)
                            tool_name = key.decode().split(":")[-1]
                            all_metrics[tool_name] = metrics
                except Exception as e:
                    self.logger.error(f"Failed to get all metrics from Redis: {e}")

        return all_metrics

    async def get_summary(self) -> Dict[str, Any]:
        """
        Get analytics summary.

        Returns:
            Summary dictionary
        """
        all_metrics = await self.get_all_metrics()

        total_executions = sum(m.total_executions for m in all_metrics.values())
        total_successful = sum(m.successful_executions for m in all_metrics.values())
        total_failed = sum(m.failed_executions for m in all_metrics.values())
        total_retries = sum(m.total_retries for m in all_metrics.values())

        success_rate = (
            (total_successful / total_executions * 100) if total_executions > 0 else 0
        )

        return {
            "total_tools": len(all_metrics),
            "total_executions": total_executions,
            "successful_executions": total_successful,
            "failed_executions": total_failed,
            "success_rate_percent": round(success_rate, 2),
            "total_retries": total_retries,
            "tools": {
                name: {
                    "executions": m.total_executions,
                    "success_rate": round(
                        m.successful_executions / m.total_executions * 100
                        if m.total_executions > 0
                        else 0,
                        2,
                    ),
                    "avg_time_ms": round(m.average_execution_time_ms, 2),
                }
                for name, m in all_metrics.items()
            },
        }

    async def _store_metrics(self, tool_name: str, metrics: ExecutionMetrics) -> None:
        """
        Store metrics.

        Args:
            tool_name: Tool name
            metrics: Metrics to store
        """
        # Update cache
        self.metrics_cache[tool_name] = metrics

        # Store in Redis
        if self.use_redis:
            redis_client = await self._get_redis()
            if redis_client:
                try:
                    key = f"tool:metrics:{tool_name}"
                    data = metrics.model_dump_json()
                    await redis_client.set(key, data)
                except Exception as e:
                    self.logger.error(f"Failed to store metrics in Redis: {e}")

    async def _store_execution_record(
        self,
        result: ToolResult,
        agent_id: str,
        workflow_id: str,
    ) -> None:
        """
        Store individual execution record.

        Args:
            result: Tool result
            agent_id: Agent ID
            workflow_id: Workflow ID
        """
        if not self.use_redis:
            return

        redis_client = await self._get_redis()
        if not redis_client:
            return

        try:
            # Store in sorted set by timestamp
            key = f"tool:executions:{result.tool_name}"
            score = result.timestamp.timestamp()

            record = {
                "tool_name": result.tool_name,
                "status": result.status.value,
                "execution_time_ms": result.execution_time_ms,
                "agent_id": agent_id,
                "workflow_id": workflow_id,
                "timestamp": result.timestamp.isoformat(),
                "error": result.error,
            }

            await redis_client.zadd(key, {json.dumps(record): score})

            # Keep only last 1000 executions per tool
            await redis_client.zremrangebyrank(key, 0, -1001)

        except Exception as e:
            self.logger.error(f"Failed to store execution record: {e}")

    async def get_recent_executions(
        self,
        tool_name: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get recent executions for a tool.

        Args:
            tool_name: Tool name
            limit: Maximum number of records

        Returns:
            List of execution records
        """
        if not self.use_redis:
            return []

        redis_client = await self._get_redis()
        if not redis_client:
            return []

        try:
            key = f"tool:executions:{tool_name}"
            records = await redis_client.zrevrange(key, 0, limit - 1)

            return [json.loads(r) for r in records]

        except Exception as e:
            self.logger.error(f"Failed to get recent executions: {e}")
            return []

    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.redis_client:
            await self.redis_client.close()
        self.logger.info("Analytics service cleaned up")

    async def close(self) -> None:
        """Alias for cleanup()."""
        await self.cleanup()
