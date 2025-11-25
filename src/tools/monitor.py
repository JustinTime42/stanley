"""Resource monitoring for tool execution."""

import logging
from typing import Optional, Dict

try:
    import psutil
except ImportError:
    psutil = None

from ..models.execution_models import ResourceUsage

logger = logging.getLogger(__name__)


class ResourceMonitor:
    """
    Resource monitor for tracking tool resource usage.

    PATTERN: Context manager for resource tracking
    GOTCHA: Must handle async context
    CRITICAL: Requires psutil package for full functionality
    """

    def __init__(self, tool_name: str):
        """
        Initialize resource monitor.

        Args:
            tool_name: Name of tool being monitored
        """
        self.tool_name = tool_name
        self.start_resources: Optional[ResourceUsage] = None
        self.end_resources: Optional[ResourceUsage] = None
        self.process = psutil.Process() if psutil else None
        self.logger = logging.getLogger(f"{__name__}.{tool_name}")

    async def __aenter__(self) -> "ResourceMonitor":
        """
        Enter async context and capture start resources.

        Returns:
            Self
        """
        self.start_resources = self._get_current_resources()
        self.logger.debug(
            f"Starting resource monitoring for {self.tool_name}: "
            f"CPU={self.start_resources.cpu_percent:.1f}%, "
            f"Memory={self.start_resources.memory_mb:.1f}MB"
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Exit async context and capture end resources.

        Args:
            exc_type: Exception type
            exc_val: Exception value
            exc_tb: Exception traceback
        """
        self.end_resources = self._get_current_resources()
        usage = self.get_usage_delta()

        self.logger.debug(
            f"Resource usage for {self.tool_name}: "
            f"CPU={usage.get('cpu_percent', 0):.1f}%, "
            f"Memory={usage.get('memory_mb', 0):.1f}MB"
        )

    def _get_current_resources(self) -> ResourceUsage:
        """
        Get current resource usage.

        Returns:
            ResourceUsage snapshot
        """
        if not psutil or not self.process:
            # Return zero values if psutil not available
            return ResourceUsage(
                cpu_percent=0.0,
                memory_mb=0.0,
            )

        try:
            # Get process info
            with self.process.oneshot():
                cpu_percent = self.process.cpu_percent()
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024

                # Get I/O counters if available
                try:
                    io_counters = self.process.io_counters()
                    io_read = io_counters.read_bytes
                    io_write = io_counters.write_bytes
                except (AttributeError, OSError):
                    io_read = 0
                    io_write = 0

                # Get network counters (process-level not available, use system)
                try:
                    net_io = psutil.net_io_counters()
                    net_sent = net_io.bytes_sent
                    net_recv = net_io.bytes_recv
                except (AttributeError, OSError):
                    net_sent = 0
                    net_recv = 0

                # Get file descriptors
                try:
                    open_files = len(self.process.open_files())
                except (AttributeError, OSError):
                    open_files = 0

                # Get thread count
                try:
                    thread_count = self.process.num_threads()
                except (AttributeError, OSError):
                    thread_count = 1

                return ResourceUsage(
                    cpu_percent=cpu_percent,
                    memory_mb=memory_mb,
                    io_read_bytes=io_read,
                    io_write_bytes=io_write,
                    network_sent_bytes=net_sent,
                    network_recv_bytes=net_recv,
                    open_files=open_files,
                    thread_count=thread_count,
                )

        except Exception as e:
            self.logger.warning(f"Error getting resource usage: {e}")
            return ResourceUsage(
                cpu_percent=0.0,
                memory_mb=0.0,
            )

    def get_usage_delta(self) -> Dict[str, float]:
        """
        Get resource usage delta (end - start).

        Returns:
            Dictionary of resource deltas
        """
        if not self.start_resources or not self.end_resources:
            return {}

        return {
            "cpu_percent": self.end_resources.cpu_percent
            - self.start_resources.cpu_percent,
            "memory_mb": self.end_resources.memory_mb - self.start_resources.memory_mb,
            "io_read_bytes": self.end_resources.io_read_bytes
            - self.start_resources.io_read_bytes,
            "io_write_bytes": self.end_resources.io_write_bytes
            - self.start_resources.io_write_bytes,
            "network_sent_bytes": self.end_resources.network_sent_bytes
            - self.start_resources.network_sent_bytes,
            "network_recv_bytes": self.end_resources.network_recv_bytes
            - self.start_resources.network_recv_bytes,
            "open_files": self.end_resources.open_files,
            "thread_count": self.end_resources.thread_count,
        }

    def get_current_usage(self) -> ResourceUsage:
        """
        Get current resource usage snapshot.

        Returns:
            Current ResourceUsage
        """
        return self._get_current_resources()

    @staticmethod
    def is_available() -> bool:
        """
        Check if resource monitoring is available.

        Returns:
            True if psutil is installed
        """
        return psutil is not None
