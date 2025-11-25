"""Tests for UnderstandingService."""

import pytest
import tempfile
from pathlib import Path


@pytest.fixture
def temp_project():
    """Create a temporary project for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create sample Python files
        src_dir = Path(tmpdir) / "src"
        src_dir.mkdir()

        main_file = src_dir / "main.py"
        main_file.write_text('''"""Main module."""

def main():
    """Main entry point."""
    print("Hello, World!")


if __name__ == "__main__":
    main()
''')

        yield tmpdir


class TestUnderstandingService:
    """Tests for UnderstandingService."""

    @pytest.mark.asyncio
    async def test_service_creation(self, temp_project):
        """Test service can be created."""
        from src.services.understanding_service import UnderstandingService

        service = UnderstandingService(temp_project)
        assert service.root_path == Path(temp_project).resolve()
        assert not service.is_analyzed

    @pytest.mark.asyncio
    async def test_analyze(self, temp_project):
        """Test analyze method."""
        from src.services.understanding_service import UnderstandingService

        service = UnderstandingService(temp_project)
        understanding = await service.analyze(mode="quick")

        assert understanding is not None
        assert service.is_analyzed

    @pytest.mark.asyncio
    async def test_get_statistics(self, temp_project):
        """Test getting statistics."""
        from src.services.understanding_service import UnderstandingService

        service = UnderstandingService(temp_project)
        await service.analyze(mode="deep")

        stats = service.get_statistics()
        assert stats["analyzed"] is True
        # Deep mode should have populated files
        assert "total_files" in stats
        assert "total_symbols" in stats

    @pytest.mark.asyncio
    async def test_query(self, temp_project):
        """Test query method."""
        from src.services.understanding_service import UnderstandingService

        service = UnderstandingService(temp_project)
        await service.analyze(mode="deep")

        response = await service.query("main function")
        assert response is not None
        assert response.confidence is not None

    @pytest.mark.asyncio
    async def test_watcher_start_stop(self, temp_project):
        """Test watcher lifecycle."""
        from src.services.understanding_service import UnderstandingService

        service = UnderstandingService(temp_project)

        # Start watcher (may fail if watchdog not installed)
        try:
            result = service.start_watcher()
            assert service.is_watching == result

            service.stop_watcher()
            assert not service.is_watching
        except ImportError:
            # watchdog not installed, skip
            pass

    @pytest.mark.asyncio
    async def test_cleanup(self, temp_project):
        """Test cleanup method."""
        from src.services.understanding_service import UnderstandingService

        service = UnderstandingService(temp_project)
        await service.cleanup()
        # Should not raise


class TestConvenienceFunction:
    """Tests for convenience function."""

    @pytest.mark.asyncio
    async def test_analyze_codebase(self, temp_project):
        """Test analyze_codebase convenience function."""
        from src.services.understanding_service import analyze_codebase

        understanding = await analyze_codebase(temp_project, mode="quick")
        assert understanding is not None
        assert understanding.structure.total_files >= 1
