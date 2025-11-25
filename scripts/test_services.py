"""Test script to verify Docker services are accessible."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import redis
from qdrant_client import QdrantClient


def test_redis():
    """Test Redis connection."""
    try:
        client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        client.ping()
        print("[OK] Redis connection successful")

        # Test basic operations
        client.set('test_key', 'test_value')
        value = client.get('test_key')
        assert value == 'test_value'
        client.delete('test_key')
        print("[OK] Redis operations working")
        return True
    except Exception as e:
        print(f"[FAIL] Redis connection failed: {e}")
        return False


def test_qdrant():
    """Test Qdrant connection."""
    try:
        client = QdrantClient(url="http://localhost:6333")
        collections = client.get_collections()
        print(f"[OK] Qdrant connection successful")
        print(f"[OK] Qdrant has {len(collections.collections)} collections")
        return True
    except Exception as e:
        print(f"[FAIL] Qdrant connection failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Agent Swarm Service Connections")
    print("=" * 60)
    print()

    redis_ok = test_redis()
    print()
    qdrant_ok = test_qdrant()
    print()

    print("=" * 60)
    if redis_ok and qdrant_ok:
        print("[SUCCESS] All services accessible!")
        print()
        print("You can now run the Agent Swarm system:")
        print("  python src/main.py")
        print()
        print("Or explore the web interfaces:")
        print("  Redis: http://localhost:8001")
        print("  Qdrant: http://localhost:6333/dashboard")
    else:
        print("[ERROR] Some services are not accessible")
        print("Try: docker-compose -f docker/docker-compose.yml restart")
    print("=" * 60)
