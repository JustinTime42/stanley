"""Quick test to check Qdrant client API."""
import asyncio
from qdrant_client import AsyncQdrantClient


async def test_api():
    client = AsyncQdrantClient(url="http://localhost:6333")

    # Print available methods
    methods = [m for m in dir(client) if not m.startswith('_') and callable(getattr(client, m))]

    print("Available AsyncQdrantClient methods:")
    search_methods = [m for m in methods if 'search' in m.lower() or 'query' in m.lower()]
    print("\nSearch/Query related methods:")
    for method in sorted(search_methods):
        print(f"  - {method}")

    await client.close()


if __name__ == "__main__":
    asyncio.run(test_api())
