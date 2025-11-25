"""Test query_points response format."""
import asyncio
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid


async def test():
    client = AsyncQdrantClient(url="http://localhost:6333")

    # Create test collection
    coll_name = "format_test"
    try:
        await client.delete_collection(coll_name)
    except:
        pass

    await client.create_collection(
        collection_name=coll_name,
        vectors_config=VectorParams(size=4, distance=Distance.COSINE),
    )

    # Add a point
    point_id = str(uuid.uuid4())
    await client.upsert(
        collection_name=coll_name,
        points=[
            PointStruct(
                id=point_id,
                vector=[0.1, 0.2, 0.3, 0.4],
                payload={"content": "test"}
            )
        ]
    )

    # Query
    results = await client.query_points(
        collection_name=coll_name,
        query=[0.1, 0.2, 0.3, 0.4],
        limit=1,
    )

    print("Query results type:", type(results))
    print("Query results:", results)
    if hasattr(results, 'points'):
        print("Has points attribute")
        print("Points:", results.points)
        if results.points:
            print("First point type:", type(results.points[0]))
            print("First point:", results.points[0])

    # Cleanup
    await client.delete_collection(coll_name)
    await client.close()


asyncio.run(test())
