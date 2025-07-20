from langchain_openai import OpenAIEmbeddings
from langgraph.store.redis import RedisStore
from langgraph.store.base import IndexConfig

REDIS_URI = "redis://localhost:6379"

index_config: IndexConfig = {
    "dims": 1536,
    "embed": OpenAIEmbeddings(model="text-embedding-3-small"),
    "ann_index_config": {"vector_type": "vector"},
    "distance_type": "cosine"
}

# Export redis_store instance
def get_redis_store() -> RedisStore:
    with RedisStore.from_conn_string(REDIS_URI) as _redis_store:
        _redis_store.setup()
        redis_store = _redis_store
    return redis_store
