from application.vectorstore.elasticsearch import ElasticsearchStore
from application.vectorstore.faiss import FaissStore
from application.vectorstore.milvus import MilvusStore
from application.vectorstore.mongodb import MongoDBVectorStore
from application.vectorstore.pgvector import PGVectorStore
from application.vectorstore.qdrant import QdrantStore


class VectorCreator:
    vectorstores = None

    @classmethod
    def _get_vectorstores(cls):
        if cls.vectorstores is None:
            cls.vectorstores = {
                "faiss": FaissStore,
                "elasticsearch": ElasticsearchStore,
                "mongodb": MongoDBVectorStore,
                "qdrant": QdrantStore,
                "milvus": MilvusStore,
                "pgvector": PGVectorStore
            }
        return cls.vectorstores

    @classmethod
    def create_vectorstore(cls, type, *args, **kwargs):
        vectorstore_class = cls._get_vectorstores().get(type.lower())
        if not vectorstore_class:
            raise ValueError(f"No vectorstore class found for type {type}")
        return vectorstore_class(*args, **kwargs)
