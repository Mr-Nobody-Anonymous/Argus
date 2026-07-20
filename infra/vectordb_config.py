"""
Vector Database Configuration for City OS
Qdrant collection schemas for face embeddings and person Re-ID
"""
from typing import Dict, Any

FACE_COLLECTION: Dict[str, Any] = {
    "name": "face_embeddings",
    "vectors": {
        "size": 512,  # ArcFace embedding dimension
        "distance": "Cosine",
        "hnsw_config": {"m": 16, "ef_construct": 100}
    },
    "payload_schema": {
        "camera_id": {"type": "integer"},
        "track_id": {"type": "integer"},
        "person_name": {"type": "text"},
        "timestamp": {"type": "float"},
        "liveness_score": {"type": "float"},
        "face_bbox": {"type": "geo"}
    }
}

REID_COLLECTION: Dict[str, Any] = {
    "name": "person_reid",
    "vectors": {
        "size": 256,  # fast-reid feature dimension
        "distance": "Dot",  # Better for Re-ID
        "hnsw_config": {"m": 32, "ef_construct": 200}
    },
    "payload_schema": {
        "camera_id": {"type": "integer"},
        "track_id": {"type": "integer"},
        "first_seen": {"type": "float"},
        "last_seen": {"type": "float"},
        "trajectory": {"type": "geo"},
        "cross_camera_matches": {"type": "integer"}
    }
}

def init_collections():
    """Initialize Qdrant collections on startup."""
    try:
        from qdrant_client import QdrantClient
        
        client = QdrantClient("localhost", port=6333)
        
        # Create collections if not exist
        try:
            client.create_collection(**FACE_COLLECTION)
        except Exception:
            pass  # Collection exists
            
        try:
            client.create_collection(**REID_COLLECTION)
        except Exception:
            pass  # Collection exists
            
        return client
    except ImportError:
        return None

if __name__ == "__main__":
    init_collections()