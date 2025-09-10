# Government RAG System - Configuration Options

# Embedding Configuration
EMBEDDING_PREFERENCE = "local_multilingual"  # Options: "google_api", "local_multilingual", "local_english"

# API Rate Limiting
MAX_API_CALLS_PER_MINUTE = 10
ENABLE_API_FALLBACK = True

# Chunking Configuration  
CHUNK_SIZE = 800  # Smaller chunks reduce API calls
CHUNK_OVERLAP = 200

# Retrieval Configuration
RETRIEVAL_K = 4  # Reduced to minimize API usage
RETRIEVAL_FETCH_K = 12

# System Configuration
ENABLE_AUDIT_LOGGING = True
ENABLE_LOCAL_EMBEDDINGS_FALLBACK = True
SHOW_API_STATUS = True

# Embedding Model Details
EMBEDDING_MODELS = {
    "google_api": {
        "model": "models/embedding-001",
        "provider": "google",
        "requires_api_key": True,
        "cost": "API usage",
        "quality": "High",
        "languages": ["English", "Multilingual"],
        "offline": False
    },
    "local_multilingual": {
        "model": "paraphrase-multilingual-MiniLM-L12-v2", 
        "provider": "sentence-transformers",
        "requires_api_key": False,
        "cost": "Free",
        "quality": "Good",
        "languages": ["English", "50+ languages"],
        "offline": True
    },
    "local_english": {
        "model": "all-MiniLM-L6-v2",
        "provider": "sentence-transformers", 
        "requires_api_key": False,
        "cost": "Free",
        "quality": "Good",
        "languages": ["English"],
        "offline": True
    }
}

# Government Compliance Settings
REQUIRE_PROVENANCE = True
ENABLE_SENSITIVE_CONTENT_DETECTION = True
ENABLE_INPUT_SANITIZATION = True
ENABLE_COMPLIANCE_CHECKING = True
