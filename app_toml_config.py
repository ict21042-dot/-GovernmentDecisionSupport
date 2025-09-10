import os
import toml
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime
import logging
from typing import Dict, Any, Optional

def load_config() -> Dict[str, Any]:
    """Load configuration from TOML file with fallback to .env"""
    config = {}
    
    # Try to load TOML config first
    if os.path.exists("config.toml"):
        try:
            config = toml.load("config.toml")
            st.success("✅ Loaded configuration from config.toml")
        except Exception as e:
            st.warning(f"⚠️ Could not load config.toml: {e}")
    
    # Fallback to .env file
    if not config and os.path.exists(".env"):
        load_dotenv()
        config = {
            "api": {
                "google_api_key": os.getenv("GOOGLE_API_KEY")
            },
            "system": {
                "data_dir": "data",
                "vector_dir": "vector_store",
                "domain": "Local Government Planning and Decision Making"
            },
            "embeddings": {
                "default_model": "multilingual",
                "batch_size": 16,
                "normalize_embeddings": True
            },
            "vector_store": {
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "search_k": 5,
                "fetch_k": 15,
                "lambda_mult": 0.7
            },
            "llm": {
                "model": "gemini-2.0-flash",
                "temperature": 0.1
            },
            "logging": {
                "level": "INFO",
                "audit_file": "gov_rag_audit.log"
            }
        }
        st.info("📄 Loaded configuration from .env file")
    
    # Environment variable override
    if os.getenv("GOOGLE_API_KEY"):
        if "api" not in config:
            config["api"] = {}
        config["api"]["google_api_key"] = os.getenv("GOOGLE_API_KEY")
    
    return config

def get_api_key(config: Dict[str, Any]) -> Optional[str]:
    """Get Google API key from config"""
    return config.get("api", {}).get("google_api_key")

def get_system_config(config: Dict[str, Any]) -> Dict[str, str]:
    """Get system configuration"""
    return config.get("system", {
        "data_dir": "data",
        "vector_dir": "vector_store",
        "domain": "Local Government Planning and Decision Making"
    })

def get_embedding_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get embedding configuration"""
    return config.get("embeddings", {
        "default_model": "multilingual",
        "batch_size": 16,
        "normalize_embeddings": True
    })

def get_vector_store_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get vector store configuration"""
    return config.get("vector_store", {
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "search_k": 5,
        "fetch_k": 15,
        "lambda_mult": 0.7
    })

def get_llm_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get LLM configuration"""
    return config.get("llm", {
        "model": "gemini-2.0-flash",
        "temperature": 0.1
    })

# Example usage in main app
def main():
    st.set_page_config(
        page_title="Government RAG - TOML Config", 
        page_icon="🏛️",
        layout="wide"
    )
    
    st.title("🏛️ Government RAG System - TOML Configuration")
    
    # Load configuration
    config = load_config()
    
    # Display configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Configuration
        api_key = get_api_key(config)
        if api_key:
            st.success("✅ Google API Key: Loaded")
            st.text(f"Key: {api_key[:20]}...")
        else:
            st.error("❌ Google API Key: Missing")
        
        # System Configuration
        system_config = get_system_config(config)
        st.subheader("📁 System")
        for key, value in system_config.items():
            st.text(f"{key}: {value}")
        
        # Embedding Configuration
        embedding_config = get_embedding_config(config)
        st.subheader("🧠 Embeddings")
        for key, value in embedding_config.items():
            st.text(f"{key}: {value}")
        
        # Vector Store Configuration
        vector_config = get_vector_store_config(config)
        st.subheader("🗃️ Vector Store")
        for key, value in vector_config.items():
            st.text(f"{key}: {value}")
        
        # LLM Configuration
        llm_config = get_llm_config(config)
        st.subheader("🤖 LLM")
        for key, value in llm_config.items():
            st.text(f"{key}: {value}")
    
    st.markdown("""
    ### 📋 Configuration Sources (Priority Order):
    1. **Environment Variables** (highest priority)
    2. **config.toml** file
    3. **.env** file (fallback)
    4. **Default values** (lowest priority)
    
    ### 🔧 TOML Configuration Benefits:
    - ✅ **Structured**: Hierarchical configuration
    - ✅ **Type-safe**: Automatic type conversion
    - ✅ **Readable**: Human-friendly format
    - ✅ **Standardized**: Python packaging standard
    - ✅ **Comments**: Inline documentation support
    """)

if __name__ == "__main__":
    main()
