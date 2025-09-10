import os
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime
import logging
import json
import asyncio
from typing import List, Dict, Any, Optional

# Import only free/local dependencies
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain.schema import Document

# Try to use the new HuggingFace embeddings package
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

# Optional: Only import Google AI if API key is available
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment
load_dotenv()

# Fix asyncio event loop issue for Streamlit
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Configure logging for audit trail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gov_rag_audit.log'),
        logging.StreamHandler()
    ]
)

# --- Free-Only Configuration ---
DATA_DIR = "data"
VECTOR_DIR = "vector_store"
SUPPORTED_LANGUAGES = ["English", "Sinhala", "Tamil"]
DOMAIN = "Local Government Planning and Decision Making"
DATA_TYPES = ["PDFs", "TXT", "Markdown", "CSV", "Meeting Minutes", "Reports"]

# Free embedding models (no API required)
FREE_EMBEDDING_OPTIONS = {
    "multilingual": {
        "name": "🌍 Multilingual (Free) - Best for Sinhala, Tamil, English",
        "model": "paraphrase-multilingual-MiniLM-L12-v2",
        "languages": "50+ languages including Sinhala, Tamil",
        "size": "420MB",
        "quality": "Good",
        "speed": "Medium",
        "recommended": True
    },
    "english_optimized": {
        "name": "🇺🇸 English Optimized (Free) - Fastest for English",
        "model": "all-MiniLM-L6-v2",
        "languages": "English (optimized)",
        "size": "80MB", 
        "quality": "Good",
        "speed": "Fast",
        "recommended": False
    },
    "multilingual_large": {
        "name": "🌏 Multilingual Large (Free) - Highest Quality",
        "model": "paraphrase-multilingual-mpnet-base-v2",
        "languages": "50+ languages including Sinhala, Tamil",
        "size": "1.1GB",
        "quality": "Excellent",
        "speed": "Slower",
        "recommended": False
    }
}

# --- System Prompts (Free LLM Alternative) ---
SIMPLE_GOVERNMENT_PROMPT = """
Based on the provided government documents, please answer the following question accurately and cite your sources.

Rules:
1. Only use information from the provided documents
2. If information is not available, say "Information not found in available documents"
3. Always cite the document name and page if available
4. Provide clear, factual answers suitable for government use

Documents:
{context}

Question: {query}

Answer:
"""

GOV_RAG_PROMPT = PromptTemplate(
    input_variables=["context", "query"],
    template=SIMPLE_GOVERNMENT_PROMPT
)

# --- Free Embedding Manager ---
class FreeEmbeddingManager:
    """Manages free embedding models only"""
    
    def __init__(self):
        self.current_model = None
        self.model_cache = {}
        
    def get_embedding_model(self, model_choice: str = "multilingual"):
        """Get free embedding model"""
        
        if model_choice in self.model_cache:
            return self.model_cache[model_choice], model_choice
        
        try:
            model_config = FREE_EMBEDDING_OPTIONS[model_choice]
            
            # Initialize HuggingFace embedding model
            embeddings = HuggingFaceEmbeddings(
                model_name=model_config["model"],
                model_kwargs={
                    'device': 'cpu',
                    'trust_remote_code': False  # Security best practice
                },
                encode_kwargs={
                    'normalize_embeddings': True,
                    'batch_size': 32
                }
            )
            
            # Cache the model
            self.model_cache[model_choice] = embeddings
            
            logging.info(f"Loaded free embedding model: {model_config['name']}")
            return embeddings, model_choice
            
        except Exception as e:
            logging.error(f"Failed to load {model_choice}: {str(e)}")
            
            # Fallback to smallest model
            if model_choice != "english_optimized":
                st.warning(f"Failed to load {model_choice}, falling back to English optimized model")
                return self.get_embedding_model("english_optimized")
            else:
                st.error("❌ Failed to load any embedding model. Please check your internet connection for first-time model download.")
                return None, None

# Initialize free embedding manager
embedding_manager = FreeEmbeddingManager()

# --- Document Loading (Same as before) ---
def load_documents(data_dir=DATA_DIR) -> List[Document]:
    """Load documents from various formats with enhanced metadata"""
    docs = []
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        st.warning(f"Created {data_dir} directory. Please add your government documents.")
        return docs
    
    for file in os.listdir(data_dir):
        if file.startswith('.'):  # Skip hidden files
            continue
            
        path = os.path.join(data_dir, file)
        try:
            if file.endswith(".pdf"):
                loader = PyPDFLoader(path)
                loaded_docs = loader.load()
                for doc in loaded_docs:
                    doc.metadata.update({
                        'document_type': 'PDF',
                        'file_name': file,
                        'loaded_at': datetime.now().isoformat(),
                        'domain': DOMAIN
                    })
                docs.extend(loaded_docs)
                
            elif file.endswith((".txt", ".md")):
                loader = TextLoader(path, encoding='utf-8')
                loaded_docs = loader.load()
                for doc in loaded_docs:
                    doc.metadata.update({
                        'document_type': 'Text',
                        'file_name': file,
                        'loaded_at': datetime.now().isoformat(),
                        'domain': DOMAIN
                    })
                docs.extend(loaded_docs)
                
            elif file.endswith(".csv"):
                loader = CSVLoader(path)
                loaded_docs = loader.load()
                for doc in loaded_docs:
                    doc.metadata.update({
                        'document_type': 'CSV',
                        'file_name': file,
                        'loaded_at': datetime.now().isoformat(),
                        'domain': DOMAIN
                    })
                docs.extend(loaded_docs)
                
        except Exception as e:
            logging.error(f"Error loading {file}: {str(e)}")
            st.error(f"Failed to load {file}: {str(e)}")
    
    logging.info(f"Loaded {len(docs)} documents from {data_dir}")
    return docs

def build_vector_store(embedding_choice: str = "multilingual") -> Optional[Chroma]:
    """Build vector store with free embeddings"""
    docs = load_documents()
    if not docs:
        st.warning("No documents found to build vector store.")
        return None, None
    
    # Optimized text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", ";", ":", " ", ""]
    )
    
    chunks = splitter.split_documents(docs)
    
    # Add chunk metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata.update({
            'chunk_id': i,
            'chunk_size': len(chunk.page_content),
            'created_at': datetime.now().isoformat()
        })
    
    # Get free embedding model
    embeddings, model_used = embedding_manager.get_embedding_model(embedding_choice)
    if not embeddings:
        return None, None
    
    try:
        model_info = FREE_EMBEDDING_OPTIONS[model_used]
        
        with st.spinner(f"Building vector store with {model_info['name']}..."):
            vectorstore = Chroma.from_documents(
                chunks, 
                embedding=embeddings, 
                persist_directory=VECTOR_DIR,
                collection_metadata={"hnsw:space": "cosine"}
            )
            vectorstore.persist()
            
            logging.info(f"Built vector store with {len(chunks)} chunks using {model_used}")
            st.success(f"✅ Vector store built with {len(chunks)} chunks using {model_info['name']}")
            
            return vectorstore, model_used
    
    except Exception as e:
        logging.error(f"Vector store creation failed: {str(e)}")
        st.error(f"❌ Vector store creation failed: {str(e)}")
        return None, None

def get_vector_store(embedding_choice: str = "multilingual"):
    """Get existing vector store or build new one"""
    try:
        if os.path.exists(VECTOR_DIR) and os.listdir(VECTOR_DIR):
            embeddings, model_used = embedding_manager.get_embedding_model(embedding_choice)
            if embeddings:
                try:
                    vectorstore = Chroma(
                        persist_directory=VECTOR_DIR, 
                        embedding_function=embeddings
                    )
                    return vectorstore, model_used
                except Exception as e:
                    logging.warning(f"Failed to load existing vector store: {str(e)}")
                    st.info("Rebuilding vector store with selected model...")
        
        # Build new vector store
        return build_vector_store(embedding_choice)
        
    except Exception as e:
        logging.error(f"Error with vector store: {str(e)}")
        st.error(f"Vector store error: {str(e)}")
        return None, None

# --- Free RAG Chain ---
def get_free_rag_chain(embedding_choice: str = "multilingual"):
    """Get RAG chain using only free components"""
    result = get_vector_store(embedding_choice)
    if not result or len(result) != 2:
        return None, None
    
    vectorstore, model_used = result
    if not vectorstore:
        return None, None
    
    # Enhanced retriever
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 15,
            "lambda_mult": 0.7
        }
    )
    
    # Use Google LLM only if API key is available, otherwise provide documents for manual review
    if GOOGLE_API_KEY:
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-pro", 
                temperature=0.1,
                google_api_key=GOOGLE_API_KEY
            )
            
            chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                chain_type="stuff",
                chain_type_kwargs={"prompt": GOV_RAG_PROMPT},
                return_source_documents=True,
                verbose=False
            )
            
            return chain, model_used, True  # Has LLM
            
        except Exception as e:
            if "429" in str(e):
                st.warning("⚠️ Google API quota exceeded. Showing document retrieval only.")
            else:
                st.warning("⚠️ Google API unavailable. Showing document retrieval only.")
    
    # Return retriever only for free operation
    return retriever, model_used, False  # No LLM

# --- Main Application ---
def main():
    st.set_page_config(
        page_title="Government RAG - Free Version", 
        page_icon="🏛️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🏛️ Government Decision Support RAG System")
    st.markdown("*100% Free Version - Local AI Assistant for Government Decision Making*")
    
    # Free system indicator
    st.success("✅ **100% FREE** - No API costs, No usage limits, No external dependencies")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Free Model Configuration")
        
        # Embedding model selection
        st.subheader("🧠 Embedding Model (Free)")
        embedding_choice = st.selectbox(
            "Choose free embedding model:",
            list(FREE_EMBEDDING_OPTIONS.keys()),
            index=0,  # Default to multilingual
            format_func=lambda x: FREE_EMBEDDING_OPTIONS[x]["name"]
        )
        
        # Show embedding model info
        selected_info = FREE_EMBEDDING_OPTIONS[embedding_choice]
        st.info(f"""
        **Model**: {selected_info['name']}
        **Languages**: {selected_info['languages']}
        **Quality**: {selected_info['quality']}
        **Speed**: {selected_info['speed']}
        **Size**: {selected_info['size']}
        **Cost**: 🆓 Completely Free
        """)
        
        if selected_info.get('recommended'):
            st.success("⭐ **Recommended** for government multilingual use")
        
        # Language selection
        selected_lang = st.selectbox(
            "Select your language:",
            SUPPORTED_LANGUAGES,
            help="Note: Translation requires Google API. Use English for full free operation."
        )
        
        # Free operation notice
        if selected_lang != "English" and not GOOGLE_API_KEY:
            st.warning("⚠️ Translation requires Google API. Consider using English interface for full free operation.")
        
        # Document management
        st.header("📄 Document Management")
        
        if os.path.exists(DATA_DIR):
            files = [f for f in os.listdir(DATA_DIR) if not f.startswith('.')]
            st.success(f"📁 {len(files)} documents loaded")
            
            if files:
                st.markdown("**Available documents:**")
                for file in files[:5]:
                    st.markdown(f"- {file}")
                if len(files) > 5:
                    st.markdown(f"... and {len(files) - 5} more")
        else:
            st.warning("📁 No data directory found")
        
        # Vector store status
        if os.path.exists(VECTOR_DIR):
            st.success("🗃️ Vector store ready")
        else:
            st.warning("🗃️ Vector store not built")
        
        # Rebuild vector store button
        if st.button("🔄 Rebuild Vector Store (Free)"):
            with st.spinner("Rebuilding with free embeddings..."):
                if os.path.exists(VECTOR_DIR):
                    import shutil
                    shutil.rmtree(VECTOR_DIR)
                result = build_vector_store(embedding_choice)
                if result and result[0]:
                    st.success("✅ Free vector store rebuilt!")
                else:
                    st.error("❌ Failed to rebuild vector store.")
            st.rerun()
        
        # System info
        st.header("ℹ️ System Info")
        st.markdown(f"""
        **Embedding**: 🆓 Free Local Models
        **LLM**: {'🆓 Google API (if available)' if GOOGLE_API_KEY else '❌ Manual Review Only'}
        **Translation**: {'🆓 Google API (if available)' if GOOGLE_API_KEY else '❌ Not Available'}
        **Storage**: 🆓 Local Files Only
        **Privacy**: 🔒 Complete (data never leaves server)
        """)
    
    # Main query interface
    st.header("💬 Ask Your Question")
    
    query = st.text_area(
        "Enter your question about government policies, regulations, or decisions:",
        height=100,
        placeholder="Example: What are the budget allocations for infrastructure development in 2025?"
    )
    
    # Process query
    if st.button("🔍 Search Documents (Free)", type="primary"):
        if not query.strip():
            st.error("Please enter a question.")
            return
        
        try:
            # Get RAG system
            result = get_free_rag_chain(embedding_choice)
            if not result or len(result) != 3:
                st.error("❌ System not available. Please ensure documents are loaded.")
                return
                
            chain_or_retriever, model_used, has_llm = result
            
            if has_llm:
                # Full RAG with LLM
                with st.spinner("🧠 Processing with AI..."):
                    rag_result = chain_or_retriever.invoke({"query": query})
                
                answer = rag_result["result"]
                source_docs = rag_result["source_documents"]
                
                st.success("✅ AI Analysis Complete")
                st.header("📋 AI Response")
                st.markdown(answer)
                
            else:
                # Document retrieval only
                with st.spinner("🔍 Searching documents..."):
                    source_docs = chain_or_retriever.get_relevant_documents(query)
                
                st.success("✅ Document Search Complete")
                st.header("📋 Relevant Documents Found")
                st.info("💡 **Manual Review Required**: AI analysis unavailable without API access. Please review the documents below to answer your question.")
            
            # Sources and provenance
            if source_docs:
                st.header("📚 Retrieved Documents")
                
                for i, doc in enumerate(source_docs, 1):
                    with st.expander(f"📄 Document {i}: {doc.metadata.get('file_name', 'Unknown')} - Relevance Score #{i}"):
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.markdown("**Content:**")
                            st.markdown(doc.page_content)
                        
                        with col2:
                            st.markdown("**Metadata:**")
                            metadata_display = {k: v for k, v in doc.metadata.items() 
                                              if k not in ['chunk_id', 'created_at']}
                            st.json(metadata_display)
            else:
                st.warning("No relevant documents found. Try rephrasing your question or add more documents to the data directory.")
                
        except Exception as e:
            logging.error(f"Query processing failed: {str(e)}")
            st.error(f"❌ An error occurred: {str(e)}")
    
    # Free operation guide
    st.markdown("---")
    st.markdown("### 🆓 Free Operation Guide")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **✅ What's Free:**
        - 🧠 Document embedding (unlimited)
        - 🔍 Semantic search (unlimited)  
        - 📄 Document retrieval (unlimited)
        - 🏪 Local storage (unlimited)
        - 🔒 Complete privacy (no external calls)
        """)
    
    with col2:
        st.markdown("""
        **⚠️ Requires API (Optional):**
        - 🤖 AI answer generation
        - 🌍 Language translation
        - 📝 Response summarization
        
        **💡 Tip:** Use English interface for full free operation
        """)
    
    st.markdown("""
    ### 📖 How to Use This Free System:
    1. **Add Documents**: Place your government documents in the `data/` folder
    2. **Build Vector Store**: Click "Rebuild Vector Store (Free)" in the sidebar  
    3. **Search**: Ask questions and get relevant document excerpts
    4. **Review**: Manually review the retrieved documents to answer your questions
    
    This system provides powerful document search capabilities completely free, with optional AI enhancements when API access is available.
    """)

if __name__ == "__main__":
    main()
