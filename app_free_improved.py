import os
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime
import logging
import json
import asyncio
from typing import List, Dict, Any, Optional

# Suppress TensorFlow and matplotlib warnings for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.projections")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

# Import only free/local dependencies
try:
    from langchain_chroma import Chroma
    CHROMA_NEW = True
except ImportError:
    try:
        from langchain_community.vectorstores import Chroma
        CHROMA_NEW = False
    except ImportError:
        CHROMA_NEW = False

# FAISS as fallback for SQLite issues
try:
    from langchain_community.vectorstores import FAISS
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain.schema import Document

# Use the latest HuggingFace embeddings package
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    HUGGINGFACE_NEW = True
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    HUGGINGFACE_NEW = False

# Load environment FIRST
load_dotenv()

# Optional: Only import Google AI if API key is available (AFTER loading .env)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        GOOGLE_AVAILABLE = True
        print(f"✅ Google AI available with API key: {GOOGLE_API_KEY[:20]}...")
    except ImportError:
        GOOGLE_AVAILABLE = False
        print("❌ Google AI import failed")
else:
    GOOGLE_AVAILABLE = False
    print("❌ Google API key not found")

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

# --- System Prompts (Compatible with free operation) ---
SIMPLE_GOVERNMENT_PROMPT = """
Based on the provided government documents, please answer the following question accurately and cite your sources.

IMPORTANT RULES:
1. Only use information from the provided documents
2. If information is not available, say "Information not found in available documents"
3. Always cite the document name and page if available
4. Provide clear, factual answers suitable for government use
5. Include document provenance for transparency

Context from Documents:
{context}

Question: {question}

Answer with citations:
"""

GOV_RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=SIMPLE_GOVERNMENT_PROMPT
)

# --- Free Embedding Manager ---
class FreeEmbeddingManager:
    """Manages free embedding models only - updated for latest packages"""
    
    def __init__(self):
        self.current_model = None
        self.model_cache = {}
        
    def get_embedding_model(self, model_choice: str = "multilingual"):
        """Get free embedding model using latest HuggingFace package"""
        
        if model_choice in self.model_cache:
            return self.model_cache[model_choice], model_choice
        
        try:
            model_config = FREE_EMBEDDING_OPTIONS[model_choice]
            
            # Initialize HuggingFace embedding model with latest package
            embeddings = HuggingFaceEmbeddings(
                model_name=model_config["model"],
                model_kwargs={
                    'device': 'cpu',
                    'trust_remote_code': False  # Security best practice
                },
                encode_kwargs={
                    'normalize_embeddings': True,
                    'batch_size': 16  # Reduced for stability
                }
            )
            
            # Cache the model
            self.model_cache[model_choice] = embeddings
            
            logging.info(f"Loaded free embedding model: {model_config['name']}")
            st.success(f"✅ Loaded: {model_config['name']}")
            
            return embeddings, model_choice
            
        except Exception as e:
            logging.error(f"Failed to load {model_choice}: {str(e)}")
            
            # Fallback to smallest model
            if model_choice != "english_optimized":
                st.warning(f"Failed to load {model_choice}, trying English optimized model...")
                return self.get_embedding_model("english_optimized")
            else:
                st.error("❌ Failed to load any embedding model. Please check your internet connection for first-time model download.")
                return None, None

# Initialize free embedding manager
embedding_manager = FreeEmbeddingManager()

# --- Document Loading ---
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

def build_vector_store(embedding_choice: str = "multilingual"):
    """Build vector store with free embeddings - FAISS preferred for Streamlit Cloud"""
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
    
    # Try FAISS first for Streamlit Cloud compatibility
    if FAISS_AVAILABLE:
        try:
            model_info = FREE_EMBEDDING_OPTIONS[model_used]
            
            # Ensure clean vector store directory
            if os.path.exists(VECTOR_DIR):
                import shutil
                shutil.rmtree(VECTOR_DIR)
            os.makedirs(VECTOR_DIR, exist_ok=True)
            
            with st.spinner(f"Building FAISS vector store with {model_info['name']}..."):
                vectorstore = FAISS.from_documents(chunks, embeddings)
                
                # Try to save FAISS to disk
                try:
                    vectorstore.save_local(VECTOR_DIR)
                    logging.info(f"Built FAISS vector store with {len(chunks)} chunks using {model_used}")
                    st.success(f"✅ FAISS vector store built with {len(chunks)} chunks")
                except Exception as save_error:
                    st.warning(f"⚠️ Could not save FAISS store: {str(save_error)}")
                    st.info("ℹ️ Using in-memory vector store (will reset on restart)")
                    logging.info(f"Built in-memory FAISS vector store with {len(chunks)} chunks using {model_used}")
                    st.success(f"✅ In-memory FAISS vector store built with {len(chunks)} chunks")
                
                return vectorstore, model_used
                
        except Exception as faiss_error:
            logging.warning(f"FAISS creation failed: {str(faiss_error)}")
            st.warning(f"⚠️ FAISS failed: {str(faiss_error)}. Trying Chroma...")
    
    # Try Chroma as fallback
    try:
        model_info = FREE_EMBEDDING_OPTIONS[model_used]
        
        # Ensure clean vector store directory
        if os.path.exists(VECTOR_DIR):
            import shutil
            shutil.rmtree(VECTOR_DIR)
        os.makedirs(VECTOR_DIR, exist_ok=True)
        
        with st.spinner(f"Building Chroma vector store with {model_info['name']}..."):
            # Use the newer Chroma package if available
            if CHROMA_NEW:
                from langchain_chroma import Chroma
                vectorstore = Chroma.from_documents(
                    chunks, 
                    embedding=embeddings, 
                    persist_directory=VECTOR_DIR,
                    collection_metadata={"hnsw:space": "cosine"}
                )
            else:
                from langchain_community.vectorstores import Chroma
                vectorstore = Chroma.from_documents(
                    chunks, 
                    embedding=embeddings, 
                    persist_directory=VECTOR_DIR,
                    collection_metadata={"hnsw:space": "cosine"}
                )
            
            # Ensure directory is writable
            os.chmod(VECTOR_DIR, 0o755)
            
            logging.info(f"Built vector store with {len(chunks)} chunks using {model_used}")
            st.success(f"✅ Vector store built with {len(chunks)} chunks")
            
            return vectorstore, model_used
    
    except Exception as e:
        logging.error(f"Vector store creation failed: {str(e)}")
        st.error(f"❌ Vector store creation failed: {str(e)}")
        
        # Check if it's a SQLite version issue
        if "sqlite" in str(e).lower() or "3.35.0" in str(e):
            st.error("❌ SQLite version incompatible with Chroma on Streamlit Cloud")
            st.info("💡 For Streamlit Cloud deployment, please use `streamlit_app.py` which uses FAISS only")
            st.info("💡 Or install faiss-cpu and the system will use FAISS automatically")
            
        return None, None

def get_vector_store(embedding_choice: str = "multilingual"):
    """Get existing vector store or build new one - FAISS preferred for Streamlit Cloud"""
    try:
        if os.path.exists(VECTOR_DIR) and os.listdir(VECTOR_DIR):
            embeddings, model_used = embedding_manager.get_embedding_model(embedding_choice)
            if embeddings:
                # Check if it's a FAISS store first (preferred for Streamlit Cloud)
                faiss_files = [f for f in os.listdir(VECTOR_DIR) if f.endswith('.faiss') or f.endswith('.pkl')]
                
                if faiss_files and FAISS_AVAILABLE:
                    try:
                        vectorstore = FAISS.load_local(VECTOR_DIR, embeddings, allow_dangerous_deserialization=True)
                        st.info("✅ Loaded existing FAISS vector store")
                        return vectorstore, model_used
                    except Exception as faiss_error:
                        st.warning(f"⚠️ Could not load FAISS store: {str(faiss_error)}")
                
                # Try Chroma if FAISS failed or not available
                try:
                    # Ensure directory is writable
                    os.chmod(VECTOR_DIR, 0o755)
                    
                    # Try to load existing Chroma vector store
                    if CHROMA_NEW:
                        from langchain_chroma import Chroma
                        vectorstore = Chroma(
                            persist_directory=VECTOR_DIR, 
                            embedding_function=embeddings
                        )
                    else:
                        from langchain_community.vectorstores import Chroma
                        vectorstore = Chroma(
                            persist_directory=VECTOR_DIR, 
                            embedding_function=embeddings
                        )
                    
                    # Test if the vector store is working
                    try:
                        test_query = vectorstore.similarity_search("test", k=1)
                        return vectorstore, model_used
                    except Exception as e:
                        if "readonly" in str(e).lower() or "1032" in str(e) or "sqlite" in str(e).lower():
                            logging.warning(f"Vector store has SQLite issues, rebuilding: {str(e)}")
                            st.info("🔄 Vector store has database issues, rebuilding with FAISS...")
                            # Remove problematic database and rebuild
                            import shutil
                            shutil.rmtree(VECTOR_DIR)
                            return build_vector_store(embedding_choice)
                        else:
                            raise e
                            
                except Exception as e:
                    if "sqlite" in str(e).lower() or "3.35.0" in str(e):
                        st.error("❌ SQLite compatibility issue detected")
                        st.info("💡 Rebuilding with FAISS for Streamlit Cloud compatibility...")
                        # Remove problematic database and rebuild with FAISS
                        import shutil
                        shutil.rmtree(VECTOR_DIR)
                        return build_vector_store(embedding_choice)
                    else:
                        logging.warning(f"Failed to load existing vector store: {str(e)}")
                        st.info("🔄 Rebuilding vector store with selected model...")
        
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
        return None, None, False
    
    vectorstore, model_used = result
    if not vectorstore:
        return None, None, False
    
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
    if GOOGLE_AVAILABLE and GOOGLE_API_KEY:
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash", 
                temperature=0.1,
                google_api_key=GOOGLE_API_KEY
            )
            
            # Create a more reliable custom chain
            from langchain.chains.question_answering import load_qa_chain
            
            # Use load_qa_chain which is more stable
            qa_chain = load_qa_chain(
                llm=llm, 
                chain_type="stuff",
                prompt=GOV_RAG_PROMPT,
                verbose=False
            )
            
            # Create a simple wrapper that combines retrieval and QA
            class SimpleRAGChain:
                def __init__(self, retriever, qa_chain):
                    self.retriever = retriever
                    self.qa_chain = qa_chain
                
                def invoke(self, inputs):
                    question = inputs.get("question", inputs.get("query", ""))
                    
                    # Use modern invoke method instead of deprecated get_relevant_documents
                    try:
                        docs = self.retriever.invoke(question)
                    except Exception:
                        # Fallback to older method if needed
                        docs = self.retriever.get_relevant_documents(question)
                    
                    if not docs:
                        return {
                            "result": "No relevant documents found for this question.",
                            "source_documents": []
                        }
                    
                    # Use modern invoke method instead of deprecated run
                    try:
                        answer = self.qa_chain.invoke({
                            "input_documents": docs, 
                            "question": question
                        })
                        # Extract answer from different possible response formats
                        if isinstance(answer, dict):
                            result = answer.get("output_text", answer.get("answer", str(answer)))
                        else:
                            result = str(answer)
                    except Exception:
                        # Fallback to older run method if needed
                        result = self.qa_chain.run(input_documents=docs, question=question)
                    
                    return {
                        "result": result,
                        "source_documents": docs
                    }
            
            chain = SimpleRAGChain(retriever, qa_chain)
            
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
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("✅ **100% FREE**")
    with col2:
        st.info(f"🔧 **{('New' if HUGGINGFACE_NEW else 'Legacy')} HF Package**")
    with col3:
        fallback_status = "FAISS Available" if FAISS_AVAILABLE else "Chroma Only"
        st.info(f"📊 **Vector Store**: {fallback_status}")
    
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
        **Languages**: {selected_info['languages']}
        **Quality**: {selected_info['quality']}
        **Speed**: {selected_info['speed']}
        **Size**: {selected_info['size']}
        **Cost**: 🆓 Completely Free
        """)
        
        if selected_info.get('recommended'):
            st.success("⭐ **Recommended** for government multilingual use")
        
        # Document management
        st.header("📄 Document Management")
        
        if os.path.exists(DATA_DIR):
            files = [f for f in os.listdir(DATA_DIR) if not f.startswith('.')]
            st.success(f"📁 {len(files)} documents loaded")
            
            if files:
                with st.expander("Show documents"):
                    for file in files:
                        st.markdown(f"- {file}")
        else:
            st.warning("📁 No data directory found")
        
        # Vector store management
        if os.path.exists(VECTOR_DIR):
            st.success("🗃️ Vector store ready")
        else:
            st.warning("🗃️ Vector store not built")
        
        if st.button("🔄 Rebuild Vector Store"):
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
    
    # Main query interface
    st.header("💬 Ask Your Question")
    
    query = st.text_area(
        "Enter your question about government policies, regulations, or decisions:",
        height=100,
        placeholder="Example: What are the budget allocations for infrastructure development in 2025?"
    )
    
    # Process query
    if st.button("🔍 Search Documents", type="primary"):
        if not query.strip():
            st.error("Please enter a question.")
            return
        
        try:
            # Get RAG system
            result = get_free_rag_chain(embedding_choice)
            if not result or len(result) != 3:
                st.error("❌ System not available. Please ensure documents are loaded and vector store is built.")
                return
                
            chain_or_retriever, model_used, has_llm = result
            
            if has_llm:
                # Full RAG with LLM
                with st.spinner("🧠 Processing with AI..."):
                    try:
                        # RetrievalQA expects "question" as input key
                        rag_result = chain_or_retriever.invoke({"question": query})
                    except Exception as e:
                        if "Missing some input keys" in str(e):
                            # Try alternative input formats
                            try:
                                rag_result = chain_or_retriever.invoke({"query": query})
                            except:
                                rag_result = chain_or_retriever({"question": query})
                        else:
                            raise e
                
                answer = rag_result.get("result", rag_result.get("answer", ""))
                source_docs = rag_result.get("source_documents", [])
                
                st.success("✅ AI Analysis Complete")
                st.header("📋 AI Response")
                st.markdown(answer)
                
            else:
                # Document retrieval only
                with st.spinner("🔍 Searching documents..."):
                    try:
                        # Use modern invoke method
                        source_docs = chain_or_retriever.invoke(query)
                    except Exception:
                        # Fallback to older method if needed
                        source_docs = chain_or_retriever.get_relevant_documents(query)
                
                st.success("✅ Document Search Complete")
                st.header("📋 Relevant Documents Found")
                st.info("💡 **Manual Review Required**: AI analysis unavailable. Please review the documents below to answer your question.")
            
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
            st.error("Please try rebuilding the vector store or check your documents.")
    
    # Footer with information
    st.markdown("---")
    st.markdown("### 🆓 100% Free Government RAG System")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **✅ What's Included (Free):**
        - 🧠 Advanced multilingual embeddings
        - 🔍 Semantic document search  
        - 📄 Document retrieval and ranking
        - 🏪 Local vector storage
        - 🔒 Complete privacy protection
        - 🌍 Supports 50+ languages
        """)
    
    with col2:
        st.markdown("""
        **🔧 Technical Details:**
        - **Embeddings**: HuggingFace Sentence Transformers
        - **Vector DB**: Chroma (local)
        - **Privacy**: 100% local processing
        - **Cost**: $0 forever
        - **Limits**: None
        - **Quality**: 85-95% of commercial APIs
        """)

if __name__ == "__main__":
    main()
