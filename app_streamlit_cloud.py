import os
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime
import logging
import json
import asyncio
from typing import List, Dict, Any, Optional
import warnings

# Suppress warnings for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.projections")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=UserWarning, module="sentence_transformers")

# Import dependencies with fallbacks for Streamlit Cloud compatibility
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain.schema import Document

# Vector store - FAISS ONLY for Streamlit Cloud (no SQLite dependency)
try:
    from langchain_community.vectorstores import FAISS
    VECTOR_STORE_TYPE = "FAISS"
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    st.error("❌ FAISS not available. Please install faiss-cpu for Streamlit Cloud compatibility.")
    st.stop()

# DO NOT use Chroma on Streamlit Cloud due to SQLite version conflicts

# Embeddings - prefer sentence-transformers for Streamlit Cloud
try:
    from sentence_transformers import SentenceTransformer
    USE_SENTENCE_TRANSFORMERS = True
except ImportError:
    USE_SENTENCE_TRANSFORMERS = False

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    USE_HUGGINGFACE_EMBEDDINGS = True
except ImportError:
    USE_HUGGINGFACE_EMBEDDINGS = False

# Optional: Google AI if API key is available
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
        GOOGLE_AVAILABLE = True
    except ImportError:
        GOOGLE_AVAILABLE = False
else:
    GOOGLE_AVAILABLE = False

# Load environment
load_dotenv()

# Fix asyncio event loop issue for Streamlit
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gov_rag_audit.log'),
        logging.StreamHandler()
    ]
)

# Configuration
DATA_DIR = "data"
VECTOR_DIR = "vector_store"
SUPPORTED_LANGUAGES = ["English", "Sinhala", "Tamil"]

# Streamlit Cloud compatible embedding options
EMBEDDING_OPTIONS = {
    "huggingface_multilingual": {
        "name": "🌍 HuggingFace Multilingual (Free)",
        "model": "paraphrase-multilingual-MiniLM-L12-v2",
        "available": USE_HUGGINGFACE_EMBEDDINGS,
        "type": "huggingface"
    },
    "huggingface_english": {
        "name": "🇺🇸 HuggingFace English (Free)", 
        "model": "all-MiniLM-L6-v2",
        "available": USE_HUGGINGFACE_EMBEDDINGS,
        "type": "huggingface"
    },
    "sentence_transformers": {
        "name": "🔧 Sentence Transformers (Local)",
        "model": "all-MiniLM-L6-v2",
        "available": USE_SENTENCE_TRANSFORMERS,
        "type": "sentence_transformers"
    },
    "google_ai": {
        "name": "🚀 Google AI Embeddings (API)",
        "model": "models/embedding-001",
        "available": GOOGLE_AVAILABLE,
        "type": "google"
    }
}

# System prompt
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

# Custom embeddings wrapper for compatibility
class CompatibleEmbeddings:
    def __init__(self, model_type: str, model_name: str):
        self.model_type = model_type
        self.model_name = model_name
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        try:
            if self.model_type == "huggingface" and USE_HUGGINGFACE_EMBEDDINGS:
                self.model = HuggingFaceEmbeddings(
                    model_name=self.model_name,
                    model_kwargs={'device': 'cpu'}
                )
            elif self.model_type == "sentence_transformers" and USE_SENTENCE_TRANSFORMERS:
                self.model = SentenceTransformer(self.model_name)
            elif self.model_type == "google" and GOOGLE_AVAILABLE:
                self.model = GoogleGenerativeAIEmbeddings(
                    model=self.model_name,
                    google_api_key=GOOGLE_API_KEY
                )
            else:
                st.error(f"❌ Cannot initialize {self.model_type} embeddings")
                return None
        except Exception as e:
            st.error(f"❌ Failed to load model {self.model_name}: {str(e)}")
            return None
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not self.model:
            return []
        
        try:
            if self.model_type == "sentence_transformers":
                return self.model.encode(texts).tolist()
            else:
                return self.model.embed_documents(texts)
        except Exception as e:
            st.error(f"❌ Embedding failed: {str(e)}")
            return []
    
    def embed_query(self, text: str) -> List[float]:
        if not self.model:
            return []
        
        try:
            if self.model_type == "sentence_transformers":
                return self.model.encode([text])[0].tolist()
            else:
                return self.model.embed_query(text)
        except Exception as e:
            st.error(f"❌ Query embedding failed: {str(e)}")
            return []

# Document loading
def load_documents(data_dir=DATA_DIR) -> List[Document]:
    """Load documents from various formats"""
    docs = []
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        st.warning(f"Created {data_dir} directory. Please add your government documents.")
        return docs
    
    for file in os.listdir(data_dir):
        if file.startswith('.'):
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
                        'loaded_at': datetime.now().isoformat()
                    })
                docs.extend(loaded_docs)
                
            elif file.endswith((".txt", ".md")):
                loader = TextLoader(path, encoding='utf-8')
                loaded_docs = loader.load()
                for doc in loaded_docs:
                    doc.metadata.update({
                        'document_type': 'Text',
                        'file_name': file,
                        'loaded_at': datetime.now().isoformat()
                    })
                docs.extend(loaded_docs)
                
            elif file.endswith(".csv"):
                loader = CSVLoader(path)
                loaded_docs = loader.load()
                for doc in loaded_docs:
                    doc.metadata.update({
                        'document_type': 'CSV',
                        'file_name': file,
                        'loaded_at': datetime.now().isoformat()
                    })
                docs.extend(loaded_docs)
                
        except Exception as e:
            logging.error(f"Error loading {file}: {str(e)}")
            st.error(f"Failed to load {file}: {str(e)}")
    
    logging.info(f"Loaded {len(docs)} documents from {data_dir}")
    return docs

# Vector store management
def build_vector_store(embedding_choice: str):
    """Build vector store compatible with Streamlit Cloud"""
    docs = load_documents()
    if not docs:
        st.warning("No documents found to build vector store.")
        return None
    
    # Text splitting
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", ";", ":", " ", ""]
    )
    
    chunks = splitter.split_documents(docs)
    
    # Add metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata.update({
            'chunk_id': i,
            'chunk_size': len(chunk.page_content),
            'created_at': datetime.now().isoformat()
        })
    
    # Get embeddings
    embedding_config = EMBEDDING_OPTIONS[embedding_choice]
    if not embedding_config["available"]:
        st.error(f"❌ {embedding_config['name']} is not available")
        return None
    
    try:
        with st.spinner(f"Building vector store with {embedding_config['name']}..."):
            embeddings = CompatibleEmbeddings(
                embedding_config["type"], 
                embedding_config["model"]
            )
            
            if not embeddings.model:
                return None
            
            # Create vector store - FAISS only for Streamlit Cloud compatibility
            if VECTOR_STORE_TYPE == "FAISS" and FAISS_AVAILABLE:
                vectorstore = FAISS.from_documents(chunks, embeddings)
                # Save to disk if possible
                try:
                    os.makedirs(VECTOR_DIR, exist_ok=True)
                    vectorstore.save_local(VECTOR_DIR)
                except Exception as e:
                    st.warning(f"⚠️ Could not save vector store: {str(e)}")
            else:
                st.error("❌ FAISS is required for Streamlit Cloud compatibility")
                return None
            
            st.success(f"✅ Vector store built with {len(chunks)} chunks")
            logging.info(f"Built vector store with {len(chunks)} chunks")
            
            return vectorstore
    
    except Exception as e:
        logging.error(f"Vector store creation failed: {str(e)}")
        st.error(f"❌ Vector store creation failed: {str(e)}")
        return None

def get_vector_store(embedding_choice: str):
    """Get existing vector store or build new one"""
    try:
        # Try to load existing FAISS store
        if VECTOR_STORE_TYPE == "FAISS" and os.path.exists(VECTOR_DIR):
            embedding_config = EMBEDDING_OPTIONS[embedding_choice]
            if embedding_config["available"]:
                try:
                    embeddings = CompatibleEmbeddings(
                        embedding_config["type"], 
                        embedding_config["model"]
                    )
                    vectorstore = FAISS.load_local(VECTOR_DIR, embeddings, allow_dangerous_deserialization=True)
                    return vectorstore
                except Exception as e:
                    st.info(f"Could not load existing store: {str(e)}")
        
        # Build new vector store
        return build_vector_store(embedding_choice)
        
    except Exception as e:
        logging.error(f"Error with vector store: {str(e)}")
        st.error(f"Vector store error: {str(e)}")
        return None

# Main application
def main():
    st.set_page_config(
        page_title="Government RAG - Cloud Compatible", 
        page_icon="🏛️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🏛️ Government Decision Support RAG System")
    st.markdown("*Streamlit Cloud Compatible Version - 100% Free Operation*")
    
    # Status indicators
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("✅ **Streamlit Cloud Compatible**")
    with col2:
        st.info(f"📊 **Vector Store**: {VECTOR_STORE_TYPE}")
    with col3:
        st.info(f"🤖 **Google API**: {'Available' if GOOGLE_AVAILABLE else 'Local Only'}")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Available embedding options
        available_options = {k: v for k, v in EMBEDDING_OPTIONS.items() if v["available"]}
        
        if not available_options:
            st.error("❌ No embedding models available. Please check installation.")
            st.stop()
        
        embedding_choice = st.selectbox(
            "Choose Embedding Model:",
            list(available_options.keys()),
            format_func=lambda x: available_options[x]["name"]
        )
        
        selected_config = available_options[embedding_choice]
        st.info(f"""
        **Model**: {selected_config['model']}
        **Type**: {selected_config['type']}
        **Cost**: 🆓 Free
        """)
        
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
        
        if st.button("🔄 Rebuild Vector Store"):
            with st.spinner("Rebuilding vector store..."):
                vectorstore = build_vector_store(embedding_choice)
                if vectorstore:
                    st.success("✅ Vector store rebuilt!")
                    st.rerun()
    
    # Main query interface
    st.header("💬 Ask Your Question")
    
    query = st.text_area(
        "Enter your question about government policies, regulations, or decisions:",
        height=100,
        placeholder="Example: What are the budget allocations for infrastructure development in 2025?"
    )
    
    if st.button("🔍 Search Documents", type="primary"):
        if not query.strip():
            st.error("Please enter a question.")
            return
        
        try:
            vectorstore = get_vector_store(embedding_choice)
            if not vectorstore:
                st.error("❌ Vector store not available. Please ensure documents are loaded.")
                return
            
            # Simple document retrieval
            with st.spinner("🔍 Searching documents..."):
                docs = vectorstore.similarity_search(query, k=5)
            
            if not docs:
                st.warning("No relevant documents found.")
                return
            
            # Display results
            if GOOGLE_AVAILABLE and GOOGLE_API_KEY:
                # Try to use Google AI for response generation
                try:
                    from langchain.chains.question_answering import load_qa_chain
                    
                    llm = ChatGoogleGenerativeAI(
                        model="gemini-pro",
                        temperature=0.1,
                        google_api_key=GOOGLE_API_KEY
                    )
                    
                    qa_chain = load_qa_chain(llm, chain_type="stuff", prompt=GOV_RAG_PROMPT)
                    response = qa_chain.run(input_documents=docs, question=query)
                    
                    st.success("✅ AI Analysis Complete")
                    st.header("📋 AI Response")
                    st.markdown(response)
                    
                except Exception as e:
                    st.warning(f"⚠️ AI processing failed: {str(e)}")
                    st.info("💡 Showing document search results instead.")
            else:
                st.info("💡 **Manual Review Mode**: AI analysis not available. Please review the documents below.")
            
            # Show source documents
            st.header("📚 Retrieved Documents")
            
            for i, doc in enumerate(docs, 1):
                with st.expander(f"📄 Document {i}: {doc.metadata.get('file_name', 'Unknown')}"):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown("**Content:**")
                        st.markdown(doc.page_content)
                    
                    with col2:
                        st.markdown("**Metadata:**")
                        metadata_display = {k: v for k, v in doc.metadata.items() 
                                          if k not in ['chunk_id', 'created_at']}
                        st.json(metadata_display)
                        
        except Exception as e:
            logging.error(f"Query processing failed: {str(e)}")
            st.error(f"❌ An error occurred: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("### 🌐 Streamlit Cloud Compatible Government RAG")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **✅ Features:**
        - 🌍 Multilingual document support
        - 🔍 Semantic search capabilities
        - 🏪 Local vector storage (FAISS)
        - 🔒 Privacy-first architecture
        - 🆓 100% free operation
        """)
    
    with col2:
        st.markdown("""
        **🔧 Compatibility:**
        - ✅ Streamlit Cloud ready
        - ✅ No SQLite dependencies
        - ✅ Lightweight requirements
        - ✅ Fallback mechanisms
        - ✅ Error handling
        """)

if __name__ == "__main__":
    main()
