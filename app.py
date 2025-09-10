import os
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime
import logging

# LangChain imports - FAISS ONLY for Streamlit Cloud compatibility
try:
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

# Vector store - FAISS ONLY (no SQLite dependency)
try:
    from langchain_community.vectorstores import FAISS
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    st.error("❌ FAISS not available. Please install faiss-cpu.")
    st.stop()

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# Sentence Transformers fallback
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Load environment
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Config ---
DATA_DIR = "data"
VECTOR_DIR = "vector_store"

# Set up logging
logging.basicConfig(level=logging.INFO)

class SentenceTransformerEmbeddings:
    """Wrapper for SentenceTransformer embeddings"""
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        
    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()
    
    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

# --- Custom Prompt (Government Decision Support) ---
GOV_PROMPT = """
You are an AI assistant specializing in government decision support and policy analysis.
You must prioritize accuracy, transparency, compliance, and explainability.

### Rules:
- Always cite sources (document name, page, date if available).
- If information is not in the provided context, respond: "Information not found in available documents."
- Never hallucinate regulations, budgets, or laws.
- Ensure outputs are unbiased and legally compliant.
- Use professional, government-appropriate language.

Context:
{context}

Question:
{query}

Answer:"""

prompt_template = PromptTemplate(
    input_variables=["context", "query"],
    template=GOV_PROMPT,
)

# --- Functions ---
def get_embeddings():
    """Get embedding model - prefer Google AI, fallback to SentenceTransformers"""
    if GOOGLE_AVAILABLE and GOOGLE_API_KEY:
        try:
            return GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        except Exception as e:
            st.warning(f"Google AI embeddings failed: {e}")
    
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        try:
            return SentenceTransformerEmbeddings("paraphrase-multilingual-MiniLM-L12-v2")
        except Exception as e:
            st.error(f"SentenceTransformers failed: {e}")
            return None
    
    st.error("No embedding models available!")
    return None

def load_documents(data_dir=DATA_DIR):
    """Load documents from data directory"""
    docs = []
    if not os.path.exists(data_dir):
        return docs
        
    for file in os.listdir(data_dir):
        path = os.path.join(data_dir, file)
        try:
            if file.endswith(".pdf"):
                loader = PyPDFLoader(path)
                docs.extend(loader.load())
            elif file.endswith(".txt"):
                loader = TextLoader(path, encoding='utf-8')
                docs.extend(loader.load())
        except Exception as e:
            st.warning(f"Could not load {file}: {e}")
    
    logging.info(f"Loaded {len(docs)} documents")
    return docs

def build_vector_store():
    """Build FAISS vector store (Streamlit Cloud compatible)"""
    docs = load_documents()
    if not docs:
        st.warning("No documents found in data directory.")
        return None
        
    embeddings = get_embeddings()
    if not embeddings:
        return None
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, 
        chunk_overlap=300
    )
    chunks = splitter.split_documents(docs)
    
    if not chunks:
        st.warning("No text chunks created from documents.")
        return None

    try:
        # Create FAISS vector store
        vectorstore = FAISS.from_documents(chunks, embeddings)
        
        # Try to save locally
        try:
            os.makedirs(VECTOR_DIR, exist_ok=True)
            vectorstore.save_local(VECTOR_DIR)
        except Exception as e:
            st.warning(f"Could not save vector store: {e}")
        
        st.success(f"✅ Built vector store with {len(chunks)} chunks")
        logging.info(f"Built FAISS vector store with {len(chunks)} chunks")
        return vectorstore
        
    except Exception as e:
        st.error(f"❌ Failed to build vector store: {e}")
        return None

def get_vector_store():
    """Load existing vector store or build new one"""
    embeddings = get_embeddings()
    if not embeddings:
        return None
    
    # Try to load existing FAISS vector store
    if os.path.exists(VECTOR_DIR):
        try:
            vectorstore = FAISS.load_local(
                VECTOR_DIR, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            logging.info("Loaded existing FAISS vector store")
            return vectorstore
        except Exception as e:
            st.warning(f"Could not load existing vector store: {e}")
    
    # Build new vector store
    return build_vector_store()

def get_rag_chain():
    """Create RAG chain with FAISS vector store"""
    vectorstore = get_vector_store()
    if not vectorstore:
        return None
    
    # Get LLM
    if GOOGLE_AVAILABLE and GOOGLE_API_KEY:
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-pro", 
                temperature=0.1,
                google_api_key=GOOGLE_API_KEY
            )
        except Exception as e:
            st.error(f"Failed to initialize LLM: {e}")
            return None
    else:
        st.error("Google API key required for LLM functionality.")
        st.info("Please set GOOGLE_API_KEY in your environment or Streamlit secrets.")
        return None
    
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 10}
    )
    
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template},
        return_source_documents=True,
    )
    return chain

# --- Streamlit UI ---
st.set_page_config(
    page_title="Government Decision Support", 
    page_icon="🏛️",
    layout="wide"
)

st.title("🏛️ Government Decision Support RAG System")
st.caption("🌐 **Streamlit Cloud Edition** - FAISS Vector Store (No SQLite Dependency)")

# Sidebar status
with st.sidebar:
    st.header("📊 System Status")
    
    # Check documents
    doc_count = 0
    if os.path.exists(DATA_DIR):
        doc_count = len([f for f in os.listdir(DATA_DIR) 
                        if f.lower().endswith(('.pdf', '.txt'))])
    
    st.metric("📄 Documents", doc_count)
    st.metric("🔧 Vector Store", "FAISS Only")
    st.metric("🔑 API Status", "Ready" if GOOGLE_API_KEY else "Missing")
    
    if st.button("🔄 Rebuild Vector Store"):
        with st.spinner("Rebuilding..."):
            if 'rag_chain' in st.session_state:
                del st.session_state.rag_chain
            st.rerun()

# Main interface
query = st.text_area(
    "Enter your policy or decision-making question:",
    placeholder="Example: What are the budget allocations for infrastructure development in 2025?",
    height=100
)

if st.button("🔍 Ask Question", type="primary"):
    if query.strip():
        if 'rag_chain' not in st.session_state:
            with st.spinner("Initializing system..."):
                st.session_state.rag_chain = get_rag_chain()
        
        if st.session_state.rag_chain:
            with st.spinner("Processing your question..."):
                try:
                    result = st.session_state.rag_chain({"query": query})
                    
                    st.subheader("🤖 Answer")
                    st.markdown(result["result"])
                    
                    # Show sources
                    if result.get("source_documents"):
                        st.subheader("📚 Sources")
                        for i, doc in enumerate(result["source_documents"][:3], 1):
                            source = doc.metadata.get('source', 'Unknown')
                            page = doc.metadata.get('page', 'N/A')
                            st.markdown(f"{i}. **{source}** (Page: {page})")
                            
                except Exception as e:
                    st.error(f"❌ Error processing query: {e}")
        else:
            st.error("❌ System not ready. Please check configuration and documents.")
    else:
        st.warning("Please enter a question.")

# Example queries
with st.expander("💡 Example Questions"):
    st.markdown("""
    - What are the current zoning regulations for commercial buildings?
    - What is the budget allocation for infrastructure development?
    - What are the environmental compliance requirements?
    - සිංහල: වාණිජ ගොඩනැගිලි සඳහා වර්තමාන කලාපීකරණ නියමයන් මොනවාද?
    - தமிழ்: வணிக கட்டிடங்களுக்கான தற்போதைய மண்டல விதிமுறைகள் என்ன?
    """)
