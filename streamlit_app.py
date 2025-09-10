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

# Import dependencies - FAISS ONLY for Streamlit Cloud
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain.schema import Document

# Vector store - FAISS ONLY (no SQLite dependency)
try:
    from langchain_community.vectorstores import FAISS
    FAISS_AVAILABLE = True
except ImportError:
    st.error("❌ FAISS not available. Please install faiss-cpu for Streamlit Cloud compatibility.")
    st.stop()

# Embeddings - prefer sentence-transformers for reliability
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
        from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
        USE_GOOGLE_AI = True
    except ImportError:
        USE_GOOGLE_AI = False
else:
    USE_GOOGLE_AI = False

# Load environment variables
load_dotenv()

# Configuration
VECTOR_DIR = "vector_store"
DATA_DIR = "data"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class SentenceTransformerEmbeddings:
    """Custom embeddings wrapper for SentenceTransformer"""
    
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()
    
    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text])[0].tolist()

# Available embedding models (Streamlit Cloud compatible)
EMBEDDING_MODELS = {
    "multilingual": {
        "name": "🌍 Multilingual (Free) - Best for Sinhala, Tamil, English",
        "model": "paraphrase-multilingual-MiniLM-L12-v2",
        "type": "sentence_transformers"
    },
    "english_optimized": {
        "name": "🇺🇸 English Optimized (Free) - Fast & Accurate",
        "model": "all-MiniLM-L6-v2", 
        "type": "sentence_transformers"
    },
    "high_quality": {
        "name": "🎯 High Quality (Free) - Best Performance",
        "model": "all-mpnet-base-v2",
        "type": "sentence_transformers"
    }
}

# Add Google AI if available
if USE_GOOGLE_AI and GOOGLE_API_KEY:
    EMBEDDING_MODELS["google_ai"] = {
        "name": "🔥 Google AI (Premium) - Requires API Key",
        "model": "models/embedding-001",
        "type": "google_ai"
    }

def get_embedding_model(embedding_choice: str):
    """Get embedding model based on choice"""
    config = EMBEDDING_MODELS.get(embedding_choice, EMBEDDING_MODELS["multilingual"])
    
    try:
        if config["type"] == "sentence_transformers" and USE_SENTENCE_TRANSFORMERS:
            embeddings = SentenceTransformerEmbeddings(config["model"])
            logging.info(f"Loaded free embedding model: {config['name']}")
            return embeddings
            
        elif config["type"] == "google_ai" and USE_GOOGLE_AI:
            embeddings = GoogleGenerativeAIEmbeddings(
                model=config["model"],
                google_api_key=GOOGLE_API_KEY
            )
            logging.info(f"Loaded Google AI embedding model: {config['name']}")
            return embeddings
            
        elif USE_HUGGINGFACE_EMBEDDINGS:
            # Fallback to HuggingFace
            embeddings = HuggingFaceEmbeddings(
                model_name=config["model"],
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            logging.info(f"Loaded HuggingFace embedding model: {config['name']}")
            return embeddings
            
    except Exception as e:
        logging.error(f"Failed to load {config['name']}: {str(e)}")
        
    # Final fallback
    if USE_SENTENCE_TRANSFORMERS:
        try:
            embeddings = SentenceTransformerEmbeddings("all-MiniLM-L6-v2")
            logging.info("Loaded fallback embedding model: all-MiniLM-L6-v2")
            return embeddings
        except Exception as e:
            logging.error(f"Fallback embedding model failed: {str(e)}")
    
    st.error("❌ No embedding models available. Please check your installation.")
    return None

def load_documents():
    """Load documents from the data directory"""
    documents = []
    
    if not os.path.exists(DATA_DIR):
        return documents
    
    for filename in os.listdir(DATA_DIR):
        filepath = os.path.join(DATA_DIR, filename)
        
        try:
            if filename.lower().endswith('.pdf'):
                loader = PyPDFLoader(filepath)
                docs = loader.load()
                
            elif filename.lower().endswith('.txt'):
                loader = TextLoader(filepath, encoding='utf-8')
                docs = loader.load()
                
            elif filename.lower().endswith('.csv'):
                loader = CSVLoader(filepath)
                docs = loader.load()
                
            else:
                continue
            
            # Add metadata
            for doc in docs:
                doc.metadata.update({
                    'source_file': filename,
                    'load_timestamp': datetime.now().isoformat(),
                    'file_path': filepath
                })
            
            documents.extend(docs)
            
        except Exception as e:
            st.warning(f"⚠️ Could not load {filename}: {str(e)}")
            logging.error(f"Error loading {filename}: {str(e)}")
    
    logging.info(f"Loaded {len(documents)} documents from {DATA_DIR}")
    return documents

def build_vector_store(embedding_choice: str):
    """Build vector store using FAISS only (Streamlit Cloud compatible)"""
    documents = load_documents()
    
    if not documents:
        st.warning("No documents found to build vector store.")
        return None
    
    embeddings = get_embedding_model(embedding_choice)
    if not embeddings:
        return None
    
    try:
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=300,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        
        if not chunks:
            st.warning("No text chunks created from documents.")
            return None
        
        with st.spinner(f"Building vector store with {EMBEDDING_MODELS[embedding_choice]['name']}..."):
            # Create FAISS vector store only
            vectorstore = FAISS.from_documents(chunks, embeddings)
            
            # Save to disk if possible
            try:
                os.makedirs(VECTOR_DIR, exist_ok=True)
                vectorstore.save_local(VECTOR_DIR)
                logging.info(f"Built FAISS vector store with {len(chunks)} chunks using {embedding_choice}")
            except Exception as e:
                st.warning(f"⚠️ Could not save vector store: {str(e)}")
            
            st.success(f"✅ Vector store built with {len(chunks)} chunks")
            return vectorstore
    
    except Exception as e:
        st.error(f"❌ Vector store creation failed: {str(e)}")
        logging.error(f"Vector store creation failed: {str(e)}")
        return None

def get_vector_store(embedding_choice: str):
    """Load existing vector store or build new one"""
    embeddings = get_embedding_model(embedding_choice)
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
            return vectorstore
        except Exception as e:
            st.warning(f"⚠️ Could not load existing vector store: {str(e)}")
            logging.warning(f"Failed to load existing vector store: {str(e)}")
    
    # Build new vector store
    return build_vector_store(embedding_choice)

def get_llm_model():
    """Get LLM model (Google Gemini if available, otherwise show error)"""
    if USE_GOOGLE_AI and GOOGLE_API_KEY:
        try:
            llm = GoogleGenerativeAI(
                model="gemini-pro",
                google_api_key=GOOGLE_API_KEY,
                temperature=0.1
            )
            return llm
        except Exception as e:
            st.error(f"❌ Failed to initialize Google Gemini: {str(e)}")
            return None
    else:
        st.error("❌ Google API key required for LLM. Please set GOOGLE_API_KEY in your environment.")
        st.info("💡 You can get a free API key from: https://makersuite.google.com/app/apikey")
        return None

def detect_language(text: str) -> str:
    """Simple language detection"""
    # Sinhala Unicode range
    if any('\u0d80' <= char <= '\u0dff' for char in text):
        return 'si'
    # Tamil Unicode range  
    elif any('\u0b80' <= char <= '\u0bff' for char in text):
        return 'ta'
    else:
        return 'en'

def translate_to_english(text: str, source_lang: str) -> str:
    """Translate to English if needed"""
    if source_lang == 'en':
        return text
    
    # For demo purposes, return original text with language note
    # In production, integrate with Google Translate or similar service
    return f"[{source_lang.upper()}] {text}"

def translate_from_english(text: str, target_lang: str) -> str:
    """Translate from English if needed"""
    if target_lang == 'en':
        return text
    
    # For demo purposes, return original text with language note
    # In production, integrate with Google Translate or similar service
    return f"[Translated to {target_lang.upper()}] {text}"

def create_government_prompt():
    """Create government-specific prompt template"""
    template = """You are a knowledgeable AI assistant for local government planning and policy. 
Use the provided context to answer questions accurately and cite your sources.

Context: {context}

Question: {question}

Instructions:
1. Provide accurate, factual information based only on the provided context
2. If information is not in the context, clearly state "Information not found in available documents"
3. Always cite your sources with document names and page numbers when available
4. Use professional, government-appropriate language
5. Structure your response clearly with bullet points or numbered lists when appropriate
6. For policy questions, mention any relevant regulations or compliance requirements

Answer:"""

    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

def process_query(query: str, vectorstore, llm):
    """Process user query with the RAG system"""
    if not vectorstore or not llm:
        return "❌ System not ready. Please check vector store and LLM configuration."
    
    try:
        # Detect language and translate if needed
        detected_lang = detect_language(query)
        english_query = translate_to_english(query, detected_lang)
        
        # Create retrieval chain
        prompt = create_government_prompt()
        from langchain.chains import RetrievalQA
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 5, "fetch_k": 10}
            ),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        
        # Get response
        result = qa_chain({"query": english_query})
        response = result.get("result", "No response generated.")
        
        # Translate back if needed
        final_response = translate_from_english(response, detected_lang)
        
        # Add source information
        sources = result.get("source_documents", [])
        if sources:
            source_info = "\n\n**Sources:**\n"
            for i, doc in enumerate(sources[:3], 1):
                source_file = doc.metadata.get('source_file', 'Unknown')
                page = doc.metadata.get('page', 'N/A')
                source_info += f"{i}. {source_file} (Page: {page})\n"
            final_response += source_info
        
        return final_response
        
    except Exception as e:
        logging.error(f"Query processing error: {str(e)}")
        return f"❌ Error processing query: {str(e)}"

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Government Decision Support (Cloud)",
        page_icon="🏛️",
        layout="wide"
    )
    
    st.title("🏛️ Government Decision Support RAG System")
    st.caption("🌐 **Streamlit Cloud Edition** - SQLite-Free with FAISS")
    
    # Initialize session state
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'llm' not in st.session_state:
        st.session_state.llm = None
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Embedding model selection
        embedding_choice = st.selectbox(
            "Choose Embedding Model:",
            options=list(EMBEDDING_MODELS.keys()),
            format_func=lambda x: EMBEDDING_MODELS[x]["name"],
            index=0
        )
        
        # System status
        st.header("📊 System Status")
        
        # Check document count
        doc_count = 0
        if os.path.exists(DATA_DIR):
            doc_count = len([f for f in os.listdir(DATA_DIR) 
                           if f.lower().endswith(('.pdf', '.txt', '.csv'))])
        
        st.metric("📄 Documents", doc_count)
        st.metric("🔧 Vector Store", "FAISS Only")
        st.metric("🔑 API Status", "Ready" if GOOGLE_API_KEY else "Missing")
        
        # Build vector store button
        if st.button("🔄 Rebuild Vector Store", type="primary"):
            with st.spinner("Building vector store..."):
                st.session_state.vectorstore = build_vector_store(embedding_choice)
        
        # Load vector store on startup or model change
        if st.session_state.vectorstore is None:
            with st.spinner("Loading vector store..."):
                st.session_state.vectorstore = get_vector_store(embedding_choice)
        
        # Initialize LLM
        if st.session_state.llm is None:
            st.session_state.llm = get_llm_model()
    
    # Main interface
    if st.session_state.vectorstore is None:
        st.error("❌ Vector store not available. Please check your documents and try rebuilding.")
        st.info("💡 Make sure you have documents in the `data/` folder and click 'Rebuild Vector Store'.")
        return
    
    if st.session_state.llm is None:
        st.error("❌ LLM not available. Please set your Google API key.")
        return
    
    st.success("✅ System ready for queries!")
    
    # Query interface
    query = st.text_area(
        "Ask your question about government policies and planning:",
        placeholder="Example: What are the budget allocations for infrastructure development in 2025?",
        height=100
    )
    
    if st.button("🔍 Submit Query", type="primary"):
        if query.strip():
            with st.spinner("Processing your query..."):
                response = process_query(query, st.session_state.vectorstore, st.session_state.llm)
                st.markdown("### 🤖 Response:")
                st.markdown(response)
        else:
            st.warning("Please enter a question.")
    
    # Example queries
    with st.expander("💡 Example Queries"):
        st.markdown("""
        **English:**
        - What are the current zoning regulations for commercial buildings?
        - What is the budget allocation for infrastructure development?
        
        **සිංහල:**
        - වාණිජ ගොඩනැගිලි සඳහා වර්තමාන කලාපීකරණ නියමයන් මොනවාද?
        - යටිතල පහසුකම් සංවර්ධනය සඳහා අරමුදල් වෙන් කිරීම කුමක්ද?
        
        **தமிழ்:**
        - வணிக கட்டிடங்களுக்கான தற்போதைய மண்டல விதிமுறைகள் என்ன?
        - உள்கட்டமைப்பு மேம்பாட்டிற்கான பட்ஜெட் ஒதுக்கீடு என்ன?
        """)

if __name__ == "__main__":
    main()
