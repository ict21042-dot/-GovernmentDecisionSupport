import os
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime
import logging
import json
import asyncio
import time
from typing import List, Dict, Any, Optional

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import LLMChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain.schema import Document
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

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

# --- Enhanced Configuration ---
DATA_DIR = "data"
VECTOR_DIR = "vector_store"
SUPPORTED_LANGUAGES = ["English", "Sinhala", "Tamil"]
DOMAIN = "Local Government Planning and Decision Making"
DATA_TYPES = ["PDFs", "TXT", "Markdown", "CSV", "Meeting Minutes", "Reports"]

# Embedding model options
EMBEDDING_OPTIONS = {
    "google_api": {
        "name": "Google AI Embeddings (API)",
        "model": "models/embedding-001",
        "requires_api": True,
        "quality": "High",
        "speed": "Fast"
    },
    "local_multilingual": {
        "name": "Local Multilingual (sentence-transformers)",
        "model": "paraphrase-multilingual-MiniLM-L12-v2",
        "requires_api": False,
        "quality": "Good",
        "speed": "Medium"
    },
    "local_english": {
        "name": "Local English (sentence-transformers)",
        "model": "all-MiniLM-L6-v2",
        "requires_api": False,
        "quality": "Good",
        "speed": "Fast"
    }
}

# --- Enhanced System Prompts ---
GOVERNMENT_RAG_SYSTEM_PROMPT = """
SYSTEM PROMPT: Government Decision Support RAG (Multilingual)

You are an AI architect specializing in Retrieval-Augmented Generation (RAG) systems for local government decision-making. 
You must prioritize accuracy, transparency, compliance, explainability, and multilingual support (Sinhala, Tamil, English).

DOMAIN: {domain}
DATA_TYPES: {data_types}
LANGUAGE: {lang}

RULES / GUIDELINES:
1. Always include **provenance**: document name, page, date if available.
2. If unsure, reply: "I don't know — requires further validation."
3. Do not hallucinate regulations, budgets, or citizen data.
4. Prioritize legally compliant, unbiased, explainable outputs.
5. Support translation of queries/answers for the target language.
6. Structure the output with clear headings when appropriate:
   - Executive Summary
   - Key Findings
   - Supporting Evidence
   - Sources and References
   - Compliance Notes (if applicable)

Context from retrieved documents:
{context}

User Question:
{query}

Provide a comprehensive, accurate answer with proper citations:
"""

TRANSLATION_TO_EN_PROMPT = PromptTemplate(
    input_variables=["text", "source_lang"],
    template="""You are a professional translator specializing in government and administrative documents.

Translate the following text from {source_lang} to English. 
- Maintain the original meaning and context
- Preserve technical terms and proper nouns
- Use formal, administrative language appropriate for government communications

Text to translate:
{text}

English translation:"""
)

TRANSLATION_FROM_EN_PROMPT = PromptTemplate(
    input_variables=["text", "target_lang"],
    template="""You are a professional translator specializing in government and administrative documents.

Translate the following text from English to {target_lang}.
- Use formal, respectful language appropriate for government communications
- Maintain technical accuracy
- Preserve the structure and formatting
- Ensure cultural appropriateness

Text to translate:
{text}

Translation in {target_lang}:"""
)

GOV_RAG_PROMPT = PromptTemplate(
    input_variables=["context", "query", "domain", "data_types", "lang"],
    template=GOVERNMENT_RAG_SYSTEM_PROMPT
)

# --- Enhanced Embedding Management ---
class EmbeddingManager:
    """Manages multiple embedding options with fallbacks and rate limiting"""
    
    def __init__(self):
        self.current_model = None
        self.api_call_count = 0
        self.last_api_reset = datetime.now()
        self.max_api_calls_per_minute = 10  # Conservative limit
        
    def get_embedding_model(self, preferred_option: str = "google_api"):
        """Get embedding model with fallback options"""
        
        # Try preferred option first
        if preferred_option == "google_api" and GOOGLE_API_KEY:
            if self._can_use_api():
                try:
                    return GoogleGenerativeAIEmbeddings(
                        model="models/embedding-001",
                        google_api_key=GOOGLE_API_KEY
                    ), "google_api"
                except Exception as e:
                    logging.warning(f"Google API failed: {str(e)}")
                    st.warning("🔄 Google API limit reached, switching to local embeddings...")
        
        # Fallback to local multilingual model
        try:
            return HuggingFaceEmbeddings(
                model_name="paraphrase-multilingual-MiniLM-L12-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            ), "local_multilingual"
        except Exception as e:
            logging.warning(f"Local multilingual model failed: {str(e)}")
            
        # Final fallback to local English model
        try:
            return HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            ), "local_english"
        except Exception as e:
            logging.error(f"All embedding models failed: {str(e)}")
            st.error("❌ All embedding models failed. Please check your setup.")
            return None, None
    
    def _can_use_api(self) -> bool:
        """Check if we can use the API based on rate limits"""
        now = datetime.now()
        
        # Reset counter if more than a minute has passed
        if (now - self.last_api_reset).seconds > 60:
            self.api_call_count = 0
            self.last_api_reset = now
        
        return self.api_call_count < self.max_api_calls_per_minute
    
    def track_api_call(self):
        """Track API usage"""
        self.api_call_count += 1

# Initialize embedding manager
embedding_manager = EmbeddingManager()

# --- Enhanced Document Loading & Vector Store ---
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
                # Enhanced metadata
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

def build_vector_store(embedding_preference: str = "local_multilingual") -> Optional[Chroma]:
    """Build vector store with enhanced chunking strategy and embedding options"""
    docs = load_documents()
    if not docs:
        st.warning("No documents found to build vector store.")
        return None, None
    
    # Enhanced text splitter for government documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Smaller chunks to reduce API calls
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", ";", ":", " ", ""]
    )
    
    chunks = splitter.split_documents(docs)
    
    # Add chunk-specific metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata.update({
            'chunk_id': i,
            'chunk_size': len(chunk.page_content),
            'created_at': datetime.now().isoformat()
        })
    
    # Get embedding model with fallbacks
    embeddings, model_used = embedding_manager.get_embedding_model(embedding_preference)
    if not embeddings:
        return None, None
    
    try:
        with st.spinner(f"Building vector store with {EMBEDDING_OPTIONS[model_used]['name']}..."):
            vectorstore = Chroma.from_documents(
                chunks, 
                embedding=embeddings, 
                persist_directory=VECTOR_DIR,
                collection_metadata={"hnsw:space": "cosine"}
            )
            vectorstore.persist()
            
            logging.info(f"Built vector store with {len(chunks)} chunks using {model_used}")
            st.success(f"✅ Vector store built with {len(chunks)} chunks using {EMBEDDING_OPTIONS[model_used]['name']}")
            
            return vectorstore, model_used
    
    except Exception as e:
        logging.error(f"Vector store creation failed: {str(e)}")
        if "429" in str(e):
            st.error("🚨 API quota exceeded. Rebuilding with local embeddings...")
            # Try with local embeddings
            embeddings, model_used = embedding_manager.get_embedding_model("local_multilingual")
            if embeddings:
                try:
                    vectorstore = Chroma.from_documents(
                        chunks, 
                        embedding=embeddings, 
                        persist_directory=VECTOR_DIR,
                        collection_metadata={"hnsw:space": "cosine"}
                    )
                    vectorstore.persist()
                    st.success(f"✅ Vector store built with local embeddings: {EMBEDDING_OPTIONS[model_used]['name']}")
                    return vectorstore, model_used
                except Exception as e2:
                    st.error(f"❌ Vector store creation failed: {str(e2)}")
                    return None, None
        else:
            st.error(f"❌ Vector store error: {str(e)}")
            return None, None

def get_vector_store(embedding_preference: str = "local_multilingual") -> Optional[Chroma]:
    """Get existing vector store or build new one"""
    try:
        if os.path.exists(VECTOR_DIR) and os.listdir(VECTOR_DIR):
            # Try to load with preferred embeddings first
            embeddings, model_used = embedding_manager.get_embedding_model(embedding_preference)
            if embeddings:
                try:
                    vectorstore = Chroma(
                        persist_directory=VECTOR_DIR, 
                        embedding_function=embeddings
                    )
                    return vectorstore, model_used
                except Exception as e:
                    logging.warning(f"Failed to load vector store with {model_used}: {str(e)}")
                    # Try with local embeddings
                    embeddings, model_used = embedding_manager.get_embedding_model("local_multilingual")
                    if embeddings:
                        vectorstore = Chroma(
                            persist_directory=VECTOR_DIR, 
                            embedding_function=embeddings
                        )
                        return vectorstore, model_used
        
        # Build new vector store
        return build_vector_store(embedding_preference)
        
    except Exception as e:
        logging.error(f"Error with vector store: {str(e)}")
        st.error(f"Vector store error: {str(e)}")
        return None, None

# --- Enhanced Language Detection & Translation ---
def detect_language(text: str) -> str:
    """Detect language of input text"""
    # Simple heuristic detection - can be enhanced with langdetect library
    sinhala_chars = any('\u0d80' <= char <= '\u0dff' for char in text)
    tamil_chars = any('\u0b80' <= char <= '\u0bff' for char in text)
    
    if sinhala_chars:
        return "Sinhala"
    elif tamil_chars:
        return "Tamil"
    else:
        return "English"

def translate_to_english(text: str, source_lang: str) -> str:
    """Translate text to English with rate limiting"""
    if source_lang == "English":
        return text
    
    try:
        # Check API availability before using
        if embedding_manager._can_use_api() and GOOGLE_API_KEY:
            llm = ChatGoogleGenerativeAI(
                model="gemini-pro", 
                temperature=0.0,
                google_api_key=GOOGLE_API_KEY
            )
            chain = LLMChain(llm=llm, prompt=TRANSLATION_TO_EN_PROMPT)
            translated = chain.run(text=text, source_lang=source_lang)
            embedding_manager.track_api_call()
            
            logging.info(f"Translation to English - Source: {source_lang}, Length: {len(text)}")
            return translated.strip()
        else:
            st.warning("⚠️ API quota reached. Translation may be limited.")
            return text
            
    except Exception as e:
        logging.error(f"Translation to English failed: {str(e)}")
        st.error(f"Translation error: {str(e)}")
        return text

def translate_from_english(text: str, target_lang: str) -> str:
    """Translate text from English with rate limiting"""
    if target_lang == "English":
        return text
    
    try:
        # Check API availability before using
        if embedding_manager._can_use_api() and GOOGLE_API_KEY:
            llm = ChatGoogleGenerativeAI(
                model="gemini-pro", 
                temperature=0.0,
                google_api_key=GOOGLE_API_KEY
            )
            chain = LLMChain(llm=llm, prompt=TRANSLATION_FROM_EN_PROMPT)
            translated = chain.run(text=text, target_lang=target_lang)
            embedding_manager.track_api_call()
            
            logging.info(f"Translation from English - Target: {target_lang}, Length: {len(text)}")
            return translated.strip()
        else:
            st.warning("⚠️ API quota reached. Translation may be limited.")
            return text
            
    except Exception as e:
        logging.error(f"Translation from English failed: {str(e)}")
        st.error(f"Translation error: {str(e)}")
        return text

# --- Enhanced RAG Chain ---
def get_rag_chain(language: str = "English", embedding_preference: str = "local_multilingual") -> Optional[RetrievalQA]:
    """Get RAG chain with enhanced retrieval and embedding options"""
    result = get_vector_store(embedding_preference)
    if not result or len(result) != 2:
        return None, None
    
    vectorstore, model_used = result
    if not vectorstore:
        return None, None
    
    # Enhanced retriever with hybrid search
    retriever = vectorstore.as_retriever(
        search_type="mmr",  # Maximum Marginal Relevance
        search_kwargs={
            "k": 4,  # Reduced to minimize API calls
            "fetch_k": 12,
            "lambda_mult": 0.7
        }
    )
    
    if embedding_manager._can_use_api() and GOOGLE_API_KEY:
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro", 
            temperature=0.1,
            google_api_key=GOOGLE_API_KEY
        )
    else:
        st.warning("⚠️ Using limited functionality due to API quota.")
        # You could implement a local LLM fallback here if needed
        return None, None
    
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={
            "prompt": GOV_RAG_PROMPT,
            "document_variable_name": "context"
        },
        return_source_documents=True,
        verbose=False
    )
    
    return chain, model_used

# --- Main Application ---
def main():
    st.set_page_config(
        page_title="Government Decision Support RAG", 
        page_icon="🏛️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🏛️ Government Decision Support RAG System")
    st.markdown("*Multilingual AI Assistant for Local Government Decision Making*")
    
    # API Status indicator
    api_status = "🟢 Available" if embedding_manager._can_use_api() and GOOGLE_API_KEY else "🔴 Quota Exceeded"
    st.markdown(f"**API Status**: {api_status}")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Embedding model selection
        st.subheader("🧠 Embedding Model")
        embedding_option = st.selectbox(
            "Choose embedding approach:",
            list(EMBEDDING_OPTIONS.keys()),
            index=1,  # Default to local multilingual
            format_func=lambda x: EMBEDDING_OPTIONS[x]["name"],
            help="Local models don't require API quota and work offline"
        )
        
        # Show embedding model info
        selected_info = EMBEDDING_OPTIONS[embedding_option]
        st.info(f"""
        **Model**: {selected_info['name']}
        **Quality**: {selected_info['quality']}
        **Speed**: {selected_info['speed']}
        **API Required**: {selected_info['requires_api']}
        """)
        
        # Language selection
        selected_lang = st.selectbox(
            "Select your language:",
            SUPPORTED_LANGUAGES,
            help="The system will translate your query and response as needed"
        )
        
        # Document management
        st.header("📄 Document Management")
        
        # Check if data directory exists and show status
        if os.path.exists(DATA_DIR):
            files = [f for f in os.listdir(DATA_DIR) if not f.startswith('.')]
            st.success(f"📁 {len(files)} documents loaded")
            
            if files:
                st.markdown("**Available documents:**")
                for file in files[:5]:  # Show first 5
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
        if st.button("🔄 Rebuild Vector Store"):
            with st.spinner("Rebuilding vector store..."):
                if os.path.exists(VECTOR_DIR):
                    import shutil
                    shutil.rmtree(VECTOR_DIR)
                result = build_vector_store(embedding_option)
                if result and result[0]:
                    st.success("Vector store rebuilt successfully!")
                else:
                    st.error("Failed to rebuild vector store.")
            st.rerun()
    
    # Main query interface
    st.header("💬 Ask Your Question")
    
    query = st.text_area(
        "Enter your question about government policies, regulations, or decisions:",
        height=100,
        placeholder="Example: What are the budget allocations for infrastructure development in 2025?"
    )
    
    # Process query
    if st.button("🔍 Search & Analyze", type="primary"):
        if not query.strip():
            st.error("Please enter a question.")
            return
        
        try:
            # Translation step
            if selected_lang != "English":
                with st.spinner("🔄 Translating query to English..."):
                    en_query = translate_to_english(query, selected_lang)
                if en_query != query:
                    st.success(f"**Translated query:** {en_query}")
            else:
                en_query = query
            
            # RAG processing
            result = get_rag_chain(selected_lang, embedding_option)
            if not result or len(result) != 2:
                st.error("❌ RAG system not available. Please ensure documents are loaded and try a different embedding model.")
                return
                
            chain, model_used = result
            if not chain:
                st.error("❌ RAG system not available.")
                return
            
            with st.spinner("🧠 Processing with AI..."):
                rag_result = chain({
                    "query": en_query,
                    "domain": DOMAIN,
                    "data_types": ", ".join(DATA_TYPES),
                    "lang": selected_lang
                })
                
                # Track API usage
                if embedding_option == "google_api":
                    embedding_manager.track_api_call()
            
            answer_en = rag_result["result"]
            source_docs = rag_result["source_documents"]
            
            # Translate response back if needed
            if selected_lang != "English":
                with st.spinner("🔄 Translating response..."):
                    final_answer = translate_from_english(answer_en, selected_lang)
            else:
                final_answer = answer_en
            
            # Display results
            st.success(f"✅ Analysis Complete (using {EMBEDDING_OPTIONS[model_used]['name']})")
            
            # Main answer
            st.header("📋 Response")
            st.markdown(final_answer)
            
            # Sources and provenance
            if source_docs:
                st.header("📚 Sources & Evidence")
                
                for i, doc in enumerate(source_docs, 1):
                    with st.expander(f"📄 Source {i}: {doc.metadata.get('file_name', 'Unknown')}"):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.markdown("**Content:**")
                            content_preview = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
                            st.markdown(content_preview)
                        
                        with col2:
                            st.markdown("**Metadata:**")
                            metadata_display = {k: v for k, v in doc.metadata.items() 
                                              if k not in ['chunk_id', 'created_at']}
                            st.json(metadata_display)
                            
        except Exception as e:
            logging.error(f"Query processing failed: {str(e)}")
            if "429" in str(e):
                st.error("🚨 API quota exceeded. Please try again in a few minutes or use local embeddings.")
                st.info("💡 **Tip**: Switch to 'Local Multilingual' embeddings in the sidebar to avoid API limits.")
            else:
                st.error(f"❌ An error occurred: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("### 🔒 Security & Privacy")
    st.markdown("""
    - Local embeddings provide better privacy and no API limits
    - All queries are logged for audit and improvement purposes
    - System complies with local data protection regulations
    """)

if __name__ == "__main__":
    main()
