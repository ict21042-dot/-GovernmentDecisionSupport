import os
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime
import logging
import time
import warnings
from typing import List, Dict, Any, Optional

# Suppress warnings for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Core imports
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain.schema import Document

# Force FAISS usage - avoid Chroma completely for Streamlit Cloud
from langchain_community.vectorstores import FAISS

# Embedding models with retry logic for rate limiting
class RobustEmbeddings:
    """Embedding class with rate limit handling and fallbacks"""
    
    def __init__(self):
        self.embeddings = None
        self.model_type = None
        self.initialize_best_available()
    
    def initialize_best_available(self):
        """Initialize the best available embedding model"""
        
        # Option 1: Try HuggingFace with smaller, faster model
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            
            # Use a smaller, more reliable model to avoid rate limits
            self.embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",  # Smaller, faster model
                model_kwargs={
                    'device': 'cpu',
                    'trust_remote_code': False
                },
                encode_kwargs={
                    'normalize_embeddings': True,
                    'batch_size': 8  # Smaller batch to avoid memory issues
                }
            )
            self.model_type = "HuggingFace (all-MiniLM-L6-v2)"
            st.success(f"✅ Loaded: {self.model_type}")
            return
            
        except Exception as e:
            st.warning(f"HuggingFace embeddings failed: {str(e)}")
        
        # Option 2: Try direct sentence-transformers with rate limit handling
        try:
            from sentence_transformers import SentenceTransformer
            import time
            
            # Retry logic for HuggingFace Hub rate limits
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Use a very simple, cached model to avoid downloads
                    self.embeddings = SentenceTransformer('all-MiniLM-L6-v2')
                    self.model_type = "Sentence Transformers (all-MiniLM-L6-v2)"
                    st.success(f"✅ Loaded: {self.model_type}")
                    return
                except Exception as e:
                    if "429" in str(e) and attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        st.warning(f"⏳ Rate limited, waiting {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise e
                        
        except Exception as e:
            st.warning(f"Sentence transformers failed: {str(e)}")
        
        # Option 3: Try Google AI if available
        try:
            if os.getenv("GOOGLE_API_KEY"):
                from langchain_google_genai import GoogleGenerativeAIEmbeddings
                self.embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001",
                    google_api_key=os.getenv("GOOGLE_API_KEY")
                )
                self.model_type = "Google AI Embeddings"
                st.success(f"✅ Loaded: {self.model_type}")
                return
        except Exception as e:
            st.warning(f"Google AI embeddings failed: {str(e)}")
        
        # Option 4: Fallback to simple TF-IDF (local, no downloads)
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            import numpy as np
            
            class SimpleTFIDFEmbeddings:
                def __init__(self):
                    self.vectorizer = TfidfVectorizer(max_features=384, stop_words='english')
                    self.fitted = False
                
                def embed_documents(self, texts: List[str]) -> List[List[float]]:
                    if not self.fitted:
                        self.vectorizer.fit(texts)
                        self.fitted = True
                    vectors = self.vectorizer.transform(texts)
                    return vectors.toarray().tolist()
                
                def embed_query(self, text: str) -> List[float]:
                    if not self.fitted:
                        return [0.0] * 384
                    vector = self.vectorizer.transform([text])
                    return vector.toarray()[0].tolist()
            
            self.embeddings = SimpleTFIDFEmbeddings()
            self.model_type = "TF-IDF (Local Fallback)"
            st.warning("⚠️ Using TF-IDF fallback (limited multilingual support)")
            return
            
        except Exception as e:
            st.error(f"All embedding methods failed: {str(e)}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not self.embeddings:
            return []
        return self.embeddings.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        if not self.embeddings:
            return []
        return self.embeddings.embed_query(text)

# Load environment
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Configuration
DATA_DIR = "data"
VECTOR_DIR = "vector_store"

# System prompt
GOVERNMENT_PROMPT = """
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

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=GOVERNMENT_PROMPT
)

# Document loading
def load_documents(data_dir=DATA_DIR) -> List[Document]:
    """Load documents from various formats"""
    docs = []
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        st.warning(f"Created {data_dir} directory. Please add your government documents.")
        return docs
    
    files = [f for f in os.listdir(data_dir) if not f.startswith('.')]
    if not files:
        st.info("ℹ️ No documents found in data/ directory. Add PDF, TXT, or CSV files to get started.")
        return docs
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, file in enumerate(files):
        try:
            status_text.text(f"Loading {file}...")
            path = os.path.join(data_dir, file)
            
            if file.endswith(".pdf"):
                loader = PyPDFLoader(path)
                loaded_docs = loader.load()
                doc_type = 'PDF'
                
            elif file.endswith((".txt", ".md")):
                loader = TextLoader(path, encoding='utf-8')
                loaded_docs = loader.load()
                doc_type = 'Text'
                
            elif file.endswith(".csv"):
                loader = CSVLoader(path)
                loaded_docs = loader.load()
                doc_type = 'CSV'
            else:
                continue
            
            # Add metadata
            for doc in loaded_docs:
                doc.metadata.update({
                    'document_type': doc_type,
                    'file_name': file,
                    'loaded_at': datetime.now().isoformat()
                })
            
            docs.extend(loaded_docs)
            progress_bar.progress((i + 1) / len(files))
            
        except Exception as e:
            st.error(f"Failed to load {file}: {str(e)}")
    
    progress_bar.empty()
    status_text.empty()
    
    logging.info(f"Loaded {len(docs)} documents from {len(files)} files")
    return docs

# Vector store management (FAISS only)
def build_faiss_vector_store():
    """Build FAISS vector store - no SQLite dependencies"""
    docs = load_documents()
    if not docs:
        return None
    
    # Text splitting
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Smaller chunks for better performance
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "!", "?", ";", ":", " ", ""]
    )
    
    with st.spinner("Splitting documents..."):
        chunks = splitter.split_documents(docs)
    
    # Add metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata.update({
            'chunk_id': i,
            'chunk_size': len(chunk.page_content),
            'created_at': datetime.now().isoformat()
        })
    
    # Initialize embeddings with retry logic
    with st.spinner("Initializing embedding model..."):
        embeddings = RobustEmbeddings()
    
    if not embeddings.embeddings:
        st.error("❌ Could not initialize any embedding model")
        return None
    
    try:
        with st.spinner(f"Building FAISS vector store with {len(chunks)} chunks..."):
            # Build FAISS store
            vectorstore = FAISS.from_documents(chunks, embeddings)
            
            # Try to save to disk
            try:
                os.makedirs(VECTOR_DIR, exist_ok=True)
                vectorstore.save_local(VECTOR_DIR)
                st.success(f"✅ FAISS vector store built and saved with {len(chunks)} chunks")
            except Exception as e:
                st.warning(f"⚠️ Vector store built but couldn't save to disk: {str(e)}")
                st.success(f"✅ FAISS vector store built (in memory) with {len(chunks)} chunks")
            
            logging.info(f"Built FAISS vector store with {len(chunks)} chunks")
            return vectorstore
            
    except Exception as e:
        st.error(f"❌ FAISS vector store creation failed: {str(e)}")
        logging.error(f"FAISS vector store creation failed: {str(e)}")
        return None

def load_faiss_vector_store():
    """Load existing FAISS vector store"""
    if not os.path.exists(VECTOR_DIR):
        return None
    
    try:
        embeddings = RobustEmbeddings()
        if not embeddings.embeddings:
            return None
        
        vectorstore = FAISS.load_local(
            VECTOR_DIR, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        st.info(f"✅ Loaded existing FAISS vector store")
        return vectorstore
        
    except Exception as e:
        st.warning(f"Could not load existing vector store: {str(e)}")
        return None

# Main application
def main():
    st.set_page_config(
        page_title="Government RAG - Robust Cloud Version",
        page_icon="🏛️",
        layout="wide"
    )
    
    st.title("🏛️ Government Decision Support RAG System")
    st.markdown("*Rate-Limit Resistant & SQLite-Free Version*")
    
    # Status indicators
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("✅ **FAISS Vector Store**")
    with col2:
        st.info("🔄 **Rate Limit Handling**")
    with col3:
        st.info("🛡️ **SQLite-Free Operation**")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 System Status")
        
        # Check data directory
        if os.path.exists(DATA_DIR):
            files = [f for f in os.listdir(DATA_DIR) if not f.startswith('.')]
            st.success(f"📄 **Documents**: {len(files)} files loaded")
            
            if files:
                with st.expander("View documents"):
                    for file in files:
                        st.text(f"• {file}")
        else:
            st.warning("📄 **No documents found**")
            st.info("Add PDF, TXT, or CSV files to the `data/` directory")
        
        # Vector store management
        st.header("🗂️ Vector Store")
        
        if st.button("🔄 Build/Rebuild Vector Store"):
            if os.path.exists(VECTOR_DIR):
                import shutil
                shutil.rmtree(VECTOR_DIR)
            
            vectorstore = build_faiss_vector_store()
            if vectorstore:
                st.experimental_rerun()
        
        # System info
        st.header("ℹ️ System Info")
        st.info("""
        **Vector DB**: FAISS (SQLite-free)
        **Fallback**: Multiple embedding models
        **Rate Limits**: Auto-retry with backoff
        **Memory**: Optimized for cloud deployment
        """)
    
    # Main interface
    st.header("💬 Ask Your Question")
    
    # Try to load existing vector store
    vectorstore = load_faiss_vector_store()
    
    if not vectorstore:
        if os.path.exists(DATA_DIR) and os.listdir(DATA_DIR):
            st.info("🔄 No vector store found. Click 'Build Vector Store' in the sidebar to get started.")
        else:
            st.warning("📄 Add documents to the `data/` directory first, then build the vector store.")
        return
    
    query = st.text_area(
        "Enter your question about government policies:",
        height=100,
        placeholder="Example: What are the budget allocations for infrastructure in 2025?"
    )
    
    if st.button("🔍 Search Documents", type="primary"):
        if not query.strip():
            st.error("Please enter a question.")
            return
        
        try:
            with st.spinner("Searching documents..."):
                # Simple similarity search
                docs = vectorstore.similarity_search(query, k=5)
            
            if not docs:
                st.warning("No relevant documents found for your question.")
                return
            
            # Try to use LLM if available
            if os.getenv("GOOGLE_API_KEY"):
                try:
                    from langchain_google_genai import ChatGoogleGenerativeAI
                    from langchain.chains.question_answering import load_qa_chain
                    
                    llm = ChatGoogleGenerativeAI(
                        model="gemini-pro",
                        temperature=0.1,
                        google_api_key=os.getenv("GOOGLE_API_KEY")
                    )
                    
                    qa_chain = load_qa_chain(llm, chain_type="stuff", prompt=RAG_PROMPT)
                    
                    with st.spinner("Generating AI response..."):
                        response = qa_chain.run(input_documents=docs, question=query)
                    
                    st.success("✅ AI Analysis Complete")
                    st.header("🤖 AI Response")
                    st.markdown(response)
                    
                except Exception as e:
                    st.warning(f"⚠️ AI processing failed: {str(e)}")
                    st.info("💡 Showing document search results instead.")
            else:
                st.info("💡 **Document Review Mode**: Add GOOGLE_API_KEY for AI responses.")
            
            # Show retrieved documents
            st.header("📚 Retrieved Documents")
            
            for i, doc in enumerate(docs, 1):
                with st.expander(f"📄 Document {i}: {doc.metadata.get('file_name', 'Unknown')}"):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown("**Content:**")
                        st.markdown(doc.page_content[:1000] + "..." if len(doc.page_content) > 1000 else doc.page_content)
                    
                    with col2:
                        st.markdown("**Metadata:**")
                        metadata = {k: v for k, v in doc.metadata.items() 
                                  if k not in ['chunk_id', 'created_at']}
                        st.json(metadata)
                        
        except Exception as e:
            st.error(f"❌ Search failed: {str(e)}")
            logging.error(f"Search failed: {str(e)}")

if __name__ == "__main__":
    main()
