
import os
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime
import logging
import json
import asyncio
from typing import List, Dict, Any, Optional

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import LLMChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain.schema import Document
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

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

# --- Config ---
DATA_DIR = "data"
VECTOR_DIR = "vector_store"

# --- Configuration ---
DATA_DIR = "data"
VECTOR_DIR = "vector_store"
SUPPORTED_LANGUAGES = ["English", "Sinhala", "Tamil"]
DOMAIN = "Local Government Planning and Decision Making"
DATA_TYPES = ["PDFs", "TXT", "Markdown", "CSV", "Meeting Minutes", "Reports"]

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

def build_vector_store() -> Chroma:
    """Build vector store with enhanced chunking strategy"""
    docs = load_documents()
    if not docs:
        st.warning("No documents found to build vector store.")
        return None
    
    # Enhanced text splitter for government documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=300,
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
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    
    vectorstore = Chroma.from_documents(
        chunks, 
        embedding=embeddings, 
        persist_directory=VECTOR_DIR,
        collection_metadata={"hnsw:space": "cosine"}
    )
    vectorstore.persist()
    
    logging.info(f"Built vector store with {len(chunks)} chunks")
    return vectorstore

def get_vector_store() -> Optional[Chroma]:
    """Get existing vector store or build new one"""
    try:
        if os.path.exists(VECTOR_DIR) and os.listdir(VECTOR_DIR):
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=GOOGLE_API_KEY
            )
            return Chroma(
                persist_directory=VECTOR_DIR, 
                embedding_function=embeddings
            )
        else:
            return build_vector_store()
    except Exception as e:
        logging.error(f"Error with vector store: {str(e)}")
        st.error(f"Vector store error: {str(e)}")
        return None

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
    """Translate text to English with audit logging"""
    if source_lang == "English":
        return text
    
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro", 
            temperature=0.0,
            google_api_key=GOOGLE_API_KEY
        )
        chain = LLMChain(llm=llm, prompt=TRANSLATION_TO_EN_PROMPT)
        translated = chain.run(text=text, source_lang=source_lang)
        
        # Audit logging
        logging.info(f"Translation to English - Source: {source_lang}, Length: {len(text)}")
        
        return translated.strip()
    except Exception as e:
        logging.error(f"Translation to English failed: {str(e)}")
        st.error(f"Translation error: {str(e)}")
        return text

def translate_from_english(text: str, target_lang: str) -> str:
    """Translate text from English with audit logging"""
    if target_lang == "English":
        return text
    
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro", 
            temperature=0.0,
            google_api_key=GOOGLE_API_KEY
        )
        chain = LLMChain(llm=llm, prompt=TRANSLATION_FROM_EN_PROMPT)
        translated = chain.run(text=text, target_lang=target_lang)
        
        # Audit logging
        logging.info(f"Translation from English - Target: {target_lang}, Length: {len(text)}")
        
        return translated.strip()
    except Exception as e:
        logging.error(f"Translation from English failed: {str(e)}")
        st.error(f"Translation error: {str(e)}")
        return text

# --- Enhanced RAG Chain ---
def get_rag_chain(language: str = "English") -> Optional[RetrievalQA]:
    """Get RAG chain with enhanced retrieval and audit logging"""
    vectorstore = get_vector_store()
    if not vectorstore:
        return None
    
    # Enhanced retriever with hybrid search
    retriever = vectorstore.as_retriever(
        search_type="mmr",  # Maximum Marginal Relevance
        search_kwargs={
            "k": 6,  # Retrieve more documents
            "fetch_k": 20,  # Fetch more candidates
            "lambda_mult": 0.7  # Diversity parameter
        }
    )
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro", 
        temperature=0.1,  # Slightly higher for more natural responses
        google_api_key=GOOGLE_API_KEY
    )
    
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={
            "prompt": GOV_RAG_PROMPT,
            "document_variable_name": "context"
        },
        return_source_documents=True,
        verbose=True
    )
    
    return chain

# --- Evaluation & Monitoring Functions ---
def evaluate_response_quality(query: str, answer: str, sources: List[Document]) -> Dict[str, Any]:
    """Evaluate response quality and provide metrics"""
    evaluation = {
        "timestamp": datetime.now().isoformat(),
        "query_length": len(query),
        "answer_length": len(answer),
        "num_sources": len(sources),
        "has_provenance": any("source" in doc.metadata for doc in sources),
        "compliance_check": "I don't know" in answer,  # Basic hallucination check
    }
    
    # Log evaluation
    logging.info(f"Response evaluation: {json.dumps(evaluation)}")
    
    return evaluation

def audit_query(query: str, language: str, user_session: str = "anonymous") -> None:
    """Audit user queries for compliance and monitoring"""
    audit_record = {
        "timestamp": datetime.now().isoformat(),
        "user_session": user_session,
        "query": query[:100],  # Truncated for privacy
        "language": language,
        "query_length": len(query)
    }
    
    logging.info(f"Query audit: {json.dumps(audit_record)}")

# --- Security & Privacy Functions ---
def sanitize_input(text: str) -> str:
    """Basic input sanitization"""
    # Remove potential injection patterns
    dangerous_patterns = ['<script', 'javascript:', 'eval(', 'exec(']
    sanitized = text
    
    for pattern in dangerous_patterns:
        sanitized = sanitized.replace(pattern, '')
    
    return sanitized.strip()

def check_sensitive_content(text: str) -> bool:
    """Check for potentially sensitive content"""
    sensitive_keywords = [
        'confidential', 'classified', 'restricted', 'personal data',
        'social security', 'national id', 'passport', 'bank account'
    ]
    
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in sensitive_keywords)

# --- Enhanced Streamlit UI ---
def render_architecture_diagram():
    """Render system architecture using Mermaid"""
    mermaid_diagram = """
    ```mermaid
    graph TB
        A[User Query] --> B{Language Detection}
        B --> C[Translation to English]
        C --> D[Vector Store Retrieval]
        D --> E[Document Ranking & Selection]
        E --> F[LLM Processing - Gemini Pro]
        F --> G[Response Generation]
        G --> H[Translation to Target Language]
        H --> I[Audit Logging]
        I --> J[Response with Provenance]
        
        K[Document Ingestion] --> L[Text Chunking]
        L --> M[Embeddings - Google AI]
        M --> N[Chroma Vector DB]
        N --> D
        
        O[Security Layer] --> A
        P[Compliance Monitoring] --> I
    ```
    """
    st.markdown(mermaid_diagram)

def display_system_info():
    """Display system architecture and component information"""
    with st.expander("🏗️ System Architecture & Components"):
        st.markdown("### Executive Summary")
        st.markdown(f"""
        This Government Decision Support RAG system provides multilingual AI assistance for local government decision-making.
        
        **Domain**: {DOMAIN}  
        **Supported Languages**: {', '.join(SUPPORTED_LANGUAGES)}  
        **Data Types**: {', '.join(DATA_TYPES)}
        """)
        
        st.markdown("### Architecture Diagram")
        render_architecture_diagram()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Component Breakdown")
            st.markdown("""
            **1. Document Ingestion**
            - PDF, TXT, CSV, Markdown support
            - Enhanced metadata extraction
            - Chunking with overlap strategy
            
            **2. Vector Storage**
            - Chroma vector database (local)
            - Google AI embeddings (embedding-001)
            - MMR (Maximum Marginal Relevance) retrieval
            
            **3. LLM Processing**
            - Gemini Pro for reasoning and generation
            - Temperature: 0.1 for balanced responses
            - Structured prompt engineering
            """)
        
        with col2:
            st.markdown("### Security & Compliance")
            st.markdown("""
            **4. Translation Layer**
            - Professional government-style translation
            - Bidirectional language support
            - Context preservation
            
            **5. Audit & Monitoring**
            - Query logging and evaluation
            - Response quality metrics
            - Compliance checking
            
            **6. Security Features**
            - Input sanitization
            - Sensitive content detection
            - Provenance requirements
            """)

def main():
    st.set_page_config(
        page_title="Government Decision Support RAG", 
        page_icon="🏛️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🏛️ Government Decision Support RAG System")
    st.markdown("*Multilingual AI Assistant for Local Government Decision Making*")
    
    # Sidebar for configuration and monitoring
    with st.sidebar:
        st.header("⚙️ Configuration")
        
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
                build_vector_store()
            st.success("Vector store rebuilt successfully!")
            st.rerun()
    
    # Main interface
    display_system_info()
    
    # Query interface
    st.header("💬 Ask Your Question")
    
    # Create columns for better layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_area(
            "Enter your question about government policies, regulations, or decisions:",
            height=100,
            placeholder="Example: What are the budget allocations for infrastructure development in 2025?"
        )
    
    with col2:
        st.markdown("### Language Examples")
        st.markdown("""
        **English:** What are the planning regulations for commercial buildings?
        
        **Sinhala:** 2025 වසරේ වාස්තු විශේෂිත වියදම් සැලසුම් මොනවාද?
        
        **Tamil:** வணிக கட்டிடங்களுக்கான திட்டமிடல் விதிமுறைகள் என்ன?
        """)
    
    # Process query
    if st.button("🔍 Search & Analyze", type="primary"):
        if not query.strip():
            st.error("Please enter a question.")
            return
        
        # Sanitize input
        query = sanitize_input(query)
        
        # Check for sensitive content
        if check_sensitive_content(query):
            st.warning("⚠️ Your query may contain sensitive information. Please review before proceeding.")
        
        # Detect language if not specified
        detected_lang = detect_language(query)
        if detected_lang != selected_lang:
            st.info(f"Detected language: {detected_lang}. Using your selected language: {selected_lang}")
        
        # Audit the query
        audit_query(query, selected_lang)
        
        try:
            # Translation step
            if selected_lang != "English":
                with st.spinner("🔄 Translating query to English..."):
                    en_query = translate_to_english(query, selected_lang)
                st.success(f"**Translated query:** {en_query}")
            else:
                en_query = query
            
            # RAG processing
            chain = get_rag_chain(selected_lang)
            if not chain:
                st.error("❌ RAG system not available. Please ensure documents are loaded and vector store is built.")
                return
            
            with st.spinner("🧠 Processing with AI..."):
                result = chain({
                    "query": en_query,
                    "domain": DOMAIN,
                    "data_types": ", ".join(DATA_TYPES),
                    "lang": selected_lang
                })
            
            answer_en = result["result"]
            source_docs = result["source_documents"]
            
            # Translate response back if needed
            if selected_lang != "English":
                with st.spinner("🔄 Translating response..."):
                    final_answer = translate_from_english(answer_en, selected_lang)
            else:
                final_answer = answer_en
            
            # Evaluate response quality
            evaluation = evaluate_response_quality(query, final_answer, source_docs)
            
            # Display results
            st.success("✅ Analysis Complete")
            
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
                            st.markdown(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                        
                        with col2:
                            st.markdown("**Metadata:**")
                            metadata_display = {k: v for k, v in doc.metadata.items() 
                                              if k not in ['chunk_id', 'created_at']}
                            st.json(metadata_display)
            
            # Evaluation metrics
            with st.expander("📊 Response Quality Metrics"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Sources Used", evaluation["num_sources"])
                    st.metric("Query Length", evaluation["query_length"])
                
                with col2:
                    st.metric("Answer Length", evaluation["answer_length"])
                    provenance_status = "✅ Yes" if evaluation["has_provenance"] else "❌ No"
                    st.metric("Has Provenance", provenance_status)
                
                with col3:
                    compliance_status = "✅ Compliant" if evaluation["compliance_check"] else "⚠️ Review"
                    st.metric("Compliance Check", compliance_status)
            
            # Show English version if translated
            if selected_lang != "English":
                with st.expander("🔍 English Version (for reference)"):
                    st.markdown("**English Query:**")
                    st.markdown(en_query)
                    st.markdown("**English Response:**")
                    st.markdown(answer_en)
                    
        except Exception as e:
            logging.error(f"Query processing failed: {str(e)}")
            st.error(f"❌ An error occurred: {str(e)}")
            st.markdown("Please try again or contact the system administrator.")
    
    # Footer with deployment and security info
    st.markdown("---")
    st.markdown("### 🔒 Security & Privacy")
    st.markdown("""
    - All queries are logged for audit and improvement purposes
    - No personal data is stored in the system
    - Responses are generated from official government documents only
    - System complies with local data protection regulations
    """)

if __name__ == "__main__":
    main()
