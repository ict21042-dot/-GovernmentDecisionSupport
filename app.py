import os
import streamlit as st
from dotenv import load_dotenv

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# Load environment
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Config ---
DATA_DIR = "data"
VECTOR_DIR = "vector_store"

# --- Custom Prompt (Government Decision Support) ---
GOV_PROMPT = """
You are an AI architect specializing in Retrieval-Augmented Generation (RAG) for public sector and government decision support. 
You must prioritize accuracy, transparency, compliance, and explainability.

### Rules:
- Always cite sources (document name, page, date if available).
- If unsure, respond: "I don’t know — requires further validation."
- Never hallucinate regulations, budgets, or laws.
- Ensure outputs are unbiased and legally compliant.

Context:
{context}

Question:
{query}

Answer:
"""

prompt_template = PromptTemplate(
    input_variables=["context", "query"],
    template=GOV_PROMPT,
)

# --- Functions ---
def load_documents(data_dir=DATA_DIR):
    docs = []
    for file in os.listdir(data_dir):
        path = os.path.join(data_dir, file)
        if file.endswith(".pdf"):
            loader = PyPDFLoader(path)
            docs.extend(loader.load())
        elif file.endswith(".txt"):
            loader = TextLoader(path)
            docs.extend(loader.load())
    return docs

def build_vector_store():
    docs = load_documents()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=VECTOR_DIR)
    vectorstore.persist()
    return vectorstore

def get_vector_store():
    if os.path.exists(VECTOR_DIR):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        return Chroma(persist_directory=VECTOR_DIR, embedding_function=embeddings)
    else:
        return build_vector_store()

def get_rag_chain():
    vectorstore = get_vector_store()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.0)
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template},
        return_source_documents=True,
    )
    return chain

# --- Streamlit UI ---
st.set_page_config(page_title="Gov Decision Support RAG", layout="wide")
st.title("🏛️ Local Government Decision Support (RAG + Gemini)")

query = st.text_input("Enter your policy or decision-making question:")

if st.button("Ask"):
    if query:
        chain = get_rag_chain()
        with st.spinner("Retrieving and reasoning..."):
            result = chain({"query": query})
        st.subheader("Answer")
        st.write(result["result"])

        st.subheader("Sources")
        for doc in result["source_documents"]:
            st.markdown(f"- **{doc.metadata.get('source','unknown')}** | Page: {doc.metadata.get('page','?')}")
