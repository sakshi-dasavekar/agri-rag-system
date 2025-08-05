#!/usr/bin/env python3
"""
Agricultural RAG System - Streamlit Web Interface with API Endpoints
Deployment Ready for Cross-Platform Communication

This version exposes HTTP endpoints that can be called by FastAPI deployed on different platforms.
"""

import streamlit as st
import os
import glob
import json
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from contextlib import asynccontextmanager

# Add current directory to Python path for imports
sys.path.append(str(Path(__file__).parent))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import LangChain components
try:
    from langchain_groq import ChatGroq
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.prompts import ChatPromptTemplate
    from langchain.chains import create_retrieval_chain
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_core.retrievers import BaseRetriever
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError as e:
    st.error(f"❌ Missing required packages: {e}")
    st.error("Please install requirements: pip install -r requirements.txt")
    st.stop()

# Global variable to store the RAG system
_rag_system = None

# Pydantic models for API
class QuestionRequest(BaseModel):
    question: str

class QuestionResponse(BaseModel):
    answer: str
    question: str
    success: bool
    datasets_used: list = []
    total_vectors: int = 0
    error: str = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources"""
    # Load RAG system on startup
    global _rag_system
    _rag_system = load_rag_system()
    yield
    # Cleanup on shutdown
    _rag_system = None

# Create FastAPI app for HTTP endpoints
api_app = FastAPI(
    title="Agricultural Expert API",
    description="API for accessing the Agricultural Expert System",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
api_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_rag_system():
    """Get or create the RAG system singleton"""
    global _rag_system
    if _rag_system is None:
        _rag_system = load_rag_system()
    return _rag_system

def get_agricultural_answer(question: str) -> dict:
    """
    Get agricultural answer programmatically (for API calls)
    
    Args:
        question (str): The agricultural question
        
    Returns:
        dict: Response with answer and metadata
    """
    try:
        rag_system = get_rag_system()
        if rag_system is None:
            return {
                "error": "RAG system not loaded",
                "success": False
            }
        
        response = rag_system['chain'].invoke({"input": question})
        answer = response["answer"]
        
        return {
            "answer": answer,
            "question": question,
            "success": True,
            "datasets_used": rag_system['datasets'],
            "total_vectors": rag_system['total_vectors']
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "success": False,
            "question": question
        }

@st.cache_resource
def load_rag_system():
    """Load the RAG system with caching"""
    
    # Load environment variables
    groq_api_key = os.getenv("GROQ_API_KEY")
    
    if not groq_api_key:
        st.error("❌ Groq API key not found. Please set GROQ_API_KEY environment variable.")
        st.info("💡 For local development, create a .env file with your GROQ_API_KEY")
        return None
    
    try:
        # Load all available vector stores
        all_vectorstores = []
        dataset_folders = glob.glob("rag_storage_filtered/*")
        
        if not dataset_folders:
            st.error("❌ No dataset folders found in rag_storage_filtered/")
            st.info("💡 Please run the data processing script first")
            return None
        
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        loaded_datasets = []
        total_vectors = 0
        
        for folder in dataset_folders:
            dataset_name = os.path.basename(folder)
            embeddings_path = os.path.join(folder, "embeddings")
            
            if os.path.exists(embeddings_path):
                try:
                    vectorstore = FAISS.load_local(embeddings_path, embeddings, allow_dangerous_deserialization=True)
                    all_vectorstores.append((dataset_name, vectorstore))
                    loaded_datasets.append(dataset_name)
                    total_vectors += vectorstore.index.ntotal
                except Exception as e:
                    st.warning(f"⚠ Error loading {dataset_name}: {e}")
        
        if not all_vectorstores:
            st.error("❌ No vector stores could be loaded")
            return None
        
        # Create combined retriever
        class CombinedRetriever(BaseRetriever):
            vectorstores: list
            
            def _get_relevant_documents(self, query, *, runnable_manager=None):
                all_docs = []
                for dataset_name, vectorstore in self.vectorstores:
                    docs = vectorstore.similarity_search(query, k=5)
                    for doc in docs:
                        doc.metadata['dataset'] = dataset_name
                    all_docs.extend(docs)
                
                # Sort by relevance and return top 5
                return all_docs[:5]
        
        combined_retriever = CombinedRetriever(vectorstores=all_vectorstores)
        
        # Initialize LLM
        llm = ChatGroq(temperature=0, model_name="llama3-70b-8192", api_key=groq_api_key)
        
        # Create retrieval chain
        agricultural_prompt_template = """
        You are an expert agricultural advisor with deep knowledge of crop production, farming practices, disease management, and agricultural technologies. 
        Answer the following question based only on the provided agricultural knowledge base:

        Context:
        {context}

        Question: {input}

        Provide a comprehensive, practical answer that includes:
        - Specific recommendations based on the context
        - Any relevant location-specific information (state/district)
        - Seasonal considerations if mentioned
        - Practical steps or solutions
        - Mention which dataset the information comes from

        Answer:
        """
        
        prompt = ChatPromptTemplate.from_template(agricultural_prompt_template)
        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(combined_retriever, document_chain)
        
        return {
            'chain': retrieval_chain,
            'datasets': loaded_datasets,
            'total_vectors': total_vectors
        }
        
    except Exception as e:
        st.error(f"❌ Error loading RAG system: {e}")
        return None

# API Endpoints
@api_app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "message": "Agricultural Expert API",
        "version": "1.0.0",
        "endpoints": {
            "/agri": "GET - Get agricultural answer (query parameter)",
            "/agri/post": "POST - Get agricultural answer (JSON body)",
            "/health": "GET - Health check"
        }
    }

@api_app.get("/agri")
def get_agri_answer(query: str):
    """
    Get agricultural answer via GET request
    
    Args:
        query (str): The agricultural question
        
    Returns:
        dict: Response with answer and metadata
    """
    try:
        result = get_agricultural_answer(query)
        
        if not result.get("success", False):
            raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get answer: {str(e)}")

@api_app.post("/agri/post", response_model=QuestionResponse)
def post_agri_answer(request: QuestionRequest):
    """
    Get agricultural answer via POST request
    
    Args:
        request (QuestionRequest): The request containing the question
        
    Returns:
        QuestionResponse: Response with answer and metadata
    """
    try:
        result = get_agricultural_answer(request.question)
        
        if not result.get("success", False):
            raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))
        
        return QuestionResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get answer: {str(e)}")

@api_app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Agricultural Expert API"}

# Streamlit UI (only runs when this file is run directly)
def main():
    """Main Streamlit application"""
    
    # Page configuration
    st.set_page_config(
        page_title="Agricultural Expert System",
        page_icon="🌾",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for better styling
    st.markdown("""
    <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            color: #2E8B57;
            text-align: center;
            margin-bottom: 2rem;
        }
        .sub-header {
            font-size: 1.5rem;
            color: #556B2F;
            margin-bottom: 1rem;
        }
        .info-box {
            background-color: #F0F8FF;
            padding: 1rem;
            border-radius: 10px;
            border-left: 5px solid #2E8B57;
            margin: 1rem 0;
        }
        .dataset-info {
            background-color: #F5F5F5;
            padding: 0.5rem;
            border-radius: 5px;
            margin: 0.25rem 0;
        }
        .stButton > button {
            background-color: #2E8B57;
            color: white;
            border-radius: 10px;
            padding: 0.5rem 2rem;
            font-weight: bold;
        }
        .stButton > button:hover {
            background-color: #3CB371;
        }
        .error-box {
            background-color: #FFE6E6;
            padding: 1rem;
            border-radius: 10px;
            border-left: 5px solid #FF4444;
            margin: 1rem 0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🌾 Agricultural Expert System</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Your AI-powered agricultural advisor</p>', unsafe_allow_html=True)
    
    # Load RAG system
    with st.spinner("🔄 Loading Agricultural Expert System..."):
        rag_system = get_rag_system()
    
    if rag_system is None:
        st.stop()
    
    # Sidebar with system info
    with st.sidebar:
        st.markdown("## 📊 System Information")
        
        # Dataset info
        st.markdown("### 📁 Available Datasets")
        for dataset in rag_system['datasets']:
            st.markdown(f"• {dataset}")
        
        st.markdown(f"### 🔢 Total Vectors: {rag_system['total_vectors']:,}")
        
        # Sample questions
        st.markdown("### 💡 Sample Questions")
        sample_questions = [
            "How to control Ranikhet disease in poultry?",
            "What are the best practices for field preparation?",
            "How to manage crop diseases effectively?",
            "What are the recommended fertilizers for different crops?",
            "How to prepare poultry feed at home?",
            "How to grow tomatoes in Maharashtra?",
            "What crops grow best in Rajasthan?",
            "How to practice crop rotation effectively?",
            "What are natural ways to control pests?",
            "How to improve soil fertility organically?"
        ]
        
        for question in sample_questions:
            if st.button(question, key=f"sample_{hash(question)}"):
                st.session_state.user_question = question
                st.rerun()
    
    # Main content area
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Question input
        st.markdown('<h2 class="sub-header">🤔 Ask Your Agricultural Question</h2>', unsafe_allow_html=True)
        
        # Initialize session state
        if 'user_question' not in st.session_state:
            st.session_state.user_question = ""
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Question input
        user_question = st.text_area(
            "Enter your agricultural question:",
            value=st.session_state.user_question,
            height=100,
            placeholder="e.g., How to control Ranikhet disease in poultry?"
        )
        
        # Submit button
        col_a, col_b, col_c = st.columns([1, 1, 1])
        with col_b:
            submit_button = st.button("🚀 Get Expert Answer", use_container_width=True)
        
        # Process question
        if submit_button and user_question.strip():
            st.session_state.user_question = user_question
            
            with st.spinner("🔍 Searching agricultural knowledge base..."):
                try:
                    response = rag_system['chain'].invoke({"input": user_question})
                    answer = response["answer"]
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        'question': user_question,
                        'answer': answer
                    })
                    
                except Exception as e:
                    st.error(f"❌ Error getting answer: {e}")
                    st.stop()
        
        # Display chat history
        if st.session_state.chat_history:
            st.markdown('<h3 class="sub-header">💬 Conversation History</h3>', unsafe_allow_html=True)
            
            for i, chat in enumerate(reversed(st.session_state.chat_history)):
                with st.expander(f"Q{i+1}: {chat['question'][:50]}...", expanded=True):
                    st.markdown("*Question:*")
                    st.write(chat['question'])
                    st.markdown("*Answer:*")
                    st.markdown(chat['answer'])
                    
                    # Add copy button
                    if st.button(f"📋 Copy Answer {i+1}", key=f"copy_{i}"):
                        st.write("✅ Answer copied to clipboard!")
        
        # Clear chat history
        if st.session_state.chat_history and st.button("🗑 Clear History"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666; padding: 1rem;">
            <p>🌾 Agricultural Expert System | Powered by LangChain & Groq</p>
            <p>Built with ❤ for farmers and agricultural professionals</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    # Check if we should run the API server or Streamlit UI
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "api":
        # Run as API server
        uvicorn.run(api_app, host="0.0.0.0", port=8501)
    else:
        # Run as Streamlit UI
        main()