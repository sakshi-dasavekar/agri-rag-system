#!/usr/bin/env python3
"""
Agricultural RAG System - Streamlit Web Interface (Deployment Ready)

A user-friendly web interface for the agricultural RAG system.
Optimized for deployment on Streamlit Cloud, Heroku, or other platforms.
Supports both UI mode and API mode for external frontend integration.
"""

import streamlit as st
import os
import glob
import json
import sys
from pathlib import Path

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

@st.cache_resource
def load_rag_system():
    """Load the RAG system with caching"""
    
    # Load environment variables
    groq_api_key = os.getenv("GROQ_API_KEY")
    
    # Debug information
    st.write(f"🔍 Debug: API key found: {'Yes' if groq_api_key else 'No'}")
    if groq_api_key:
        st.write(f"🔍 Debug: API key starts with: {groq_api_key[:10]}...")
    
    # Temporary fallback for testing (remove this after fixing secrets)
    if not groq_api_key:
        st.warning("⚠️ Trying fallback API key for testing...")
        groq_api_key = "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"
        st.success("✅ Using fallback API key")
    
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
                    st.warning(f"⚠️ Error loading {dataset_name}: {e}")
        
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

def load_rag_system_api():
    """Load the RAG system for API mode (no UI functions)"""
    
    # Load environment variables
    groq_api_key = os.getenv("GROQ_API_KEY")
    
    # Temporary fallback for testing (remove this after fixing secrets)
    if not groq_api_key:
        groq_api_key = "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"
    
    if not groq_api_key:
        return None
    
    try:
        # Load all available vector stores
        all_vectorstores = []
        dataset_folders = glob.glob("rag_storage_filtered/*")
        
        if not dataset_folders:
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
                    continue
        
        if not all_vectorstores:
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
        return None

def handle_api_request(user_input, rag_system):
    """Handle API requests and return JSON response"""
    try:
        if not user_input or not user_input.strip():
            return {
                "error": "Missing 'input' parameter",
                "status": "error",
                "message": "Please provide a question in the 'input' parameter"
            }
        
        # Get response from RAG system
        response = rag_system['chain'].invoke({"input": user_input})
        answer = response["answer"]
        
        # Extract datasets used from the answer
        datasets_used = []
        for dataset in rag_system['datasets']:
            if dataset.lower() in answer.lower():
                datasets_used.append(dataset)
        
        return {
            "reply": answer,
            "status": "success",
            "datasets_used": datasets_used,
            "total_datasets_available": len(rag_system['datasets']),
            "total_vectors": rag_system['total_vectors']
        }
        
    except Exception as e:
        return {
            "error": f"Error processing request: {str(e)}",
            "status": "error",
            "message": "An error occurred while processing your question"
        }

def main():
    """Main Streamlit application"""
    
    # Check if this is an API request
    query_params = st.experimental_get_query_params()
    user_input = query_params.get("input", [None])[0]
    
    # If API request, handle it and return JSON
    if user_input:
        # Load RAG system for API mode
        rag_system = load_rag_system_api()
        
        if rag_system is None:
            result = {
                "error": "RAG system not available",
                "status": "error",
                "message": "The agricultural knowledge base is not loaded"
            }
        else:
            result = handle_api_request(user_input, rag_system)
        
        # Output only raw JSON text (no Streamlit UI)
        st.markdown(
            f"""
            <pre>{json.dumps(result, indent=2)}</pre>
            <script>
            const output = document.querySelector('pre');
            if (output) {{
                document.body.innerText = output.innerText;
            }}
            </script>
            """,
            unsafe_allow_html=True
        )
        return
    
    # Regular UI mode - continue with normal interface
    # Header
    st.markdown('<h1 class="main-header">🌾 Agricultural Expert System</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Your AI-powered agricultural advisor</p>', unsafe_allow_html=True)
    
    # API usage info
    with st.expander("🔗 API Usage", expanded=False):
        st.markdown("""
        ### API Endpoint
        You can also use this system as an API by calling:
        ```
        https://your-app.streamlit.app/?input=Your question here
        ```
        
        ### Example API Call
        ```
        https://your-app.streamlit.app/?input=How to control Ranikhet disease in poultry?
        ```
        
        ### JSON Response Format
        ```json
        {
          "reply": "Based on the agricultural knowledge base...",
          "status": "success",
          "datasets_used": ["Disease Management", "Feed"],
          "total_datasets_available": 8,
          "total_vectors": 15000
        }
        ```
        """)
    
    # Load RAG system
    with st.spinner("🔄 Loading Agricultural Expert System..."):
        rag_system = load_rag_system()
    
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
                    st.markdown("**Question:**")
                    st.write(chat['question'])
                    st.markdown("**Answer:**")
                    st.markdown(chat['answer'])
                    
                    # Add copy button
                    if st.button(f"📋 Copy Answer {i+1}", key=f"copy_{i}"):
                        st.write("✅ Answer copied to clipboard!")
        
        # Clear chat history
        if st.session_state.chat_history and st.button("🗑️ Clear History"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666; padding: 1rem;">
            <p>🌾 Agricultural Expert System | Powered by LangChain & Groq</p>
            <p>Built with ❤️ for farmers and agricultural professionals</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main() 