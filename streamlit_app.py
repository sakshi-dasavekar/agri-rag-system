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
except ImportError as e:
    st.error(f"Missing required packages: {e}")
    st.stop()

@st.cache_resource
def load_rag_system():
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        return None

    dataset_folders = glob.glob("rag_storage_filtered/*")
    if not dataset_folders:
        return None

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    all_vectorstores = []
    for folder in dataset_folders:
        dataset_name = os.path.basename(folder)
        embeddings_path = os.path.join(folder, "embeddings")
        if os.path.exists(embeddings_path):
            vectorstore = FAISS.load_local(embeddings_path, embeddings, allow_dangerous_deserialization=True)
            all_vectorstores.append((dataset_name, vectorstore))

    class CombinedRetriever(BaseRetriever):
        vectorstores: list

        def _get_relevant_documents(self, query, *, runnable_manager=None):
            all_docs = []
            for dataset_name, vectorstore in self.vectorstores:
                docs = vectorstore.similarity_search(query, k=5)
                for doc in docs:
                    doc.metadata['dataset'] = dataset_name
                all_docs.extend(docs)
            return all_docs[:5]

    combined_retriever = CombinedRetriever(vectorstores=all_vectorstores)
    llm = ChatGroq(temperature=0, model_name="llama3-70b-8192", api_key=groq_api_key)

    agricultural_prompt_template = """
    You are an expert agricultural advisor...
    Context:
    {context}

    Question: {input}

    Answer:
    """

    prompt = ChatPromptTemplate.from_template(agricultural_prompt_template)
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(combined_retriever, document_chain)

    return retrieval_chain

# Load system
rag_chain = load_rag_system()

# Check if input is provided in URL
query_params = st.query_params
api_input = query_params.get("input", [None])[0]

if api_input:
    if rag_chain is None:
        result = {"error": "System not ready"}
    else:
        try:
            response = rag_chain.invoke({"input": api_input})
            answer = response.get("answer", "No response.")
            result = {"reply": answer}
        except Exception as e:
            result = {"error": str(e)}

    # Return only JSON-like output
    st.markdown(
        f"""
        <pre>{json.dumps(result, ensure_ascii=False)}</pre>
        <script>
        const output = document.querySelector('pre');
        if (output) {{
            document.body.innerText = output.innerText;
        }}
        </script>
        """,
        unsafe_allow_html=True
    )
    st.stop()

# Fallback UI
st.title("🌾 Agricultural Expert System API")
st.write("This endpoint is designed for programmatic access.")
st.write("Append ?input=Your+Question to the URL to get JSON response.")