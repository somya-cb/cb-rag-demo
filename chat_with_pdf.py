import tempfile
import os
import time
from datetime import timedelta

import streamlit as st
from couchbase.cluster import Cluster
from couchbase.auth import PasswordAuthenticator
from couchbase.options import ClusterOptions

from langchain_couchbase.vectorstores import CouchbaseSearchVectorStore
from langchain_couchbase.cache import CouchbaseCache
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.globals import set_llm_cache


def parse_bool(value: str) -> bool:
    """Parse boolean values from environment variables."""
    return value.lower() in ("yes", "true", "t", "1")


def check_environment_variable(variable_name: str) -> None:
    """Check if environment variable is set."""
    if variable_name not in os.environ:
        st.error(
            f"{variable_name} environment variable is not set. Please add it to the secrets.toml file"
        )
        st.stop()


def save_to_couchbase_without_embeddings(
    uploaded_file, cluster, db_bucket: str, db_scope: str, db_collection: str
) -> None:
    """
    Chunk the PDF & store it in Couchbase WITHOUT embeddings.
    The Eventing function will automatically generate embeddings asynchronously.
    """
    if uploaded_file is not None:
        temp_dir = tempfile.TemporaryDirectory()
        temp_file_path = os.path.join(temp_dir.name, uploaded_file.name)

        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
            loader = PyPDFLoader(temp_file_path)
            docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=150
        )
        doc_pages = text_splitter.split_documents(docs)

        bucket = cluster.bucket(db_bucket)
        scope = bucket.scope(db_scope)
        collection = scope.collection(db_collection)

        for i, doc in enumerate(doc_pages):
            doc_id = f"{uploaded_file.name}_{i}_{int(time.time())}"
            doc_dict = {
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            }
            collection.upsert(doc_id, doc_dict)

        st.info(
            f"✅ PDF loaded into Couchbase in {len(doc_pages)} chunks. "
            "Embeddings will be generated automatically by Eventing service."
        )
        st.warning(
            "⏳ Note: It may take a few seconds for the Eventing function to generate embeddings for all chunks."
        )


@st.cache_resource(show_spinner="Connecting to Vector Store")
def get_vector_store(
    _cluster, db_bucket: str, db_scope: str, db_collection: str, _embedding, index_name: str
):
    """Return the Couchbase vector store."""
    vector_store = CouchbaseSearchVectorStore(
        cluster=_cluster,
        bucket_name=db_bucket,
        scope_name=db_scope,
        collection_name=db_collection,
        embedding=_embedding,
        index_name=index_name,
    )
    return vector_store


@st.cache_resource(show_spinner="Connecting to Cache")
def get_cache(_cluster, db_bucket: str, db_scope: str, cache_collection: str):
    """Return the Couchbase cache."""
    cache = CouchbaseCache(
        cluster=_cluster,
        bucket_name=db_bucket,
        scope_name=db_scope,
        collection_name=cache_collection,
    )
    return cache


@st.cache_resource(show_spinner="Connecting to Couchbase")
def connect_to_couchbase(connection_string: str, db_username: str, db_password: str):
    """Connect to Couchbase."""
    auth = PasswordAuthenticator(db_username, db_password)
    options = ClusterOptions(auth)
    cluster = Cluster(connection_string, options)
    cluster.wait_until_ready(timedelta(seconds=5))
    return cluster


def stream_string(s: str, chunk_size: int = 10):
    """Stream a string with a delay to simulate streaming."""
    for i in range(0, len(s), chunk_size):
        yield s[i : i + chunk_size]
        time.sleep(0.02)


if __name__ == "__main__":
    st.set_page_config(
        page_title="Chat with your PDF using Langchain, Couchbase & Ollama",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="auto",
        menu_items=None,
    )

    AUTH_ENABLED = parse_bool(os.getenv("AUTH_ENABLED", "False"))

    if not AUTH_ENABLED:
        st.session_state.auth = True
    else:
        if "auth" not in st.session_state:
            st.session_state.auth = False

        AUTH = os.getenv("LOGIN_PASSWORD")
        check_environment_variable("LOGIN_PASSWORD")

        user_pwd = st.text_input("Enter password", type="password")
        pwd_submit = st.button("Submit")

        if pwd_submit and user_pwd == AUTH:
            st.session_state.auth = True
        elif pwd_submit and user_pwd != AUTH:
            st.error("Incorrect password")

    if st.session_state.auth:
        DB_CONN_STR = os.getenv("DB_CONN_STR")
        DB_USERNAME = os.getenv("DB_USERNAME")
        DB_PASSWORD = os.getenv("DB_PASSWORD")
        DB_BUCKET = os.getenv("DB_BUCKET")
        DB_SCOPE = os.getenv("DB_SCOPE")
        DB_COLLECTION = os.getenv("DB_COLLECTION")
        INDEX_NAME = os.getenv("INDEX_NAME")
        CACHE_COLLECTION = os.getenv("CACHE_COLLECTION")

        check_environment_variable("HUGGINGFACEHUB_API_TOKEN")
        check_environment_variable("DB_CONN_STR")
        check_environment_variable("DB_USERNAME")
        check_environment_variable("DB_PASSWORD")
        check_environment_variable("DB_BUCKET")
        check_environment_variable("DB_SCOPE")
        check_environment_variable("DB_COLLECTION")
        check_environment_variable("INDEX_NAME")
        check_environment_variable("CACHE_COLLECTION")

        embedding = HuggingFaceEndpointEmbeddings(
            model="BAAI/bge-base-en-v1.5",
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        )

        cluster = connect_to_couchbase(DB_CONN_STR, DB_USERNAME, DB_PASSWORD)

        vector_store = get_vector_store(
            cluster, DB_BUCKET, DB_SCOPE, DB_COLLECTION, embedding, INDEX_NAME
        )

        retriever = vector_store.as_retriever()

        cache = get_cache(cluster, DB_BUCKET, DB_SCOPE, CACHE_COLLECTION)
        set_llm_cache(cache)

        template = """You are a helpful bot. If you cannot answer based on the context provided, respond with a generic answer. Answer the question as truthfully as possible using the context below:
        {context}

        Question: {question}"""

        prompt = ChatPromptTemplate.from_template(template)

        llm = Ollama(model="llama3.2", temperature=0.1)

        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        template_without_rag = """You are a helpful bot. Answer the question as truthfully as possible.

        Question: {question}"""

        prompt_without_rag = ChatPromptTemplate.from_template(template_without_rag)
        llm_without_rag = Ollama(model="llama3.2")

        chain_without_rag = (
            {"question": RunnablePassthrough()}
            | prompt_without_rag
            | llm_without_rag
            | StrOutputParser()
        )

        couchbase_logo = "https://emoji.slack-edge.com/T024FJS4M/couchbase/4a361e948b15ed91.png"

        st.title("Chat with PDF")
        st.markdown(
            "Answers with [Couchbase logo](https://emoji.slack-edge.com/T024FJS4M/couchbase/4a361e948b15ed91.png) are generated using *RAG* while 🤖️ are generated by pure *LLM (Llama 3.2 via Ollama)*"
        )

        with st.sidebar:
            st.header("Upload your PDF")
            st.info(
                "📝 **How it works**: PDF chunks are stored in Couchbase, "
                "then the Eventing Service automatically generates embeddings asynchronously."
            )

            with st.form("upload pdf"):
                uploaded_file = st.file_uploader(
                    "Choose a PDF.",
                    help="The document will be chunked and stored. Embeddings will be generated automatically by the Eventing service.",
                    type="pdf",
                )
                submitted = st.form_submit_button("Upload")
                if submitted:
                    save_to_couchbase_without_embeddings(
                        uploaded_file, cluster, DB_BUCKET, DB_SCOPE, DB_COLLECTION
                    )

            st.subheader("How does it work?")
            st.markdown(
                """
                For each question, you will get two answers: 
                * one using RAG ([Couchbase logo](https://emoji.slack-edge.com/T024FJS4M/couchbase/4a361e948b15ed91.png))
                * one using pure LLM - Ollama Llama 3.2 (🤖️). 
                """
            )

            st.markdown(
                """
                **Architecture:**
                1. **Document Upload**: PDF is chunked and stored in Couchbase (without embeddings)
                2. **Eventing Service**: Automatically generates embeddings for new documents asynchronously
                3. **Query Time**: User questions are embedded using HuggingFace, then we search for similar document embeddings
                4. **RAG**: Relevant context is passed to Ollama LLM for answer generation
                """
            )

            if st.checkbox("View Code"):
                st.write(
                    "View the code here: [GitHub](https://github.com/couchbase-examples/rag-demo)"
                )

            st.divider()
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": "Hi, I'm a chatbot who can chat with the PDF. How can I help you?",
                        "avatar": "🤖️",
                    }
                )
                st.rerun()

        if "messages" not in st.session_state:
            st.session_state.messages = []
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": "Hi, I'm a chatbot who can chat with the PDF. How can I help you?",
                    "avatar": "🤖️",
                }
            )

        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar=message["avatar"]):
                st.markdown(message["content"])

        if question := st.chat_input("Ask a question based on the PDF"):
            st.chat_message("user").markdown(question)

            st.session_state.messages.append(
                {"role": "user", "content": question, "avatar": "👤"}
            )

            with st.chat_message("assistant", avatar=couchbase_logo):
                rag_response = chain.invoke(question)
                st.write_stream(stream_string(rag_response))

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": rag_response,
                    "avatar": couchbase_logo,
                }
            )

            pure_llm_response = chain_without_rag.invoke(question)

            with st.chat_message("ai", avatar="🤖️"):
                st.write_stream(stream_string(pure_llm_response))

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": pure_llm_response,
                    "avatar": "🤖️",
                }
            )
