import streamlit as st
import os
import requests
from bs4 import BeautifulSoup
from langchain.llms import HuggingFaceHub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Global variable to store vector DB
VECTORSTORE_PATH = "chroma_db"
persisted_vectorstore = None

# Scrape text from URL


def scrape_url(url):
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return f"Error: {str(e)}"

    soup = BeautifulSoup(response.content, "html.parser")
    return soup.get_text()

# Split and store the scraped text into ChromaDB


def setup_vectorstore(text):
    global persisted_vectorstore

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    docs = text_splitter.create_documents([text])

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=VECTORSTORE_PATH
    )
    vectordb.persist()
    persisted_vectorstore = vectordb

# Retrieve answer using stored vector DB


def answer_query(query):
    global persisted_vectorstore

    if not persisted_vectorstore:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        persisted_vectorstore = Chroma(
            persist_directory=VECTORSTORE_PATH,
            embedding_function=embeddings
        )

    retriever = persisted_vectorstore.as_retriever()

    # Still uses OpenAI for answering

    llm = ChatOpenAI(model_name="gpt-3.5-turbo")

    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa.run(query)


# --- Streamlit UI ---
st.title("🧠 RAG App with Web Scraping + Free Embeddings")
url_input = st.text_input("🔗 Enter URL to scrape and store")

if st.button("Scrape & Store"):
    if url_input:
        with st.spinner("Scraping and embedding..."):
            scraped_text = scrape_url(url_input)
            if scraped_text.startswith("Error:"):
                st.error(scraped_text)
            else:
                setup_vectorstore(scraped_text)
                st.success("Webpage scraped and stored in Chroma DB!")
    else:
        st.warning("Please enter a valid URL.")

question_input = st.text_input("❓ Ask a question based on the stored page")

if st.button("Get Answer"):
    if question_input:
        with st.spinner("Generating answer..."):
            try:
                response = answer_query(question_input)
                st.success(response)
            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        st.warning("Please ask a question.")
