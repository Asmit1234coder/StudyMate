import streamlit as st
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from langchain_text_splitters import RecursiveCharacterTextSplitter

import tempfile
import os

load_dotenv()

st.set_page_config(
    page_title="StudyMate",
    page_icon="📚",
    layout="wide"
)

st.title("📚 StudyMate")
st.caption(
    "Upload document and ask questions"
)

embedding_model = MistralAIEmbeddings(
    model="mistral-embed"
)

llm = ChatMistralAI(
    model="mistral-small-2506"
)

prompt = ChatPromptTemplate.from_messages(
[
(
"system",
"""
You are an AI study assistant.

Answer ONLY from context.

If answer is missing say:

I could not find the answer in document.
"""
),
(
"human",
"""
Context:
{context}

Question:
{question}
"""
)
]
)

uploaded_file = st.sidebar.file_uploader(
    "Upload File",
    type=[
        "txt",
        "pdf"
    ]
)

if uploaded_file:

    text=""

    if uploaded_file.name.endswith(".txt"):

        text = uploaded_file.read().decode(
            "utf-8"
        )

    elif uploaded_file.name.endswith(".pdf"):

        from langchain_community.document_loaders import PyPDFLoader

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp:

            tmp.write(
                uploaded_file.getvalue()
            )

            temp_path = tmp.name

        loader = PyPDFLoader(
            temp_path
        )

        docs = loader.load()

        text = "\n".join(
            [
                d.page_content
                for d in docs
            ]
        )

        os.remove(temp_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_text(
        text
    )

    documents = [
        Document(
            page_content=i
        )
        for i in chunks
    ]

    vectorstore = Chroma.from_documents(
        documents,
        embedding_model
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k":4,
            "fetch_k":10,
            "lambda_mult":0.5
        }
    )

    st.success(
        "Document indexed successfully"
    )

    question = st.chat_input(
        "Ask anything..."
    )

    if question:

        with st.chat_message(
            "user"
        ):

            st.write(question)

        docs = retriever.invoke(
            question
        )

        if docs:

            context = "\n\n".join(
                [
                    x.page_content
                    for x in docs
                ]
            )

            final_prompt = prompt.invoke(
                {
                    "context":context,
                    "question":question
                }
            )

            response = llm.invoke(
                final_prompt
            )

            with st.chat_message(
                "assistant"
            ):

                st.write(
                    response.content
                )

            with st.expander(
                "Retrieved Chunks"
            ):

                for i,d in enumerate(
                    docs
                ):

                    st.write(
                        f"Chunk {i+1}"
                    )

                    st.write(
                        d.page_content
                    )

else:

    st.info(
        "Upload file from sidebar"
    )