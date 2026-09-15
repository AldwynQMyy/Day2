import os
from pathlib import Path

import chromadb
import streamlit as st
from docx import Document
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader


# ============================================================
# 0. CONFIGURATION
# ============================================================

load_dotenv()

st.set_page_config(
    page_title="Exercise 2.4 - Vector Database RAG",
    page_icon="🗄️",
    layout="wide"
)

st.title("🗄️ Exercise 2.4 - RAG with a Vector Database")

st.write(
    """
    In Exercise 2.3, we implemented retrieval manually.

    In this exercise, we will use **Chroma**, a vector database,
    to store document chunks and retrieve the most relevant
    information automatically.
    """
)


# ============================================================
# OPENAI CLIENT
# ============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

ANSWER_MODEL = os.getenv(
    "OPENAI_ANSWER_MODEL",
    "gpt-4o"
)

if OPENAI_API_KEY:

    openai_client = OpenAI(
        api_key=OPENAI_API_KEY
    )

else:

    openai_client = None


# ============================================================
# CHROMA VECTOR DATABASE
# ============================================================

CHROMA_PATH = "./chroma_db"

COLLECTION_NAME = "legal_document"

chroma_client = chromadb.PersistentClient(
    path=CHROMA_PATH
)


# ============================================================
# SESSION STATE
# ============================================================

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "rag_result" not in st.session_state:
    st.session_state.rag_result = None


# ============================================================
# READ DOCUMENT
# ============================================================

def read_document(uploaded_file):

    extension = Path(
        uploaded_file.name
    ).suffix.lower()

    # --------------------------------------------------------
    # TXT
    # --------------------------------------------------------

    if extension == ".txt":

        return uploaded_file.getvalue().decode(
            "utf-8"
        )

    # --------------------------------------------------------
    # PDF
    # --------------------------------------------------------

    elif extension == ".pdf":

        uploaded_file.seek(0)

        reader = PdfReader(
            uploaded_file
        )

        pages = []

        for page in reader.pages:

            text = page.extract_text()

            if text:
                pages.append(text)

        return "\n".join(pages)

    # --------------------------------------------------------
    # DOCX
    # --------------------------------------------------------

    elif extension == ".docx":

        uploaded_file.seek(0)

        document = Document(
            uploaded_file
        )

        paragraphs = []

        for paragraph in document.paragraphs:

            text = paragraph.text.strip()

            if text:
                paragraphs.append(text)

        return "\n".join(paragraphs)

    return None


# ============================================================
# CHUNK DOCUMENT
# ============================================================

def chunk_document(text, chunk_size):

    words = text.split()

    chunks = []

    for i in range(
        0,
        len(words),
        chunk_size
    ):

        chunk_words = words[
            i:i + chunk_size
        ]

        chunk = " ".join(
            chunk_words
        )

        chunks.append(chunk)

    return chunks


# ============================================================
# CREATE VECTOR DATABASE
# ============================================================

def create_vector_database(
    chunks,
    document_name
):

    # --------------------------------------------------------
    # Delete old collection if it exists
    # --------------------------------------------------------

    try:

        chroma_client.delete_collection(
            name=COLLECTION_NAME
        )

    except Exception:

        pass


    # --------------------------------------------------------
    # Create a new collection
    #
    # We use cosine distance because this is a text
    # similarity exercise.
    #
    # No embedding function is manually provided.
    # Chroma will use its default embedding function.
    # --------------------------------------------------------

    collection = (
        chroma_client.create_collection(
            name=COLLECTION_NAME,
            configuration={
                "hnsw": {
                    "space": "cosine"
                }
            }
        )
    )


    # --------------------------------------------------------
    # Create IDs
    # --------------------------------------------------------

    ids = []

    metadatas = []

    for index in range(
        len(chunks)
    ):

        chunk_number = index + 1

        chunk_name = (
            f"chunk_{chunk_number:03}.txt"
        )

        ids.append(
            f"chunk_{chunk_number:03}"
        )

        metadatas.append(
            {
                "source": document_name,
                "chunk_number": chunk_number,
                "chunk_name": chunk_name
            }
        )


    # --------------------------------------------------------
    # Add the chunks to Chroma
    #
    # IMPORTANT:
    #
    # We do NOT manually create embeddings here.
    #
    # Chroma:
    # 1. creates embeddings
    # 2. stores embeddings
    # 3. builds the vector index
    # --------------------------------------------------------

    collection.add(
        ids=ids,
        documents=chunks,
        metadatas=metadatas
    )

    return collection


# ============================================================
# GET EXISTING COLLECTION
# ============================================================

def get_collection():

    try:

        return chroma_client.get_collection(
            name=COLLECTION_NAME
        )

    except Exception:

        return None


# ============================================================
# RETRIEVE RELEVANT CHUNKS
# ============================================================

def retrieve_chunks(
    question,
    number_of_results=3
):

    collection = get_collection()

    if collection is None:

        return None

    total_chunks = collection.count()

    if total_chunks == 0:

        return None

    number_of_results = min(
        number_of_results,
        total_chunks
    )


    # --------------------------------------------------------
    # THIS replaces most of Exercise 2.3
    #
    # Chroma will:
    #
    # question
    #    ↓
    # embedding
    #    ↓
    # vector search
    #    ↓
    # nearest chunks
    # --------------------------------------------------------

    results = collection.query(
        query_texts=[
            question
        ],
        n_results=number_of_results,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    return results


# ============================================================
# BUILD CONTEXT FOR LLM
# ============================================================

def build_context(results):

    documents = results[
        "documents"
    ][0]

    metadatas = results[
        "metadatas"
    ][0]

    context_parts = []

    for document, metadata in zip(
        documents,
        metadatas
    ):

        chunk_name = metadata[
            "chunk_name"
        ]

        context_parts.append(
            f"""
SOURCE: {chunk_name}

{document}
"""
        )

    return "\n\n".join(
        context_parts
    )


# ============================================================
# GENERATE ANSWER
# ============================================================

def answer_question(
    question,
    results
):

    context = build_context(
        results
    )

    prompt = f"""
You are a legal education assistant.

Answer the student's question ONLY using the
document extracts supplied below.

Rules:

1. Do not use outside legal knowledge.

2. Do not invent cases, statutes, treaty provisions,
   legal rules, or facts.

3. If the supplied extracts do not contain enough
   information to answer the question, clearly say so.

4. Explain the answer clearly for a law student.

5. Whenever you use information from a source,
   cite its chunk name in square brackets.

Example citation:

[chunk_003.txt]

STUDENT QUESTION:

{question}


DOCUMENT EXTRACTS:

{context}
"""

    response = openai_client.responses.create(
        model=ANSWER_MODEL,
        input=prompt,
        store=False
    )

    return response.output_text


# ============================================================
# PART 1 - UPLOAD DOCUMENT
# ============================================================

st.header(
    "Step 1 - Upload a Document"
)

uploaded_file = st.file_uploader(
    "Upload TXT, PDF, or DOCX",
    type=[
        "txt",
        "pdf",
        "docx"
    ]
)


if uploaded_file is not None:

    document_text = read_document(
        uploaded_file
    )

    if (
        document_text is None
        or not document_text.strip()
    ):

        st.error(
            "No readable text was found."
        )

    else:

        word_count = len(
            document_text.split()
        )

        st.success(
            f"Uploaded: {uploaded_file.name}"
        )

        st.write(
            f"Approximate word count: "
            f"**{word_count}**"
        )


        # ====================================================
        # PART 2 - CHUNK SIZE
        # ====================================================

        st.header(
            "Step 2 - Chunk and Index the Document"
        )

        chunk_size = st.number_input(
            "Words per chunk",
            min_value=50,
            max_value=2000,
            value=300,
            step=50
        )


        if st.button(
            "Create Vector Database",
            type="primary"
        ):

            try:

                # --------------------------------------------
                # Chunk document
                # --------------------------------------------

                chunks = chunk_document(
                    document_text,
                    int(chunk_size)
                )

                st.session_state.chunks = (
                    chunks
                )

                st.session_state.document_name = (
                    uploaded_file.name
                )

                st.session_state.rag_result = (
                    None
                )


                # --------------------------------------------
                # Create vector database
                # --------------------------------------------

                with st.spinner(
                    "Creating embeddings and "
                    "building vector database..."
                ):

                    collection = (
                        create_vector_database(
                            chunks,
                            uploaded_file.name
                        )
                    )


                st.success(
                    "Vector database created."
                )

                st.write(
                    f"Chunks stored: "
                    f"**{collection.count()}**"
                )

                st.write(
                    f"Database directory: "
                    f"`{CHROMA_PATH}`"
                )

            except Exception as error:

                st.error(
                    f"Vector database error: "
                    f"{error}"
                )


# ============================================================
# SHOW VECTOR DATABASE STATUS
# ============================================================

collection = get_collection()

if collection is not None:

    count = collection.count()

    if count > 0:

        st.divider()

        st.info(
            f"Vector database ready: "
            f"{count} document chunks indexed."
        )


        # ====================================================
        # PART 3 - ASK QUESTION
        # ====================================================

        st.header(
            "Step 3 - Ask the Document"
        )

        number_of_results = st.slider(
            "Number of chunks to retrieve",
            min_value=1,
            max_value=min(
                5,
                count
            ),
            value=min(
                3,
                count
            )
        )


        question = st.text_input(
            "Ask a question about the document:",
            placeholder=(
                "For example: "
                "What is a permanent establishment?"
            )
        )


        if st.button(
            "Ask Question",
            type="primary"
        ):

            if not question.strip():

                st.warning(
                    "Please enter a question."
                )

            elif openai_client is None:

                st.error(
                    "OPENAI_API_KEY was not found."
                )

            else:

                try:

                    # ========================================
                    # RETRIEVAL
                    # ========================================

                    with st.spinner(
                        "Searching the vector database..."
                    ):

                        results = retrieve_chunks(
                            question,
                            number_of_results
                        )


                    # ========================================
                    # GENERATION
                    # ========================================

                    with st.spinner(
                        "Generating answer..."
                    ):

                        answer = answer_question(
                            question,
                            results
                        )


                    st.session_state.rag_result = {
                        "question": question,
                        "answer": answer,
                        "results": results
                    }

                except Exception as error:

                    st.error(
                        f"Error: {error}"
                    )


# ============================================================
# DISPLAY RESULT
# ============================================================

if st.session_state.rag_result:

    result = st.session_state.rag_result

    st.divider()

    st.header(
        "Answer"
    )

    with st.chat_message(
        "user"
    ):

        st.write(
            result["question"]
        )


    with st.chat_message(
        "assistant"
    ):

        st.write(
            result["answer"]
        )


    # ========================================================
    # SHOW RETRIEVAL RESULTS
    # ========================================================

    st.header(
        "Retrieved Sources"
    )

    results = result[
        "results"
    ]

    documents = results[
        "documents"
    ][0]

    metadatas = results[
        "metadatas"
    ][0]

    distances = results[
        "distances"
    ][0]


    for rank, (
        document,
        metadata,
        distance
    ) in enumerate(
        zip(
            documents,
            metadatas,
            distances
        ),
        start=1
    ):

        chunk_name = metadata[
            "chunk_name"
        ]

        st.subheader(
            f"{rank}. {chunk_name}"
        )

        st.write(
            f"Vector distance: "
            f"`{distance:.4f}`"
        )

        with st.expander(
            f"View {chunk_name}"
        ):

            st.write(
                document
            )