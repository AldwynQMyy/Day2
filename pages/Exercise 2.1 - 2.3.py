import os
from pathlib import Path

import numpy as np
import streamlit as st
from docx import Document
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader


# ============================================================
# 0. BASIC CONFIGURATION
# ============================================================

load_dotenv()

st.set_page_config(
    page_title="Manual RAG Lab",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Manual RAG Lab")

st.write(
    """
    This application combines three exercises:

    **Exercise 2.1:** Chunk a document  
    **Exercise 2.2:** Compare text using embeddings  
    **Exercise 2.3:** Implement RAG manually
    """
)


# ============================================================
# OPENAI CONFIGURATION
# ============================================================

EMBEDDING_MODEL = "text-embedding-3-large"

# You can change this model if necessary
ANSWER_MODEL = "gpt-4o"


def get_openai_client():

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return None

    return OpenAI(api_key=api_key)


client = get_openai_client()


# ============================================================
# SESSION STATE
# ============================================================

# Chunks created in Exercise 2.1
if "chunks" not in st.session_state:
    st.session_state.chunks = []

# Embeddings of chunks used in Exercise 2.3
if "chunk_embeddings" not in st.session_state:
    st.session_state.chunk_embeddings = []

# Name of uploaded document
if "document_name" not in st.session_state:
    st.session_state.document_name = None

# Last RAG result
if "rag_result" not in st.session_state:
    st.session_state.rag_result = None


# ============================================================
# DOCUMENT FUNCTIONS
# ============================================================

def read_document(uploaded_file):
    """
    Extract text from TXT, PDF, or DOCX.
    """

    file_extension = Path(
        uploaded_file.name
    ).suffix.lower()

    # --------------------------------------------------------
    # TXT
    # --------------------------------------------------------

    if file_extension == ".txt":

        return uploaded_file.getvalue().decode(
            "utf-8"
        )

    # --------------------------------------------------------
    # PDF
    # --------------------------------------------------------

    elif file_extension == ".pdf":

        uploaded_file.seek(0)

        reader = PdfReader(uploaded_file)

        pages = []

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                pages.append(page_text)

        return "\n".join(pages)

    # --------------------------------------------------------
    # DOCX
    # --------------------------------------------------------

    elif file_extension == ".docx":

        uploaded_file.seek(0)

        document = Document(uploaded_file)

        paragraphs = []

        for paragraph in document.paragraphs:

            text = paragraph.text.strip()

            if text:
                paragraphs.append(text)

        return "\n".join(paragraphs)

    else:

        return None


# ============================================================
# CHUNKING FUNCTION
# ============================================================

def chunk_document(text, chunk_size):
    """
    Divide text into chunks based on number of words.
    """

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

        chunk = " ".join(chunk_words)

        chunks.append(chunk)

    return chunks


# ============================================================
# SAVE CHUNKS
# ============================================================

def save_chunks(chunks):
    """
    Save every chunk as a TXT file.
    """

    chunk_directory = Path("chunks")

    chunk_directory.mkdir(
        exist_ok=True
    )

    # Remove old chunk files
    for old_file in chunk_directory.glob(
        "chunk_*.txt"
    ):
        old_file.unlink()

    # Save new chunks
    for index, chunk in enumerate(
        chunks,
        start=1
    ):

        file_path = (
            chunk_directory
            / f"chunk_{index:03}.txt"
        )

        with open(
            file_path,
            "w",
            encoding="utf-8"
        ) as file:

            file.write(chunk)

    return chunk_directory


# ============================================================
# EMBEDDING FUNCTIONS
# ============================================================

def get_embedding(text):
    """
    Create one embedding using the OpenAI API.
    """

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )

    return response.data[0].embedding


def get_embeddings(texts):
    """
    Create embeddings for multiple texts in one API call.
    """

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )

    # Make sure embeddings remain in their original order
    ordered_data = sorted(
        response.data,
        key=lambda item: item.index
    )

    return [
        item.embedding
        for item in ordered_data
    ]


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(vector_a, vector_b):
    """
    Calculate cosine similarity manually.
    """

    a = np.array(
        vector_a,
        dtype=float
    )

    b = np.array(
        vector_b,
        dtype=float
    )

    numerator = np.dot(a, b)

    denominator = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denominator == 0:
        return 0.0

    return float(
        numerator / denominator
    )


# ============================================================
# RAG RETRIEVAL FUNCTION
# ============================================================

def retrieve_most_relevant_chunk(
    question,
    chunks,
    chunk_embeddings
):
    """
    Compare the question embedding with every chunk embedding.
    """

    question_embedding = get_embedding(
        question
    )

    results = []

    for index, chunk_embedding in enumerate(
        chunk_embeddings
    ):

        score = cosine_similarity(
            question_embedding,
            chunk_embedding
        )

        results.append(
            {
                "index": index,
                "chunk_name": f"chunk_{index + 1:03}.txt",
                "similarity": score
            }
        )

    # Sort from highest similarity to lowest
    results.sort(
        key=lambda item: item["similarity"],
        reverse=True
    )

    best_result = results[0]

    best_index = best_result["index"]

    best_chunk = chunks[best_index]

    return (
        best_chunk,
        best_result,
        results
    )


# ============================================================
# LLM ANSWER FUNCTION
# ============================================================

def answer_question_with_context(
    question,
    context,
    source_name
):
    """
    Ask the LLM to answer using only the retrieved chunk.
    """

    prompt = f"""
You are a legal education assistant.

Answer the student's question ONLY using the document extract
provided below.

Rules:

1. Do not use outside knowledge.
2. Do not invent legal rules, cases, statutes, treaty provisions,
   or facts.
3. If the document extract does not contain enough information
   to answer the question, clearly say:
   "The retrieved document extract does not contain enough
   information to answer this question."
4. Explain the answer clearly for a law student.
5. At the end of the answer, cite the source exactly as:
   [{source_name}]

STUDENT QUESTION:

{question}

DOCUMENT EXTRACT:

{context}
"""

    response = client.responses.create(
        model=ANSWER_MODEL,
        input=prompt,
        store=False
    )

    return response.output_text


# ============================================================
# CREATE THREE TABS
# ============================================================

tab1, tab2, tab3 = st.tabs(
    [
        "📄 2.1 Chunk Document",
        "🔢 2.2 Compare Chunks",
        "🤖 2.3 Ask the Document"
    ]
)


# ============================================================
# EXERCISE 2.1
# ============================================================

with tab1:

    st.header(
        "Exercise 2.1 - Chunking a Document"
    )

    st.write(
        """
        Upload a document and divide it into
        smaller chunks.
        """
    )

    uploaded_file = st.file_uploader(
        "Upload a document",
        type=[
            "txt",
            "pdf",
            "docx"
        ],
        key="document_uploader"
    )

    if uploaded_file is not None:

        document_text = read_document(
            uploaded_file
        )

        if (
            document_text is None
            or document_text.strip() == ""
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

            chunk_size = st.number_input(
                "Words per chunk",
                min_value=50,
                max_value=2000,
                value=300,
                step=50
            )

            if st.button(
                "Chunk Document",
                type="primary",
                key="chunk_button"
            ):

                chunks = chunk_document(
                    document_text,
                    int(chunk_size)
                )

                directory = save_chunks(
                    chunks
                )

                # Save in Streamlit session
                st.session_state.chunks = chunks

                st.session_state.document_name = (
                    uploaded_file.name
                )

                # Reset previous embeddings
                st.session_state.chunk_embeddings = []

                # Reset previous RAG result
                st.session_state.rag_result = None

                st.success(
                    f"Created {len(chunks)} chunks."
                )

                st.write(
                    f"Chunks saved in: "
                    f"`{directory}`"
                )

                # --------------------------------------------
                # Read first saved chunk
                # --------------------------------------------

                first_chunk_path = (
                    directory
                    / "chunk_001.txt"
                )

                with open(
                    first_chunk_path,
                    "r",
                    encoding="utf-8"
                ) as file:

                    first_chunk = file.read()

                # --------------------------------------------
                # Display first chunk
                # --------------------------------------------

                st.subheader(
                    "First Saved Chunk"
                )

                st.text_area(
                    "chunk_001.txt",
                    value=first_chunk,
                    height=300,
                    disabled=True
                )

    # --------------------------------------------------------
    # Show existing session information
    # --------------------------------------------------------

    if st.session_state.chunks:

        st.divider()

        st.info(
            f"Current document: "
            f"{st.session_state.document_name} | "
            f"{len(st.session_state.chunks)} chunks "
            f"available for Exercise 2.3."
        )


# ============================================================
# EXERCISE 2.2
# ============================================================

with tab2:

    st.header(
        "Exercise 2.2 - Comparing Chunks"
    )

    st.write(
        """
        Enter two different texts.

        The application will:

        **Text → Embedding → Cosine Similarity**
        """
    )

    text_a = st.text_area(
        "Text A",
        height=180,
        key="text_a"
    )

    text_b = st.text_area(
        "Text B",
        height=180,
        key="text_b"
    )

    if st.button(
        "Compare Texts",
        type="primary",
        key="compare_button"
    ):

        if client is None:

            st.error(
                "OPENAI_API_KEY was not found."
            )

        elif not text_a.strip():

            st.warning(
                "Please enter Text A."
            )

        elif not text_b.strip():

            st.warning(
                "Please enter Text B."
            )

        else:

            try:

                with st.spinner(
                    "Creating embeddings..."
                ):

                    embedding_a = get_embedding(
                        text_a
                    )

                    embedding_b = get_embedding(
                        text_b
                    )

                    similarity = cosine_similarity(
                        embedding_a,
                        embedding_b
                    )

                st.subheader(
                    "Cosine Similarity"
                )

                st.metric(
                    "Similarity Score",
                    f"{similarity:.4f}"
                )

                st.write(
                    """
                    A higher score indicates that the
                    two texts are more semantically similar.
                    """
                )

                # --------------------------------------------
                # Show part of each embedding
                # --------------------------------------------

                with st.expander(
                    "View part of the embeddings"
                ):

                    st.write(
                        "First 10 values of Embedding A:"
                    )

                    st.code(
                        str(
                            embedding_a[:10]
                        )
                    )

                    st.write(
                        "First 10 values of Embedding B:"
                    )

                    st.code(
                        str(
                            embedding_b[:10]
                        )
                    )

                    st.write(
                        f"Embedding dimensions: "
                        f"{len(embedding_a)}"
                    )

            except Exception as error:

                st.error(
                    f"OpenAI API error: {error}"
                )


# ============================================================
# EXERCISE 2.3
# ============================================================

with tab3:

    st.header(
        "Exercise 2.3 - Implementing RAG Manually"
    )

    st.write(
        """
        This exercise combines the previous two exercises.

        The application will:

        **Question → Embedding → Similarity Search →  
        Relevant Chunk → LLM → Answer**
        """
    )

    # --------------------------------------------------------
    # Check document
    # --------------------------------------------------------

    if not st.session_state.chunks:

        st.warning(
            """
            No chunks are currently available.

            Please complete Exercise 2.1 first.
            """
        )

    else:

        st.success(
            f"Document ready: "
            f"{st.session_state.document_name}"
        )

        st.write(
            f"Number of chunks: "
            f"**{len(st.session_state.chunks)}**"
        )

        # ----------------------------------------------------
        # STEP 1: Create chunk embeddings
        # ----------------------------------------------------

        st.subheader(
            "Step 1 - Prepare the document for retrieval"
        )

        if not st.session_state.chunk_embeddings:

            st.write(
                """
                The document chunks exist, but they do not yet
                have embeddings.
                """
            )

            if st.button(
                "Create Embeddings for All Chunks",
                type="primary",
                key="index_button"
            ):

                if client is None:

                    st.error(
                        "OPENAI_API_KEY was not found."
                    )

                else:

                    try:

                        with st.spinner(
                            "Creating embeddings for document chunks..."
                        ):

                            chunk_embeddings = (
                                get_embeddings(
                                    st.session_state.chunks
                                )
                            )

                            st.session_state.chunk_embeddings = (
                                chunk_embeddings
                            )

                        st.success(
                            f"Created embeddings for "
                            f"{len(chunk_embeddings)} chunks."
                        )

                    except Exception as error:

                        st.error(
                            f"OpenAI API error: {error}"
                        )

        else:

            st.success(
                "Chunk embeddings are ready."
            )

        # ----------------------------------------------------
        # STEP 2: Ask question
        # ----------------------------------------------------

        st.subheader(
            "Step 2 - Ask a Question"
        )

        question = st.text_input(
            "Ask a question about the document:",
            placeholder=(
                "For example: What is a permanent establishment?"
            ),
            key="rag_question"
        )

        if st.button(
            "Ask Document",
            type="primary",
            key="ask_button"
        ):

            if client is None:

                st.error(
                    "OPENAI_API_KEY was not found."
                )

            elif not st.session_state.chunk_embeddings:

                st.warning(
                    "Please create the chunk embeddings first."
                )

            elif not question.strip():

                st.warning(
                    "Please enter a question."
                )

            else:

                try:

                    # ----------------------------------------
                    # RETRIEVAL
                    # ----------------------------------------

                    with st.spinner(
                        "Finding the most relevant chunk..."
                    ):

                        (
                            best_chunk,
                            best_result,
                            all_results
                        ) = retrieve_most_relevant_chunk(
                            question,
                            st.session_state.chunks,
                            st.session_state.chunk_embeddings
                        )

                    source_name = (
                        best_result["chunk_name"]
                    )

                    # ----------------------------------------
                    # GENERATION
                    # ----------------------------------------

                    with st.spinner(
                        "Generating answer..."
                    ):

                        answer = (
                            answer_question_with_context(
                                question,
                                best_chunk,
                                source_name
                            )
                        )

                    st.session_state.rag_result = {
                        "question": question,
                        "answer": answer,
                        "source": source_name,
                        "source_text": best_chunk,
                        "similarity": (
                            best_result["similarity"]
                        ),
                        "retrieval_results": (
                            all_results
                        )
                    }

                except Exception as error:

                    st.error(
                        f"Error: {error}"
                    )

        # ----------------------------------------------------
        # DISPLAY RESULT
        # ----------------------------------------------------

        if st.session_state.rag_result:

            result = st.session_state.rag_result

            st.divider()

            st.subheader(
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

            # ------------------------------------------------
            # RETRIEVAL INFORMATION
            # ------------------------------------------------

            st.subheader(
                "Retrieved Source"
            )

            st.write(
                f"**Source:** `{result['source']}`"
            )

            st.write(
                f"**Cosine similarity:** "
                f"{result['similarity']:.4f}"
            )

            with st.expander(
                "View the exact document chunk used"
            ):

                st.write(
                    result["source_text"]
                )

            # ------------------------------------------------
            # SHOW ALL SIMILARITY SCORES
            # ------------------------------------------------

            with st.expander(
                "View retrieval results"
            ):

                st.write(
                    """
                    The application compared the question
                    against every document chunk.
                    """
                )

                for item in result[
                    "retrieval_results"
                ]:

                    st.write(
                        f"**{item['chunk_name']}** "
                        f"— {item['similarity']:.4f}"
                    )