import streamlit as st
from pathlib import Path
from pypdf import PdfReader
from docx import Document


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Exercise 2.1 - Chunking a Document",
    page_icon="📄",
    layout="centered"
)

st.title("📄 Exercise 2.1 - Chunking a Document")

st.write(
    """
    Upload a document, divide it into chunks, save the chunks,
    and display the first saved chunk.
    """
)


# --------------------------------------------------
# Function 1: Read the uploaded document
# --------------------------------------------------

def read_document(uploaded_file):

    file_extension = Path(uploaded_file.name).suffix.lower()

    # TXT file
    if file_extension == ".txt":

        return uploaded_file.read().decode("utf-8")


    # PDF file
    elif file_extension == ".pdf":

        pdf_reader = PdfReader(uploaded_file)

        text = ""

        for page in pdf_reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        return text


    # DOCX file
    elif file_extension == ".docx":

        document = Document(uploaded_file)

        paragraphs = []

        for paragraph in document.paragraphs:

            if paragraph.text.strip():
                paragraphs.append(paragraph.text)

        return "\n".join(paragraphs)


    else:

        return None


# --------------------------------------------------
# Function 2: Chunk the document
# --------------------------------------------------

def chunk_document(text, chunk_size):

    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):

        chunk_words = words[i:i + chunk_size]

        chunk = " ".join(chunk_words)

        chunks.append(chunk)

    return chunks


# --------------------------------------------------
# Function 3: Save chunks
# --------------------------------------------------

def save_chunks(chunks):

    chunk_directory = Path("chunks")

    # Create the directory if it does not already exist
    chunk_directory.mkdir(exist_ok=True)

    # Remove old chunk files
    for old_file in chunk_directory.glob("chunk_*.txt"):
        old_file.unlink()

    # Save each new chunk
    for number, chunk in enumerate(chunks, start=1):

        file_path = chunk_directory / f"chunk_{number:03}.txt"

        with open(file_path, "w", encoding="utf-8") as file:
            file.write(chunk)

    return chunk_directory


# --------------------------------------------------
# 1. Allow the user to upload a document
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a document",
    type=["txt", "pdf", "docx"]
)


# --------------------------------------------------
# Only continue if a document has been uploaded
# --------------------------------------------------

if uploaded_file is not None:

    st.success(f"Uploaded: {uploaded_file.name}")

    # Read the document
    document_text = read_document(uploaded_file)

    if document_text is None:

        st.error("The document format is not supported.")

    elif document_text.strip() == "":

        st.error(
            "No readable text was found in the document."
        )

    else:

        st.write(
            f"Document contains approximately "
            f"{len(document_text.split())} words."
        )


        # --------------------------------------------------
        # 2. Allow the user to select the chunk size
        # --------------------------------------------------

        chunk_size = st.number_input(
            "Number of words in each chunk:",
            min_value=50,
            max_value=2000,
            value=300,
            step=50
        )


        # --------------------------------------------------
        # Chunk button
        # --------------------------------------------------

        if st.button(
            "Chunk Document",
            type="primary"
        ):

            # Create chunks
            chunks = chunk_document(
                document_text,
                int(chunk_size)
            )


            # --------------------------------------------------
            # 3. Save every chunk
            # --------------------------------------------------

            chunk_directory = save_chunks(chunks)

            st.success(
                f"The document was divided into "
                f"{len(chunks)} chunks."
            )

            st.write(
                f"Chunks have been saved in: "
                f"`{chunk_directory}`"
            )


            # --------------------------------------------------
            # 4. Read the first saved chunk
            # --------------------------------------------------

            first_chunk_path = (
                chunk_directory / "chunk_001.txt"
            )

            with open(
                first_chunk_path,
                "r",
                encoding="utf-8"
            ) as file:

                first_chunk = file.read()


            # --------------------------------------------------
            # 5. Display the first chunk
            # --------------------------------------------------

            st.subheader("First Chunk")

            st.text_area(
                "Contents of chunk_001.txt",
                first_chunk,
                height=300
            )