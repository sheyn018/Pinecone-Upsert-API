import streamlit as st
from dotenv import load_dotenv, find_dotenv
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Pinecone

load_dotenv(find_dotenv(), override=True)
pinecone_api_key = os.getenv("PINECONE_API_KEY")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def dynamic_chunk_size(document_length):
    """Determine dynamic chunk size based on document length."""
    if document_length < 1000:
        return max(500, document_length)
    else:
        return 700
    
def get_text_chunks(text):
    document_length = len(text)
    chunk_size = dynamic_chunk_size(document_length)
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def upsert_vectors(text_chunks):
    """Upsert vectors to Pinecone."""
    # Load the sentence transformer model
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    # Encode the text chunks to get embeddings
    embeddings = model.encode(text_chunks)

    # Generate unique IDs for each text chunk
    ids = [f"id_{i}" for i in range(len(text_chunks))]
    
    # Create a list of dictionaries in the format Pinecone expects
    vectors = []
    for id, embedding, text in zip(ids, embeddings, text_chunks):
        data = {
            "id": id,
            "values": embedding.tolist(),  # Convert numpy array to list if necessary
            "metadata": {"content": text}
        }
        vectors.append(data)

    from pinecone import Pinecone
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index("huggingface")

    index.upsert(
        vectors=vectors,
        namespace="ns1"
    )
    

def main():
    load_dotenv()
    st.set_page_config(
        page_title="Chat with multiple PDFs",
        page_icon=":books:"
    )

    st.header("Chat with multiple PDFs :books:")
    

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)
                print(raw_text)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                print(len(text_chunks))

                # create vector store
                vectorstore = upsert_vectors(text_chunks)

                # # create conversation chain
                # st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == '__main__':
    main()
