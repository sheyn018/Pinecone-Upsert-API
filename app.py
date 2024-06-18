import streamlit as st
from dotenv import load_dotenv, find_dotenv
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
import docx2txt
import streamlit.components.v1 as components

load_dotenv(find_dotenv(), override=True)
pinecone_api_key = os.getenv("PINECONE_API_KEY")
project_id = os.getenv("PROJECT_ID")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_word_text(word_docs):
    text = ""
    for doc in word_docs:
        text += docx2txt.process(doc)
    return text

def get_txt_text(txt_docs):
    text = ""
    for txt in txt_docs:
        text += txt.read().decode("utf-8") + "\n"
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
        page_title="Knowledge Base Upload",
        page_icon=":books:",
        layout="wide"
    )

    st.sidebar.header("Upload Documents into Vector Database")
    
    st.sidebar.subheader("Upload your documents")
    files = st.sidebar.file_uploader(
        "Upload PDF, Word, or TXT files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )
    
    if st.sidebar.button("Process"):
        if not files:
            st.sidebar.warning("Please upload at least one document before processing.")
            return
        
        with st.spinner("Processing documents..."):
            raw_text = ""
            
            pdf_docs = [file for file in files if file.type == "application/pdf"]
            if pdf_docs:
                raw_text += get_pdf_text(pdf_docs)
            
            word_docs = [file for file in files if file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]
            if word_docs:
                raw_text += get_word_text(word_docs)
            
            txt_docs = [file for file in files if file.type == "text/plain"]
            if txt_docs:
                raw_text += get_txt_text(txt_docs)

            if not raw_text:
                st.sidebar.error("No text extracted from the uploaded documents. Please check the files and try again.")
                return

            # get the text chunks
            text_chunks = get_text_chunks(raw_text)
            st.sidebar.write(f"Number of text chunks: {len(text_chunks)}")

            # create vector store
            upsert_vectors(text_chunks)

            st.sidebar.success("Documents processed and vectors upserted successfully!")
    
    st.header("Chatbot")
    chatbot_html = f"""
    <div class="chatbot-inner section-inner">
        <div id="flat-chat"></div>
        <style>
            #flat-chat {{
                width: 100%;
                height: 800px; /* Adjust the height as needed */
                border: none;
                overflow: hidden;
            }}
        </style>
        <script type="text/javascript">
            (function(d, t) {{
                var v = d.createElement(t),
                    s = d.getElementsByTagName(t)[0];
                v.onload = function() {{
                    window.voiceflow.chat.load({{
                        verify: {{
                            projectID: '{project_id}'
                        }},
                        url: 'https://general-runtime.voiceflow.com',
                        versionID: 'production',
                        render: {{
                            mode: 'embedded',
                            target: document.getElementById('flat-chat'),
                        }},
                        autostart: false,
                    }});
                }};
                v.src = "https://cdn.voiceflow.com/widget/bundle.mjs";
                s.parentNode.insertBefore(v, s);
            }})(document, 'script');
        </script>
    </div>
    """

    components.html(chatbot_html, height=850)

if __name__ == '__main__':
    main()
