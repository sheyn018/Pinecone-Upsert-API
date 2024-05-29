from sentence_transformers import SentenceTransformer

# Load the pre-trained model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Your sentences to encode
sentences = ["This is the first sentence.", "This is the second sentence."]

# Encode the sentences
embeddings = model.encode(sentences)

# Access the encoded vectors (shape: [num_sentences, embedding_dim])
print(embeddings)