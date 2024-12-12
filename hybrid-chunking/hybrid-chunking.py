import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA

# Load documents from PDFs
directory_path = "docs/"
pdf_files = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith(".pdf")]
documents = []
for pdf_file in pdf_files:
    loader = PyPDFLoader(pdf_file)
    documents.extend(loader.load())

# Initialize embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents, embeddings, persist_directory="./chroma_store")
retriever = vectorstore.as_retriever()

# Function to assess query complexity (this can be more sophisticated)
def assess_query_complexity(query):
    # Simple logic: If query length > threshold, it's complex
    return len(query.split()) > 7  # Example threshold

# User query input
query = input("Enter your query: ")

# Retrieve relevant documents
retrieved_docs = retriever.invoke(query)

# Decide on chunking approach based on query complexity
if assess_query_complexity(query):
    print("Using late chunking for query:", query)
    # Late chunking: dynamically chunk the retrieved documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = []
    for doc in retrieved_docs:
        chunks.extend(text_splitter.split_documents([doc]))
else:
    print("Using pre-chunking for query:", query)
    # Pre-chunking: documents were already split during vector store creation
    chunks = retrieved_docs

# Initialize the LLM
llm = OllamaLLM(model="llama3", base_url="http://127.0.0.1:11434")

# Combine the content of the documents into a single context
context = "\n\n".join([doc.page_content for doc in retrieved_docs])

prompt = f"""
You are an expert AI assistant. Use the provided context and your own knowledge to answer the question in a clear, concise, and professional manner. 

### Instructions:
1. First, prioritize using the context to provide the answer.
2. If additional information is needed, supplement your response with your own knowledge.
3. Always provide the sources for any information retrieved from the context.
4. If the context does not answer the question and you rely solely on your own knowledge, clearly state that no external sources were used.

### Context:
{context}

### Question:
{query}

### Answer:
"""

# Generate the response using Ollama
response = llm.invoke(prompt)

# Print response and source information
print("Answer:", response)
