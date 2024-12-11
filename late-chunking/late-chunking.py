import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM 
from langchain.chains import RetrievalQA

# Optional: Load environment variables if needed
load_dotenv()

# Disable LangChain Smith (if not required)
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Suppress HuggingFace Tokenizer Parallelism Warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Step 1: Load all PDFs from a directory
directory_path = "docs/"
pdf_files = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith(".pdf")]

# Load full documents (no chunking yet)
documents = []
for pdf_file in pdf_files:
    loader = PyPDFLoader(pdf_file)
    documents.extend(loader.load())

# Step 2: Initialize embeddings and create Chroma database using full documents
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents, embeddings, persist_directory="./chroma_store")
# print(f"Chroma database created with {len(documents)} full documents!")

# Query the vectorstore using `invoke`
retriever = vectorstore.as_retriever()

# Manually input query
query = input("Enter your query: ")
retrieved_docs = retriever.invoke(query)
# print(f"Retrieved {len(retrieved_docs)} documents relevant to the query.")

# Step 3: Apply late chunking
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# Dynamically split the retrieved documents into chunks
chunks = []
for doc in retrieved_docs:
    chunks.extend(text_splitter.split_documents([doc]))

# Display the first chunk (for verification)
# if chunks:
#     print(f"Generated {len(chunks)} chunks from {len(retrieved_docs)} retrieved documents.")
#     print("First chunk content:", chunks[0].page_content)

# Step 4: Initialize the Ollama LLM
llm = OllamaLLM(model="llama3", base_url="http://127.0.0.1:11434")  # Replace with your model and base URL

# Advanced prompt with retrieval and Llama 3's knowledge
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
print("Source Documents:")
for doc in retrieved_docs:
    print(f"- Page Content: {doc.page_content[:200]}...")  # Truncated for readability
    print(f"  Metadata: {doc.metadata}")
