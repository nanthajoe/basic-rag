import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA

# Load documents from PDFs
def load_documents(directory_path):
    pdf_files = [
        os.path.join(directory_path, file)
        for file in os.listdir(directory_path)
        if file.endswith(".pdf")
    ]
    documents = []
    for pdf_file in pdf_files:
        loader = PyPDFLoader(pdf_file)
        documents.extend(loader.load())
    return documents

directory_path = "docs/"
documents = load_documents(directory_path)

# Initialize embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents, embeddings, persist_directory="./chroma_store")
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 10})  # Enable MMR and fetch top 10

# Function to assess query complexity (hybrid logic)
def assess_query_complexity(query):
    complex_keywords = {"why", "how", "impact", "explain", "analyze", "compare"}
    word_count = len(query.split())
    # Complex if word count > 7 or contains complex keywords
    return word_count > 7 or any(word in query.lower() for word in complex_keywords)

# Ensure retrieved documents cover diverse sources
def diversify_retrieval(docs, max_per_source=5):
    source_count = {}
    diversified_docs = []
    for doc in docs:
        source = doc.metadata['source']
        if source_count.get(source, 0) < max_per_source:
            diversified_docs.append(doc)
            source_count[source] = source_count.get(source, 0) + 1
    return diversified_docs

# User query input
query = input("Enter your query: ")

# Retrieve relevant documents
retrieved_docs = retriever.invoke(query)
retrieved_docs = diversify_retrieval(retrieved_docs)

# Log retrieved document metadata
print("Retrieved Documents Metadata:")
for doc in retrieved_docs:
    print(f"- Source: {doc.metadata['source']}, Page: {doc.metadata['page']}")

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

# Combine the content of the chunks/documents into a single context
context = "\n\n".join([chunk.page_content for chunk in chunks])

# Initialize the LLM
llm = OllamaLLM(model="llama3", base_url="http://127.0.0.1:11434")

# Advanced prompt with hybrid context
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
sources = set()
for chunk in chunks:
    sources.add((chunk.metadata['source'], chunk.metadata['page']))

# Sort sources by document name and page number
sorted_sources = sorted(sources, key=lambda x: (x[0], x[1]))
for source, page in sorted_sources:
    print(f"- {source} (Page {page})")
