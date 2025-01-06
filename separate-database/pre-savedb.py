import os
import pickle
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA

# Utility to log execution time
def log_time(start_time, step_name):
    end_time = time.time()
    print(f"{step_name} took {end_time - start_time:.4f} seconds")
    return end_time

# Load documents from PDFs
def load_documents(directory_path):
    start_time = time.time()
    pdf_files = [
        os.path.join(directory_path, file)
        for file in os.listdir(directory_path)
        if file.endswith(".pdf")
    ]
    documents = []
    for pdf_file in pdf_files:
        loader = PyPDFLoader(pdf_file)
        documents.extend(loader.load())
    log_time(start_time, "Loading Documents")
    return documents

# Function to check if preprocessed data exists for pre-chunk and late-chunk
def load_preprocessed_data(directory_path, chunking_method):
    start_time = time.time()
    emb_file = os.path.join(directory_path, f'{chunking_method}_embeddings.pkl')
    chunk_file = os.path.join(directory_path, f'{chunking_method}_chunks.pkl')
    if os.path.exists(emb_file) and os.path.exists(chunk_file):
        with open(emb_file, 'rb') as f:
            embeddings_data = pickle.load(f)
        with open(chunk_file, 'rb') as f:
            chunks_data = pickle.load(f)
        log_time(start_time, f"Loading Preprocessed Data ({chunking_method})")
        return embeddings_data, chunks_data
    log_time(start_time, f"Loading Preprocessed Data ({chunking_method})")
    return None, None

# Function to save preprocessed data for pre-chunk and late-chunk
def save_preprocessed_data(directory_path, embeddings_data, chunks_data, chunking_method):
    os.makedirs(directory_path, exist_ok=True)
    start_time = time.time()
    emb_file = os.path.join(directory_path, f'{chunking_method}_embeddings.pkl')
    chunk_file = os.path.join(directory_path, f'{chunking_method}_chunks.pkl')
    
    with open(emb_file, 'wb') as f:
        pickle.dump(embeddings_data, f)
    with open(chunk_file, 'wb') as f:
        pickle.dump(chunks_data, f)
    log_time(start_time, f"Saving Preprocessed Data ({chunking_method})")

# Function to assess query complexity (hybrid logic)
def assess_query_complexity(query):
    start_time = time.time()
    complex_keywords = {"why", "how", "impact", "explain", "analyze", "compare"}
    word_count = len(query.split())
    result = word_count > 7 or any(word in query.lower() for word in complex_keywords)
    log_time(start_time, "Assessing Query Complexity")
    return result

# Ensure retrieved documents cover diverse sources
def diversify_retrieval(docs, max_per_source=5):
    start_time = time.time()
    source_count = {}
    diversified_docs = []
    for doc in docs:
        source = doc.metadata['source']
        if source_count.get(source, 0) < max_per_source:
            diversified_docs.append(doc)
            source_count[source] = source_count.get(source, 0) + 1
    log_time(start_time, "Diversifying Retrieval")
    return diversified_docs

# Function to load or create a vectorstore
def load_or_create_vectorstore(vectorstore_dir, embeddings, chunks_data):
    start_time = time.time()
    if os.path.exists(vectorstore_dir):
        print(f"Loading existing vector store from {vectorstore_dir}")
        vectorstore = Chroma(persist_directory=vectorstore_dir, embedding_function=embeddings)
    else:
        print(f"Creating a new vector store at {vectorstore_dir}")
        # Create a new vector store from the chunks
        vectorstore = Chroma.from_documents(chunks_data, embeddings, persist_directory=vectorstore_dir)
        vectorstore.persist()  # Persist the new vector store
    log_time(start_time, f"Loading/Creating Vector Store ({vectorstore_dir})")
    return vectorstore

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Path to store preprocessed data
preprocessed_data_dir = './preprocessed_data'

# Check if preprocessed data exists for pre-chunking and late-chunking
pre_chunk_embeddings_data, pre_chunk_chunks_data = load_preprocessed_data(preprocessed_data_dir, 'prechunk')
late_chunk_embeddings_data, late_chunk_chunks_data = load_preprocessed_data(preprocessed_data_dir, 'latechunk')

# Load documents and process if not preprocessed for either method
if pre_chunk_embeddings_data is None or pre_chunk_chunks_data is None:
    start_time = time.time()
    documents = load_documents("docs/")
    print(f"Number of documents loaded: {len(documents)}")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = [text_splitter.split_documents([doc]) for doc in documents]
    chunks = [chunk for sublist in chunks for chunk in sublist]  # Flatten the list

    if not chunks:
        raise ValueError("Chunks are empty after splitting. Check the input documents or splitter configuration.")
    
    # Add metadata to each chunk
    for chunk in chunks:
        if 'source' in chunk.metadata and 'page' in chunk.metadata:
            print(f"Chunk Source: {chunk.metadata['source']}, Page: {chunk.metadata['page']}")

    print(f"Number of chunks created: {len(chunks)}")

    # Embed the chunks for pre-chunking
    pre_chunk_embeddings_data = embeddings.embed_documents([chunk.page_content for chunk in chunks])

    # Save the preprocessed data for pre-chunking
    save_preprocessed_data(preprocessed_data_dir, pre_chunk_embeddings_data, chunks, 'prechunk')

    # Ensure `pre_chunk_chunks_data` is set
    pre_chunk_chunks_data = chunks
    log_time(start_time, "Preprocessing Documents for Pre-Chunking")


# Initialize vector store for pre-chunking
if not pre_chunk_chunks_data:
    raise ValueError("Pre-chunk chunks data is empty. Cannot create vector store.")

pre_chunk_vectorstore = load_or_create_vectorstore("./chroma_store_preq", embeddings, pre_chunk_chunks_data)

# Load documents and process for late-chunking if not preprocessed
if late_chunk_embeddings_data is None or late_chunk_chunks_data is None:
    start_time = time.time()
    documents = load_documents("docs/")
    late_chunk_embeddings_data = None  # No embeddings yet for late-chunking
    late_chunk_chunks_data = documents  # Keep the raw documents for late chunking
    
    # Save the preprocessed data for late-chunking
    save_preprocessed_data(preprocessed_data_dir, late_chunk_embeddings_data, late_chunk_chunks_data, 'latechunk')
    log_time(start_time, "Preprocessing Documents for Late-Chunking")

# Initialize vector store for late-chunking
late_chunk_vectorstore = load_or_create_vectorstore("./chroma_store_late", embeddings, late_chunk_chunks_data)

# User query input
query = input("Enter your query: ")

# Assess query complexity and choose retrieval method
start_time = time.time()
if assess_query_complexity(query):
    print("Using late chunking for query:", query)
    vectorstore = late_chunk_vectorstore
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
else:
    print("Using pre-chunking for query:", query)
    vectorstore = pre_chunk_vectorstore

log_time(start_time, "Query Complexity Assessment")

# Retrieve relevant documents
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 10})
start_time = time.time()
retrieved_docs = retriever.invoke(query)
retrieved_docs = diversify_retrieval(retrieved_docs)
log_time(start_time, "Retrieving and Diversifying Documents")

# Log retrieved document metadata
print("Retrieved Documents Metadata:")
for doc in retrieved_docs:
    print(f"- Source: {doc.metadata['source']}, Page: {doc.metadata['page']}")

# Decide on chunking approach based on query complexity
if assess_query_complexity(query):
    print("Using late chunking for query:", query)
    # Late chunking: dynamically chunk the retrieved documents
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
start_time = time.time()
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
log_time(start_time, "Generating Answer with Ollama")

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