from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Load the document
loader = TextLoader("text.txt", encoding="utf-8")
documents = loader.load()

# 2. Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
)
chunks = text_splitter.split_documents(documents)

# 3. Create embeddings (runs locally on CPU, no API key needed)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. Store chunks in an in-memory FAISS vector store
vectorstore = FAISS.from_documents(chunks, embeddings)

# 5. Export retriever for use in app.py
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
