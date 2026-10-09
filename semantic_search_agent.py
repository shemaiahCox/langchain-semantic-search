import os
import getpass
from typing import List
import pypdf

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.runnables import chain
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. API Key & Environment Setup
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

# 2. PDF Loader Helper Function
def load_pdf_pages(file_path: str) -> list[Document]:
    """Loads a PDF file page by page into LangChain Document objects."""
    reader = pypdf.PdfReader(file_path)
    return [
        Document(
            page_content=page.extract_text() or "",
            metadata={"source": file_path, "page": i},
        )
        for i, page in enumerate(reader.pages)
    ]

# 3. Load PDF and Split Content
file_path = "../example_data/nke-10k-2023.pdf"

# Fallback check for missing demo file path
if os.path.exists(file_path):
    docs = load_pdf_pages(file_path)
else:
    # Minimal fallback sample if PDF is not available locally
    docs = [
        Document(
            page_content="NIKE, Inc. was incorporated in 1967 under the laws of the State of Oregon.",
            metadata={"source": "sample", "page": 3},
        ),
        Document(
            page_content="In the United States, NIKE has eight significant distribution centers.",
            metadata={"source": "sample", "page": 4},
        ),
    ]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
)
all_splits = text_splitter.split_documents(docs)

# 4. Initialize Embeddings and Vector Store
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = InMemoryVectorStore(embeddings)

# Index chunked documents
ids = vector_store.add_documents(documents=all_splits)

# 5. Define Custom Chains / Retrievers

# Decorator-based custom retriever
@chain
def custom_retriever(query: str) -> List[Document]:
    return vector_store.similarity_search(query, k=1)

# Standard VectorStoreRetriever instance
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)

# 6. Execution Examples
if __name__ == "__main__":
    print("--- 1. String Search Result ---")
    results = vector_store.similarity_search("How many distribution centers does Nike have in the US?")
    if results:
        print(f"Content snippet: {results[0].page_content[:200]}...")
        print(f"Metadata: {results[0].metadata}")

    print("\n--- 2. Search with Score Result ---")
    scored_results = vector_store.similarity_search_with_score("What was Nike's revenue in 2023?")
    if scored_results:
        doc, score = scored_results[0]
        print(f"Distance Score: {score}")
        print(f"Content snippet: {doc.page_content[:200]}...")

    print("\n--- 3. Batch Query via VectorStoreRetriever ---")
    batch_results = retriever.batch(
        [
            "How many distribution centers does Nike have in the US?",
            "When was Nike incorporated?",
        ]
    )
    for idx, res_list in enumerate(batch_results, start=1):
        if res_list:
            print(f"Query {idx} Top Result: {res_list[0].page_content[:150]}...")