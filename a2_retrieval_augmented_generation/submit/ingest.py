import argparse
from http import client
import json
import time
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymilvus import DataType, MilvusClient
from tqdm import tqdm
import os
from langchain_core.documents import Document

# -------- Load JSONL Files --------
def load_jsonl_files(dataset_path: str) -> list[str]:
    """
    Load all JSONL files from the specified dataset path and return
    their text contents as a list of strings.
    """
    file_list = []
    for filename in os.listdir(dataset_path):
        file_path = os.path.join(dataset_path, filename)
        if os.path.isfile(file_path) and filename.endswith('.jsonl'):
            file_list.append(file_path)

    docs = []
    for file in file_list:
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    doc = Document(
                        page_content=data["text"],
                        metadata={"source": data["path"]}
                    )
                    docs.append(doc.page_content)
                except (json.JSONDecodeError, KeyError):
                    continue      

    print(f"Loaded {len(docs)} documents from {len(file_list)} files.")
    return docs

# -------- Chunk Text --------
def chunk_docs(documents: list[str], chunk_size: int =1000, chunk_overlap: int = 100) -> list[str]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    docs_chunks = []
    for doc in documents:
        chunks = text_splitter.split_text(doc)
        docs_chunks.extend(chunks)

    return docs_chunks

# -------- Create Embeddings --------
from sentence_transformers import SentenceTransformer
import numpy as np

def create_embedding(model: str, texts: list[str]) -> np.ndarray:

    embed_model = SentenceTransformer(model)
    embeddings = embed_model.encode(texts, batch_size=32, show_progress_bar=True)
    
    return embeddings

# -------- Ingest embeddings to Milvus --------
def ingest_vector(documents: list[str], db_path: str, model: str, vector_size: int) -> None:
    
    chunked_docs = chunk_docs(documents)
    print(f"Chunked into {len(chunked_docs)} chunks.")

    embeddings = create_embedding(model, chunked_docs)
    print(f"Created embeddings with shape: {embeddings.shape}")

    milvus_data = []
    for i in range(embeddings.shape[0]):
        milvus_data.append({
            "id": i,
            "vector": embeddings[i, :].tolist(),
            "text": chunked_docs[i]
        })
    print(f"Prepared {len(milvus_data)} records for Milvus ingestion.")

    database = os.path.join(db_path, "milvus_literature.db")
    # MilvusClient(uri="./database/milvus_literature.db")
    client = MilvusClient(uri=database)
    
    collection_name = "literature_collection"
    
    # Drop old collection if exists
    if client.has_collection(collection_name=collection_name):
        print(f"Dropping existing collection: {collection_name}")
        client.drop_collection(collection_name=collection_name)
        
    # Create collection
    print(f"Creating collection: {collection_name}")
    client.create_collection(
        collection_name=collection_name,
        dimension=vector_size,
        metric_type="IP"
    )

    # Insert milvus_data
    print(f"Inserting {len(milvus_data)} entities into collection...")
    res = client.insert(
        collection_name=collection_name,
        data=milvus_data
    )

    #print("Insert result:")
    #print(res)
    print(f"Successfully inserted {res['insert_count']} entities.")

# -------- Ingest Text to Whoosh --------
from whoosh.fields import Schema, TEXT, ID
from whoosh import index

def ingest_text(documents: list[str], db_path: str) -> None:
    chunked_docs = chunk_docs(documents)
    print(f"Chunked into {len(chunked_docs)} chunks.")
    
    # create whoosh schema: ID for unique chunk, TEXT for full-text search
    schema = Schema(
        id=ID(stored=True, unique=True), 
        content=TEXT(stored=True)
        )
    
    # prepare index directory
    index_dir = os.path.join(db_path, "whoosh_index")
    if not os.path.exists(index_dir):
        os.makedirs(index_dir)
        ix = index.create_in(index_dir, schema)
        print(f"Created new Whoosh index at {index_dir}.")
    
    else:
        if index.exists_in(index_dir):
            print(f"Whoosh index already exists at {index_dir}, recreating...")
            
            # clean old index
            for f in os.listdir(index_dir):
                os.remove(os.path.join(index_dir, f))
                
            ix = index.create_in(index_dir, schema)
            
        else:
            ix = index.create_in(index_dir, schema)
            print(f"Created new Whoosh index at {index_dir}.")

    writer = ix.writer()
    for i, chunk in enumerate(tqdm(chunked_docs, desc="Indexing chunks")):
        writer.add_document(
            id=str(i), 
            content=chunk
            )
    writer.commit()
    print(f"Indexed {len(chunked_docs)} chunks into Whoosh.")
    
# -------- Main Ingest Function with Argument Parsing --------    
import argparse
def parse_args():
    parser = argparse.ArgumentParser(
        description="Ingest vectors into a database using a specified model and dataset."
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to the input dataset."
    )
    parser.add_argument(
        "--database-path",
        type=str,
        required=True,
        help="Path to the output database file."
    )

    subparsers = parser.add_subparsers(dest='strategy')
    subparsers.required = True
    parser_vector = subparsers.add_parser('vector')
    parser_vector.add_argument(
        "--model",
        type=str,
        required=True,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Name or path of the model to use for vector generation."
    )
    parser_vector.add_argument(
        "--embedding-size", 
        type=int, 
        required=True, 
        default=384,
        help="Embedding size of the vectors."
    )

    subparsers.add_parser('kw')    
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # load documents
    documents = load_jsonl_files(args.dataset_path)

    if args.strategy == "vector":
        ingest_vector(documents, args.database_path, args.model, args.embedding_size)
    elif args.strategy == "kw":
        ingest_text(documents, args.database_path)
        

if __name__ == "__main__":
    main()
