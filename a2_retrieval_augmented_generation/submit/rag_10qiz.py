import argparse
import os
import numpy as np
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from whoosh.qparser import QueryParser
from whoosh import index
from langchain_openai import ChatOpenAI
from inference_auth_token import get_access_token

#%%
import json
import ast
ques_file = "/Users/chenxi/Documents/GitHub/agent/assignment-2-rag-Chenxi1818/dataset/questions.jsonl"

questions = []
keywords = []
with open(ques_file, "r", encoding="utf-8") as f:
    for line in f:
        data = ast.literal_eval(line)
        questions.append(data['question'])
        keywords.append(data['keywords'])

#%%
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a RAG pipeline with a specified retrieval strategy and models."
    )

    parser.add_argument("--generation-model", type=str, required=True, help="LLM for text generation.")
    parser.add_argument("--database-path", type=str, required=True, help="Path to database.")
    # parser.add_argument("--query", type=str, required=True, help="Query string to retrieve and generate an answer for.")

    subparsers = parser.add_subparsers(dest="strategy", required=True)

    # Vector-based retrieval
    parser_vector = subparsers.add_parser("vector")
    parser_vector.add_argument("--embedding-model", type=str, required=True, help="Embedding model name or path.")

    # Keyword-based retrieval
    subparsers.add_parser("kw")

    return parser.parse_args()


def query_milvus(database_path: str, embedding_model: str, query: str):
    # 1. Reconnect to the database
    db_path = os.path.join(database_path, "milvus_literature.db")
    
    client = MilvusClient(uri=db_path)

    collection_name = "literature_collection"

    embed_model = SentenceTransformer(embedding_model)
    query_embedding = embed_model.encode([query]).astype(np.float32)

    search_results = client.search(
        collection_name=collection_name,
        data=query_embedding.tolist(),
        limit=5,
        search_params={"metric_type": "IP", "params": {}},
        output_fields=["text"]
    )

    print("Top 5 matching chunks:")
    for result in search_results[0]:
        print(f"Score: {result['distance']:.4f}")
        print(f"Text: {result['entity']['text'][:300]}...")
        print("-" * 80)

    context = "\n".join([result['entity']['text'] for result in search_results[0]])
    # print(context)
    return context


def query_whoosh(database_path: str, keywords: list[str], top_k: int = 5):
    index_dir = os.path.join(database_path, "whoosh_index")
    if not index.exists_in(index_dir):
        raise ValueError(f"No Whoosh index found at {index_dir}")

    ix = index.open_dir(index_dir)
    query_text = " ".join(keywords)

    parser = QueryParser("content", schema=ix.schema)
    query = parser.parse(query_text)

    with ix.searcher() as searcher:
        results = searcher.search(query, limit=top_k)
        hits = [
            {"rank": i + 1, "score": r.score, "content": r["content"]}
            for i, r in enumerate(results)
        ]

    print(f"Top {len(hits)} results for query: {keywords}")
    for h in hits:
        print(f"[{h['rank']}] (Score: {h['score']:.4f}) {h['content'][:100]}...")

    context = "\n".join([h["content"] for h in hits])
    return context

# from langchain.agents import create_agent

from openai import OpenAI
def query_llm(generation_model: str, context, query) -> str:
    access_token = get_access_token()
    
    llm = OpenAI(
        api_key=access_token,
        base_url="https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1"        
    )
    
    SYSTEM_PROMPT = """
    You are a helpful assistant that can answer the question based on the given context.
    """
    
    USER_PROMPT = f"""
    Use the following pieces of information enclosed in <context> tag to provide an answer to the question enclosed in <question> tags. 
    Give me a single paragraph to answer.
    <context>
    {context}
    </context>
    <question>
    {query}
    </question>
    Answer: 
    """
    
    response = llm.chat.completions.create(
        model=generation_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT}
            ]
        )

    return response.choices[0].message.content


if __name__ == "__main__":
    args = parse_args()

    print(f"Strategy: {args.strategy}")
    print(f"Generation model: {args.generation_model}")
    # print(f"Embedding model: {args.embedding_model}")
    print(f"Database path: {args.database_path}")
    # print(f"Query: {args.query}")

    if args.strategy == "vector":
        print("Using vector-based retrieval.")
        responses_ve = []
        for query in questions:
            context = query_milvus(args.database_path, args.embedding_model, query)
            response1 = query_llm(args.generation_model, context, query)
            
            responses_ve.append(response1)
            
        print(f"the answers to 10 questions based on Vector database {responses_ve}")
        
    elif args.strategy == "kw":
        print("Using keyword-based retrieval.")
        responses_kw = []
        for keyword in keywords:
            context = query_whoosh(args.database_path, keyword)
            response2 = query_llm(args.generation_model, context, keyword)
            
            responses_kw.append(response2)
        
        print(f"the answers to 10 questions based on text database {responses_kw}")
        
    else:
        raise ValueError(f"Unknown strategy: {args.strategy}")

    
    print("\n=== Final Answer ===")
    
    print("----------------------------")
    
