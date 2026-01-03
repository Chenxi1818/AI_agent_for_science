[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/I3mNBtJw)
# A2 — Retrieval Augmented Generation
## Tasks
1. Ingest data into Milvus (vector database)
1. Ingest data into Whoosh
1. Set up augmented retrieval tools

**Important Notes (Changes from Assignment 1)**: 
- For this assignment, **use what ever packages you'd like**. Please update requirements.txt with what you use.
- You **do not have to use the inference service for embeddings**. You can use whatever model/technique you'd like. Include your choice in the a2_report.pdf.  You should not require any API keys (but can get the access token for the argonne inference service).
Please **use the inference service for chat completion/generation**.
- **You do not need to stick to the strict semantics of the command** but provide instructions on how to run your ingestion pipeline. 
## Deliverables
 - ingest.py
 - rag.py
 - a2_report.pdf

## Details

### Task 1: Ingest data into Milvus
The github repository contains scientific publications sampled from NIH_LitArch, parsed and provided as plain text in .jsonl format. While models currently are improving their support for long context lengths, long contexts are more costly and perhaps more unreliable than short contexts. In order to ingest the documents, start by chunking each document into shorter documents that are more manageable for retrieval.  I recommend LangChain’s [`RecursiveTextSplitter`](https://python.langchain.com/api_reference/text_splitters/character/langchain_text_splitters.character.RecursiveCharacterTextSplitter.html) which tries to split the document into paragraphs, then sentences if the paragraphs are too large. For each chunk, create an embedding using one of the embedding models from the Argonne Inference Service, and insert it into a [milvus](https://milvus.io/docs/quickstart.md#Set-Up-Vector-Database) database.

Test your code using the following command:

```python ingest.py vector --model <model> --embedding-size <int> --dataset-path <path> --database-path <path>```

### Task 2: Ingest data into Whoosh
We will also implement a plain text search approach for retrieval. Use the same method you wrote above to chunk the documents. Then insert them into a [whoosh](https://whoosh.readthedocs.io/en/latest/quickstart.html) database. Whoosh is an embedded database for indexing and searching documents. Implement a text only schema for a document collection and insert documents into that collection.

Test your code using the following command:

```python ingest.py kw --dataset-path <path> --database-path <path>```

### Task 3: Set up Retrieval Tools
We will now use tool calling to implement retrieval augmented generation. Implement two tools: an embedding retrieval tool and a key word retrieval tool. The embedding retrieval tool should accept a query string as an argument, use the embedding model to create an embedding vector for the query and search the milvus database for results. The key-word query tool should accept a list of key words and search the whoosh database for results. Both should provide the top k retrieved chunks as a result string.

Then implement a query method that creates an agent that uses the generation model with access to a retrieval tool to answer a question. Based on the strategy given in the argument, add only one of the tools to the agent.

```python rag.py --generation-model <model> --query <query> --database-path <path>  <vector | kw>  --embedding-model <model> ```

Come up with a dataset of at least 10 questions to evaluate against both retrieval strategies. 
Do you find any draw-backs or advantages of either approach?. It may be hard to find the  differences with the size of documents we provided. If you do not see a difference, hypothesize why this is, and in what scenarios if any you would see a difference. Include a table comparing the performance in a report, along with this analysis.

(To come up with questions, you can read a few of the documents you ingest, or get ChatGPT to read a document and come up with a question.)