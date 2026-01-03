import argparse
import json
import time

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from inference_auth_token import get_access_token
from openai import OpenAI
from pymilvus import MilvusClient
from tqdm import tqdm


def query_model_for_embedding(input, model, timeout):
    '''invoke a reasoning model'''
        # Get your access token
    access_token = get_access_token()

    client = OpenAI(
        api_key=access_token,
        base_url="https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1"
    )

    try:
        # start timer
        start_time = time.time()

        # override options just for this one call
        response = client.embeddings.create(
            model=model, # choose embedding model, eg. "mistralai/Mistral-7B-Instruct-v0.3-embed"
            input=input,
            encoding_format="float",
            timeout=timeout
            )
        
        # stop timer
        end_time = time.time()
        latency = end_time - start_time

        # get content and token count
        response_content = response.choices[0].message.content
        # Use word count as proxy
        token_count = len(response_content.split())

        return {
            "answer_text": response_content,
            "latency": latency,
            "tokens": token_count
        }

    # --- Improved Error Handling ---
    except APITimeoutError:
        print(f"Warning: Request timed out after {timeout} seconds.", file=sys.stderr)
        return None
    except APIError as e:
        print(f"Warning: API Error (Status {e.status_code}): {e.message}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        return None
    


def ingest_vector(documents: list[str], db_path: str, model: str, vector_size: int) -> None:
    '''
    documents: list[str] -- input type hint. tells the programmer that this parameter should be a list where each item inside the list is a str
    -> None -- return type hint
    '''


def ingest_text(documents: list[str], db_path: str) -> None:
    ...


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

    ''' add subparser to allow selecting the different task 
    python ingest.py vector --model <model> --embedding-size <int> --dataset-path <path> --database-path <path>
    
    '''
    subparsers = parser.add_subparsers(dest='strategy')
    subparsers.required = True
    parser_vector = subparsers.add_parser('vector')

    parser_vector.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name or path of the model to use for vector generation."
    )
    parser_vector.add_argument("--embedding-size", type=int, required=True, help="Embedding size of the vectors.")

    ''' add subparser to allow selecting the different task 
    python ingest.py kw --dataset-path <path> --database-path <path>
    
    '''
    subparsers.add_parser('kw')

    return parser.parse_args()

def main():
    args = parse_args()


if __name__ == "__main__":
    main()