import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a RAG (Retrieval-Augmented Generation) pipeline with a specified strategy and models."
    )

    parser.add_argument(
        "--generation-model",
        type=str,
        required=True,
        help="Name or path of the model used for text generation."
    )

    parser.add_argument(
        "--database-path",
        type=str,
        required=True,
        help="Path to the vector or keyword database."
    )

    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="The input query string to retrieve and generate an answer for."
    )

    subparsers = parser.add_subparsers(dest='strategy')
    subparsers.required = True
    parser_vector = subparsers.add_parser('vector')
    parser_vector.add_argument(
        "--embedding-model",
        type=str,
        required=True,
        help="Name or path of the model to use for vector generation."
    )

    subparsers.add_parser('kw')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(f"Strategy: {args.strategy}")
    print(f"Generation model: {args.generation_model}")
    print(f"Embedding model: {args.embedding_model}")
    print(f"Database path: {args.database_path}")
    print(f"Query: {args.query}")