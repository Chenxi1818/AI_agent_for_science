

# create milvus database
python submit/ingest.py \
  --dataset-path /Users/chenxi/Documents/GitHub/agent/assignment-2-rag-Chenxi1818/dataset \
  --database-path /Users/chenxi/Documents/GitHub/agent/assignment-2-rag-Chenxi1818/db \
  vector \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --embedding-size 384

# create whoosh database
python submit/ingest.py \
  --dataset-path /Users/chenxi/Documents/GitHub/agent/assignment-2-rag-Chenxi1818/dataset \
  --database-path /Users/chenxi/Documents/GitHub/agent/assignment-2-rag-Chenxi1818/db \
  kw
  
