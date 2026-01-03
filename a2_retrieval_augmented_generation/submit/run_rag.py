python submit/rag.py \
    --generation-model openai/gpt-oss-120b \
    --database-path /Users/chenxi/Documents/GitHub/agent/assignment-2-rag-Chenxi1818/db \
    --query "what is treatment for pain?" \
    vector \
    --embedding-model sentence-transformers/all-MiniLM-L6-v2
    

python submit/rag_10qiz.py \
    --generation-model openai/gpt-oss-120b \
    --database-path /Users/chenxi/Documents/GitHub/agent/assignment-2-rag-Chenxi1818/db \
    vector \
    --embedding-model sentence-transformers/all-MiniLM-L6-v2
    

python submit/rag_10qiz.py \
    --generation-model openai/gpt-oss-120b \
    --database-path /Users/chenxi/Documents/GitHub/agent/assignment-2-rag-Chenxi1818/db \
    kw
    