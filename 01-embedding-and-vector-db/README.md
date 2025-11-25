# My Embedding and Vector DB Journey

## Embedding Techniques
* **Models**: from 'shibing624/text2vec-base-chinese-sentence' to 'BAAI/bge-m3'
* **Vector Databases**: From 'faiss' to '[Milvus](https://milvus.io/)'
* **Reranker Models**: From 'netease-youdao/bce-reranker-base_v1' to 'BAAI/bge-reranker-v2-m3'

## Examples
* Embedding Example: Several examples of embedding and vectorization based on LangChain
* OpenAI Compatible Embedding API Service: Implemented with [FastAPI](https://fastapi.tiangolo.com/)
    * Start Service: `python main.py`
    * Test Service: `curl -X POST "http://127.0.0.1:8000/v1/embeddings" -H "Content-Type: application/json" -d '{"model": "BAAI/bge-m3", "inputs": ["hello"]}'`

