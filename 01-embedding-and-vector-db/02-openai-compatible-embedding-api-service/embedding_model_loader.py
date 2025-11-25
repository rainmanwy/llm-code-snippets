"""
Created by Rainmanwy
"""
import os
import datetime
from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger


os.environ['TRANSFORMERS_OFFLINE'] = '1'
EMBEDDING_MODEL = 'BAAI/bge-m3' # Embedding Model name or PATH

logger.info(f'Loading embedding model: {EMBEDDING_MODEL}, {str(datetime.datetime.now())}')
EMBEDDINGS = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu', 'trust_remote_code': True}
)