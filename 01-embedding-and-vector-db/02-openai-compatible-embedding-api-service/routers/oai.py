"""
Created by Rainmanwy
"""
import os
import base64

from fastapi import APIRouter
import numpy as np

from models import (
    OpenAICreateEmbeddingRequest,
    OpenAICreateEmbeddingResponse,
    OpenAIEmbedding,
    OpenAIEmbeddingUsage
)
from embedding_model_loader import EMBEDDINGS, EMBEDDING_MODEL


router = APIRouter(
    prefix='/v1',
    tags=['openai']
    )


def _convert_list_to_base64(emb):
    """Convert embedding list to base64 encoded string.
    
    Args:
        emb: List of float values representing the embedding vector.
        
    Returns:
        str: Base64 encoded string representation of the embedding.
    """
    float_arr = np.array(emb, dtype=np.float32)
    raw_bytes = float_arr.tobytes()
    return base64.b64encode(raw_bytes).decode()


@router.post("/embeddings")
async def openai_embeddings(request: OpenAICreateEmbeddingRequest) -> OpenAICreateEmbeddingResponse:
    """Generate embeddings for input text using OpenAI-compatible API.
    
    Args:
        request: OpenAI embedding request containing input text(s) and options.
        
    Returns:
        OpenAICreateEmbeddingResponse: Response containing embeddings and metadata.
    """
    input_texts = request.input
    input_texts = input_texts if isinstance(input_texts, list) else [input_texts]

    embeds = await EMBEDDINGS.aembed_documents(input_texts)
    if request.encoding_format is None or request.encoding_format == 'float':
        data = [OpenAIEmbedding(embedding=emb, index=idx, object="embedding") for idx, emb in enumerate(embeds)]
    else:
        data = [OpenAIEmbedding(embedding=_convert_list_to_base64(emb), index=idx, object="embedding") for idx, emb in enumerate(embeds)]
    response = OpenAICreateEmbeddingResponse(
        object = 'list',
        data = data,
        model = os.path.basename(EMBEDDING_MODEL),
        usage = OpenAIEmbeddingUsage(prompt_tokens=1, total_tokens=1)
    )
    return response
