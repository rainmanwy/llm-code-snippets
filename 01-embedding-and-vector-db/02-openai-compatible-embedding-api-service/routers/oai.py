"""
Created by Rainmanwy
"""
import os
import base64

from fastapi import APIRouter, HTTPException
from loguru import logger
import numpy as np

from models import *
from embedding_model_loader import EMBEDDINGS, EMBEDDING_MODEL


router = APIRouter(
    prefix='/v1', 
    tags=['openai']
    )


def _convert_list_to_base64(emb):
    float_arr = np.array(emb, dtype=np.float32)
    raw_bytes = float_arr.tobytes()
    return base64.b64encode(raw_bytes).decode()


@router.post("/embeddings")
async def openai_embeddings(request: OpenAICreateEmbeddingRequest) -> OpenAICreateEmbeddingResponse:
    input = request.input
    input = input if isinstance(input, list) else [input]

    embeds = await EMBEDDINGS.aembed_documents(input)
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
