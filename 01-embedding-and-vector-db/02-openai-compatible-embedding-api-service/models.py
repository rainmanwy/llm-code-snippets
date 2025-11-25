"""
Created by Rainmanwy
"""
from typing import List, Optional, Union, Literal

from pydantic import BaseModel, Field


class OpenAICreateEmbeddingRequest(BaseModel):
    model: Optional[str] = Field(description="The model to use for generating completions.", default=None)
    input: Union[str, List[str]] = Field(description="The input to embed.")
    user: Optional[str] = Field(default=None)
    encoding_format: Optional[str] = Field(default=None)


class OpenAIEmbeddingUsage(BaseModel):
    prompt_tokens: int = Field(default=0)
    total_tokens: int = Field(default=0)


class OpenAIEmbedding(BaseModel):
    index: int
    object: Optional[str] = Field(default="embedding")
    embedding: Union[List[float], List[List[float]], str, List[str]]


class OpenAICreateEmbeddingResponse(BaseModel):
    object: Literal["list"] = Field(default="list")
    model: str
    data: List[OpenAIEmbedding]
    usage: OpenAIEmbeddingUsage




