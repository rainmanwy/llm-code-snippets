"""
Created by Rainmanwy
"""
import json

from loguru import logger
from fastapi import FastAPI, Request

from routers import oai


app = FastAPI(docs_url=None, redoc_url=None)
app.include_router(oai.router) # openai compatible api


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    json_data = ''
    try:
        json_data = await request.json()
    except json.decoder.JSONDecodeError as e:
        pass
    logger.debug(f"Request Body: {json_data}")
    response = await call_next(request)
    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)