import io

from starlette.responses import Response
from sentence_transformers import SentenceTransformer, util
from fastapi import FastAPI, File
from pydantic import BaseModel
from typing import Dict, Any 
import json 

#class Books(BaseModel):

app = FastAPI(
    title="Generador de embeddings", 
    description="""Transforma cualquier petición o descripción en un vector.""",
    version="0.1.0",
)

base_model = 'paraphrase-multilingual-MiniLM-L12-v2'
model = SentenceTransformer(base_model)

@app.post("/embed_text")
def embed(query:Dict[Any,Any]):
    """Get embeddings of a given query"""
    print(query)
    embedding=  model.encode(query['text']).tolist()
    return {'embedding':embedding}


