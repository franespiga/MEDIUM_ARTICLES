import io

from starlette.responses import Response
import cloudpickle
import pandas as pd 
import tensorflow as tf
import requests
from fastapi import FastAPI, File
from pydantic import BaseModel
from typing import Dict, Any 
import elastic_utils


embedding_servicer = "http://embeddings:8502"

msg.info("Connecting to ElasticSearch...")
HOST = 'localhost'
PORT = 9200
# CONNECT TO INSTANCE
es = elastic_utils.get_connection_to_es(HOST, PORT)
msg.good(f"Connected to ElasticSearch node {HOST}:{PORT}")

def prettify_results(results):
    pretty_results = []

    cols = ['title', 'description', 'authors', 'categories', 'ISBN']
    for r in results:
        aux = {}
        for c in cols:
            aux[c] = r[c]
        pretty_results.append(aux)
    return pretty_results

app = FastAPI(
    title="Recomendador de libros",
    description="""Obtén una recomendación basada en AI.
                   Visit this URL at port 8501 for the streamlit interface.""",
    version="0.1.0",
)
  
def compute_scores(a, b):
    #a = tf.nn.l2_normalize(a, axis=1)
    #b = tf.nn.l2_normalize(b, axis=1)
    cosine_similarities = tf.matmul(a, b, transpose_b = True)
    #clip_cosine_similarities = tf.clip_by_value(cosine_similarities, -1.0, 1.0)
    scores = cosine_similarities #1.0 - tf.acos(clip_cosine_similarities) / math.pi
    return scores

def embed(query, server_url : str):
    print(f"Sending to url {server_url}")
    print(f"Description: {query}")
    r = requests.post(
        server_url, json = query, timeout = 8000
    )
    return r

@app.post("/get_recommendations")
def get_recommendations(query:Dict[Any,Any], top_k:int = 5):
    """Get segmentation maps from image file"""

    candidates = tf.constant([DATA[i]['embedding'] for i in query['ISBN_array']])
    print(f"With {len(query['ISBN_array'])} candidate(s)")
    print(candidates.shape)
    if len(candidates.shape)==1:
        print("SINGLE INPUT ENTERS HERE")
        candidates = tf.nn.l2_normalize(candidates, axis=0)
        candidates = tf.expand_dims(candidates, 0)
    else:
        candidates = tf.math.reduce_mean(candidates, axis = 0)
        candidates = tf.nn.l2_normalize(candidates, axis=0)
        candidates = tf.expand_dims(candidates, 0)
    
    if 'top_k' in query:
        top_k = query['top_k']

    sc = compute_scores(candidates, embeddings)

    results = tf.math.top_k(sc, top_k * 10)

    df_books = pd.DataFrame({'title': [DATA[ISBNs[i]]['title'] for i in results[1].numpy().squeeze().tolist()],
                              'ISBN': [DATA[ISBNs[i]]['isbn10'] for i in results[1].numpy().squeeze().tolist()]})
    df_books = df_books[~df_books.ISBN.isin(query['ISBN_array'])].head(top_k)
    print(df_books)
    return df_books.to_json()#Response(books)


@app.post("/get_recommendation_from_description")
def get_recommendations_from_description(query:Dict[Any,Any], top_k:int = 5):
    candidates = embed(query, f"{embedding_servicer}/embed_text").json()
   
    msg.info("Retrieving candidates based on description:", query['description'])
    results = elastic_utils.get_most_similar(candidates, ['description_embedding', 'categories_embedding'], es, 'books', 10, timeout_secs = 300, debug_mode = True)
    print(prettify_results(results['results']))

    return df_books.to_json()#Response(books)