from sentence_transformers import SentenceTransformer, util
from wasabi import msg 
import typer 
import cloudpickle
import tensorflow as tf 
import jsonlines

"""
Scoring function
"""
import math 

def compute_scores(a, b):
    cosine_similarities = tf.matmul(a, b, transpose_b = True)
    return cosine_similarities

"""
Load NLP model
"""
base_model = 'paraphrase-multilingual-MiniLM-L12-v2'
msg.info("Loading model")
model = SentenceTransformer(base_model)
msg.good("Embedding model loaded")

def main(
    query: str = typer.Argument(..., help="query to encode using the NLP model"), 
    top_k: int = typer.Option(5, help = "Number of top K candidates to be retrieved"),
    database: str = typer.Option('./data/database_100k_mvp.jsonl', "--db-path", "-db", help="Database file location"),
    embeddings_path: str = typer.Option('./data/embeddings_100k_mvp.p', "--emb-path", help="Embeddings file location"),
):
    msg.info("Loading embeddings")
    try:
        DB_embeddings = tf.constant(cloudpickle.load(open(embeddings_path,'rb')))
        DB_embeddings = tf.nn.l2_normalize(DB_embeddings, axis=1)
        msg.good("Embeddings loaded")
    except:
        msg.fail("Error loading embeddings", exits = 1)

    msg.info(f"Encoding query {query}")
    try:
        embedding = model.encode(query)
        msg.good("Success encoding query")
    except:
        msg.fail("Error encoding query")

    msg.info("Retrieving the best candidates from the database")
    embedding = tf.nn.l2_normalize(tf.expand_dims(tf.constant(embedding),0), axis=1)
    scores = compute_scores(embedding, DB_embeddings)
    # the top k candidates are retrieved
    candidates = tf.math.top_k(scores, top_k)[1].numpy().squeeze().tolist()

    results = []
    with jsonlines.open(database, mode='r') as reader:
        for idx, obj in enumerate(reader):
            if idx in candidates:
                result = {}
                for k,v in obj.items():
                    if k in ['title', 'authors', 'description']:
                        result[k] = v
                results.append(result)
            
    msg.info(f"Printing {query} candidates for query {query}")
    for result in results:
        print(result)
        print('*************\n')

if __name__ == '__main__':
    typer.run(main)