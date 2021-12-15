from sentence_transformers import SentenceTransformer, util
from wasabi import msg 
import typer 

"""
Load NLP model
"""
base_model = 'paraphrase-multilingual-MiniLM-L12-v2'
msg.info("Loading model")
model = SentenceTransformer(base_model)
msg.good("Embedding model loaded")

def main(
    query: str = typer.Argument(..., help="query to encode using the NLP model")
):
    msg.info(f"Encoding query {query}")
    try:
        embedding = model.encode(query)
        msg.good("Success encoding query")
    except:
        msg.fail("Error encoding query")

    msg.info("Embedding:")
    print(embedding)

if __name__ == '__main__':
    typer.run(main)