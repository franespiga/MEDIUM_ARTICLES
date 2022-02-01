import utils
from sentence_transformers import SentenceTransformer, util
from wasabi import msg
import typer

def prettify_results(results):
    pretty_results = []

    cols = ['title', 'description', 'authors', 'categories']
    for r in results:
        aux = {}
        for c in cols:
            aux[c] = r[c]
        pretty_results.append(aux)
    return pretty_results
    
def main(    
    host : str = typer.Argument("localhost", help="ElasticSearch host address"),
    port : int = typer.Argument(9200, help="ElasticSearch port"),
    description : str = typer.Argument("Quiero leer un libro de castillos y princesas", help="Sample description to test"), 
    category : str = typer.Argument("Quiero leer algo del realismo ruso", help="Sample category to test"), 
    base_model: str = typer.Option('paraphrase-multilingual-MiniLM-L12-v2', "--nlp-model", help="NLP model from SBERT"),
    ):

    msg.info("Connecting to ElasticSearch...")

    # CONNECT TO INSTANCE
    es = utils.get_connection_to_es(host, port)
    msg.good(f"Connected to ElasticSearch node {host}:{port}")

    # CREATE EMBEDDINGS
    msg.info("Loading model")
    model = SentenceTransformer(base_model)
    msg.good("Embedding model loaded")

    description_embedding = model.encode(description)
    category_embedding = model.encode(category)

    # RETRIEVING TOP K RESULTS BASED ON DESCRIPTION
    msg.info("Retrieving candidates based on description:", description)
    results = utils.get_most_similar(list(description_embedding), 'description_embedding', es, 'books', 10, timeout_secs = 300, debug_mode = False)
    print(prettify_results(results['results']))

    # RETRIEVING TOP K RESULTS BASED ON CATEGORY
    msg.info("Retrieving candidates based on category:", category)
    results = utils.get_most_similar(list(category_embedding), 'categories_embedding', es, 'books', 10, timeout_secs = 300, debug_mode = False)
    print(prettify_results(results['results']))

if __name__ == '__main__':
    typer.run(main)