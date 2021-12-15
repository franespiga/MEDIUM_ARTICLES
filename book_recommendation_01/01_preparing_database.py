import pandas as pd
import numpy as np 
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm 
import tensorflow as tf
import jsonlines
import cloudpickle
from wasabi import msg
import typer

def main(
    in_path : str = typer.Argument(..., help="Location of the Book Repository original files"), 
    out_path : str = typer.Argument(..., help="Location of the output files (Database and Embeddings"), 
    base_model : str = typer.Option('paraphrase-multilingual-MiniLM-L12-v2', '--nlp-model', help = "Sentence Transformer model to compute the embeddings"), 
    recreate_database : bool = typer.Option(False, '--recreate-database'),
    recreate_embeddings : bool = typer.Option(False, '--recreate-embeddings')
):
    if recreate_database:
        """
        Load Data
        """
        # Data has been obtained from https://www.kaggle.com/sp1thas/book-depository-dataset
        # Original source: book depository
        msg.info("Loading data")
        data= pd.read_csv(f'{in_path}/dataset.csv')
        authors = pd.read_csv(f'{in_path}/authors.csv')
        categories = pd.read_csv(f'{in_path}/categories.csv')
        msg.good("Data loaded")

        """
        Load NLP model
        """
        msg.info("Loading model")
        model = SentenceTransformer(base_model)
        msg.good("Embedding model loaded")

        """
        Create the dataset
        """
        percentage_from_source = 0.1 # Use 10% of the original dataset
        with jsonlines.open(f'{out_path}/database_100k_mvp.jsonl', mode='w') as writer:
            indexed = 0
            for idx, row in tqdm(data.sample(round(data.shape[0]*percentage_from_source)).iterrows()):
                try:
                    record = {
                        'title': row['title'], 
                        'ISBN' : str(row['isbn13']),
                        'description' : row['description'],
                        'authors' : authors[authors.author_id.isin(eval(row['authors']))].author_name.tolist(),
                        'categories' : categories[categories.category_id.isin(eval(row['categories']))].category_name.tolist(),
                    }

                    record['description_embedding'] = model.encode(record['description']).tolist()
                    record['categories_embedding'] = np.mean(model.encode(record['categories']), axis = 0).tolist()
                    writer.write(record)
                    indexed+=1
                    if indexed % 100 == 0:
                        msg.info(f"Indexed {indexed} records")
                except:
                    pass

    if recreate_database or recreate_embeddings:
        """
        Store the embeddings and check that the jsonlines file is readable
        """
        description_embeddings = []
        with jsonlines.open(f'{out_path}/database_100k_mvp.jsonl', mode='r') as reader:
            for obj in reader:
                description_embeddings.append(obj['description_embedding'])

        description = tf.constant(description_embeddings, tf.float32)
        cloudpickle.dump(description_embeddings, open(f'{out_path}/embeddings_100k_mvp.p','wb'))

        msg.good("DATABASE export successful")

if __name__ == '__main__':
    typer.run(main)