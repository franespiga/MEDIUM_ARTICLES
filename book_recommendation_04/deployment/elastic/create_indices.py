from wasabi import msg
from elasticsearch import Elasticsearch, helpers
from utils import *
import typer


def main(
    host : str = typer.Argument("localhost", help="ElasticSearch host address"),
    port : int = typer.Argument(9200, help="ElasticSearch port"),
    reset: bool = typer.Option(True, help="clear indices on ElasticSearch"),
    reset_all: bool = typer.Option(False, "--reset-all", help="clear indices on ElasticSearch")
):

#python index_data --host localhost --port 9200 --index signals --reset 1 --


    # CONNECT TO INSTANCE
    es = get_connection_to_es(host, port)

    existing_indices =[i for i in es.indices.get('*')]
    msg.info(f"Current indices {', '.join(existing_indices)}")
    if reset_all:
        msg.info("Deleting all indices")
        for i in existing_indices:
            es.indices.delete(index=i, ignore=[400, 404])

    # REBUILD INDEX IF NECESSARY
    indices = {'description_embedding' : {
                    'type':'dense_vector',
                    'name':'description_embedding',
                    'dims':384
                },
                'categories_embedding' : {
                    'type':'dense_vector',
                    'name':'categories_embedding',
                    'dims':384
                },
               'timestamp' : {'type':'timestamp'}, 'books' : {'type':'string'}}


    if reset:
        indices_to_create = list(indices.keys())
    else:
        indices_to_create = list(set(list(indices.keys()))-set(existing_indices))

    create_all = True
    if create_all:
        info = create_indices(es, delete_if_exists = reset)
        print(info)
    else:
        for idx in indices_to_create:
            index_info = indices[idx]
            if index_info['type'] == 'dense_vector': 
                msg.info(f"Creating dense vector index {idx}")
                info = create_index(index_info, es, delete_if_exists = reset)
                print(info)
            elif index_info['type'] == 'timestamp': 
                msg.info(f"Creating timestamp index {idx}")
                info = create_timestamp_index( es, delete_if_exists = reset)
                print(info)
            elif index_info['type'] == 'string': 
                msg.info(f"Creating string index {idx}")
                info = create_str_index(idx,  es, delete_if_exists = reset)
                print(info)
if __name__ == '__main__':
    typer.run(main)