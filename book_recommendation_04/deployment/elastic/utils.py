import requests, json, os
from elasticsearch import Elasticsearch, helpers
from tqdm import tqdm 
import jsonlines
import time
import datetime 
from typing import List, Tuple, Dict
import math 
import numpy as np 
from wasabi import msg


# CONNECTION -----------------------------
def get_connection_to_es(host, port):
    res = requests.get(f'http://{host}:{port}')
    print("Connecting to the node")
    print (res.content)
    return Elasticsearch([{'host': host, 'port': port}])
    
# INDICES --------------------------------
def create_indices(conn_es, delete_if_exists : bool = True):
    
    if delete_if_exists:
        existing_indices =[i for i in conn_es.indices.get('*')]
        for i in existing_indices:
            conn_es.indices.delete(index=i, ignore=[400, 404])
        
        print("Creating index")
        create_query = {
            "mappings": {
                "properties": {
                    f"{'description_embedding'}": {
                        "type": "dense_vector",
                        "dims": 384
                    },
                    f"{'categories_embedding'}": {
                        "type": "dense_vector",
                        "dims": 384
                    },
                    'title': {
                        "type" : "text"
                    }
                }


            }
        }

        conn_es.indices.create(index='books', body=create_query)
    return conn_es.indices.exists('books')

def create_index(idx_parameters, conn_es, delete_if_exists : bool = True):
    
    if delete_if_exists:
        conn_es.indices.delete(index=idx_parameters['name'], ignore=[400, 404])
        
        print("Creating index")

        create_query = {
            "mappings": {
                "properties": {
                    f"{idx_parameters['name']}": {
                        "type": "dense_vector",
                        "dims": idx_parameters['dims']
                    }
                }
            }
        }
        
        info = conn_es.indices.create(index=idx_parameters['name'], body=create_query)
    print(info)
    return conn_es.indices.exists(idx_parameters['name'])

def create_timestamp_index(conn_es, delete_if_exists : bool = True):
    
    if delete_if_exists:
        conn_es.indices.delete(index='timestamp', ignore=[400, 404])
        
        print("Creating index timestamp")

        create_query = {
            "mappings": {
                "properties": {
                    "timestamp": {
                        "type" : "date"
                    }
                }
            }
        }
        
        conn_es.indices.create(index='timestamp', body=create_query)
    return conn_es.indices.exists('timestamp')

def create_str_index(str_name, conn_es, delete_if_exists : bool = True):
    
    if delete_if_exists:
        conn_es.indices.delete(index=str_name, ignore=[400, 404])
        
        print(f"Creating index {str_name}")

        create_query = {
            "mappings": {
                "properties": {
                    str_name: {
                        "type" : "text"
                    }
                }
            }
        }
        
        conn_es.indices.create(index=str_name, body=create_query)
    return conn_es.indices.exists(str_name)
# INDEXING --------------------------------
def index_data_to_es(records : List, params : Dict, conn_es, dump_size : int=100):
    # number of entries in ElasticSearch
    current_elements = conn_es.count()['count']
    

    bulk = []
    i = 0
    
    for record in records:
        # Insert batch in ElasticSearch
        if i%dump_size==0:
            helpers.bulk(conn_es, bulk)
            bulk = []

        # Obtain timestamp
        timestr = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d')
        
        record = {**record,    
            '_index': params['index'],
            '_id' : record['ISBN'],
            'timestamp' : timestr
                 }

        bulk.append(record)
        
    helpers.bulk(conn_es, bulk)

    final_elements = conn_es.count()['count']
    
    return {'code':200, 'message': f"Inserted on index: {params['index']}; {final_elements-current_elements} of {len(records)} elements"}



# DATA RETRIEVAL
def get_most_similar(embedding, attribute , conn_es, 
                     index_ : str, top_k : int, timeout_secs : int = 300, debug_mode= False):
    
    if isinstance(attribute, str):
        search_query = {
            "size": top_k,
            "_source": {
                "includes": ["title", "authors", "categories", "description"]
            },
            "query": {
                "script_score": {
                    "query": {
                        "match_all": {}
                    },
                    "script": {
                        "source": f"cosineSimilarity(params.queryVector, '{attribute}') + 1.0",
                        "params": {
                            "queryVector": embedding
                        }
                    }
                }
            }
        }

    elif isinstance(attribute, list):

        search_query = {
            "size": top_k,
            "_source": {
                "includes": ["title", "authors", "categories", "description"]
            },
            "query": {
                "script_score": {
                    "query": {
                        "match_all": {}
                    },
                    "script": {
                        "source": f"0.5*(cosineSimilarity(params.queryVector, '{attribute[0]}') + cosineSimilarity(params.queryVector, '{attribute[1]}')) + 1.0",
                        "params": {
                            "queryVector": embedding
                        }
                    }
                }
            }
        }

    response = conn_es.search(
    index=index_,
    body=search_query,
    request_timeout=timeout_secs    
    )

    if debug_mode:
        msg.info("Entering debug mode, printing response...")
        print(response)
    IDS = [i['_id'] for i in response['hits']['hits']]

    return {'results':[{**conn_es.get(id = IDS[i], index = index_)['_source'], 
                                             'score':response['hits']['hits'][i]['_score']}for i in range(len(IDS))]}





def get_top_between_dates(signal_embedding, conn_es, 
                                index_date : str, 
                                index_similarity : str,
                                start_date :str, end_date : str,
                                top_k : int, timeout_secs : int = 300):

    search_query = {
        "size": top_k,
        "_source": {
            "includes": ["description"]
        },
        "query": {
            "range":{
                index_date: {
                    "gte": start_date, 
                    "lte": end_date
                }
            }
        }
    }

    
    response = conn_es.search(
    index=index_similarity,
    body=search_query,
    request_timeout=timeout_secs    
    )
    
    IDS = [i['_id'] for i in response['hits']['hits']]

    return {'results':[{**conn_es.get(id = IDS[i], index = index_similarity)['_source'], 
                                             'score':response['hits']['hits'][i]['_score']}for i in range(len(IDS))]}

def get_most_similar_between_dates(signal_embedding, conn_es, index_similarity : str, 
                                index_date : str, 
                                start_date :str, end_date : str,
                                include_start : bool, include_end :bool,
                                top_k : int, timeout_secs : int = 300):

    search_query = {
        "size": top_k,
        "_source": {
            "includes": ["description"]
        },
        "query": {
            "script_score": {
                "query": {
                    "range":{
                        index_date: {
                            f"gt{'e' if include_start else ''}": start_date, 
                            f"lt{'e' if include_start else ''}": end_date
                        }
                    }

                },
                "script": {
                    "source": "cosineSimilarity(params.queryVector, 'signals_embedding') + 1.0",
                    "params": {
                        "queryVector": signal_embedding
                    }
                }
            }


        }
    }
    
    response = conn_es.search(
    index=index_similarity,
    body=search_query,
    request_timeout=timeout_secs    
    )

    IDS = [i['_id'] for i in response['hits']['hits']]

    return {'results':[{**conn_es.get(id = IDS[i], index = index_similarity)['_source'], 
                                             'score':response['hits']['hits'][i]['_score']}for i in range(len(IDS))]}
