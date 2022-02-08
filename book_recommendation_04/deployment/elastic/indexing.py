from wasabi import msg
import typer
from pathlib import Path

from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer, util

import pandas as pd
import os
import numpy as np 
import jsonlines
import json
import math

from utils import *
from tqdm import tqdm 
import datetime 

def check_record(record):
    for k, v in record.items():
        if isinstance(v, str):
            if v=='':
                return False
        if isinstance(v, list):
            if len(v)==0:
                return False
            elif not isinstance(v[0] ,str):
                return False

    return True

def main(
    # fmt: off
    in_file: str = typer.Argument(..., help="Path to the input file"),
    host : str = typer.Argument("localhost", help="ElasticSearch host address"),
    port : int = typer.Argument(9200, help="ElasticSearch port"),
    base_model: str = typer.Option('paraphrase-multilingual-MiniLM-L12-v2', "--nlp-model", help="NLP model from SBERT"),
    
    n_process: int = typer.Option(1, "--n-process", "-n", help="Number of processes (multiprocessing)"),
    start_from:int = typer.Option(1, "--start-line", help = "Start indexing from the Nth record"),
    resume_from_log:str = typer.Option(None, "--resume-from-log", help = "Resume from last line indexed from logfile"),
    docs_per_batch: int = typer.Option(10 ** 3, "--max-docs", "-m", help="Maximum docs per batch"),
    total_docs: int = typer.Option(10 ** 3, "--total-docs", help = "Total docs to index (debugging, local testing)")

    # fmt: on
):

    msg.info(f"Starting indexing process @{datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M')}")

    input_path = Path(in_file)
    if not input_path.exists():
        msg.fail("Can't find input file", in_file, exits=1)

    """
    Load NLP model
    """
    msg.info("Loading model")
    model = SentenceTransformer(base_model)
    msg.good("Embedding model loaded")


    """
    Connect to ElasticSearch database
    """
    # CONNECT TO INSTANCE
    es = get_connection_to_es(host, port)
    msg.good(f"Connected to ElasticSearch node {host}:{port}")

    """
    Read jsonlines file
    """
    required_columns = ['title', 'ISBN', 'description', 'authors', 'categories']
    embedding_columns = ['description', 'categories']
    params = {'index' : 'books'}#[f"{i}_embedding" for i in embedding_columns]}

    lines = 0
    batch = 0
    records = []

    if resume_from_log is not None:
        log = json.load(open(resume_from_log,'r'))
        start_from = log['last_line']+1
        if log['file'] != in_file:
            msg.fail(f"Source file from log {resume_from_log} ({log['file']}) is not the same as {in_file}", exits = 1)

    with jsonlines.open(in_file) as reader:
        for obj in tqdm(reader):
            lines+=1
            try:
                if lines<start_from:
                    pass
                else:
                    if check_record(obj):                    
                        for emb_col in embedding_columns:
                                emb = model.encode(obj[emb_col])
                                emb = np.mean(emb, axis = 0) if  len(emb.shape)==2 else emb
                                obj[f'{emb_col}_embedding'] = list(emb)

                        records.append(obj)
                        batch+=1
                    else:
                        pass
            except:
                pass

            if batch>=docs_per_batch:
                try:
                    index_data_to_es(records, params, es, docs_per_batch)
                    msg.good(f"Batch of size {docs_per_batch} successfully inserted in {host}:{port}")
                    batch = 0
                    records = []
                    msg.info(f"Last line processed from file {in_file}: {lines}")
                except:
                    msg.warn(f"Warning: batch was not inserted in ElasticSearch instance in {host}:{port}")
                    batch = 0
                    records = []

            if lines-start_from>=total_docs:
                break 

        if len(records)>0:
            # insert last batch
            msg.info(f"Last line processed from file {in_file}: {lines}")
            index_data_to_es(records, params, es, len(records))
            msg.good(f"Batch of size {len(records)} successfully inserted in {host}:{port}")
            msg.info(f"End of indexing process @{datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M')}")

    """
    log current indexing status
    """
    log = {'file': in_file, 
     'last_line': lines}
    json.dump(log, 
        open(f"./indexing_log_{datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d%H%M')}_{os.path.basename(in_file)}.json", 'w'))
if __name__ == '__main__':
    typer.run(main)


# python index_data home/fespiga/git/LUCES/data/original_dump.jsonl --total-docs 1000
