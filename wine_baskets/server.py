from fastapi import FastAPI, HTTPException
from typing import Dict, Any 
from wasabi import msg
import uvicorn

app_name="wine-basket"

# METADATA ----------------------------------------------------------------
tags_metadata = [
    {
        "name": "price_basket",
        "description": "Get the best basket given a number of wines and a max price" ,
    }
]
    
# APP DEFINITION
app = FastAPI(
    title="WINE BASKET", 
    description="""Builds the best wine basket according to some criteria""",
    version="0.1.0",
    openapi_tags=tags_metadata
)

# ENDPOINTS ---------------------------------------------------------------
@app.get("/health")
def hb() -> Dict:
    return {'heartbeat': True}

@app.get("/price_basket", tags = "price_basket")
def embed(query:Dict[Any,Any]) -> Dict:
    """Encode a given set of Goods & Services"""
    if not 'terms' in query:
        raise HTTPException(status_code=400, detail="No terms present in the query to be encoded.")
    else:
        q = query['terms'] if isinstance(query['terms'],list) else query['terms'].split(";")
        msg.info("Computing embeddings...", q)
        embedding=encode(q)
        msg.good("Embeddings computed")
        return {'encoded_terms':embedding.tolist()}


if __name__ == "__main__":
    uvicorn.run(
        f"{encoder_suffix}server:app",
        host=configuration.get("encoding_host", "0.0.0.0"),
        port=configuration.get("encoding_port", 8502),
    )