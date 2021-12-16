import io
import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder
import streamlit as st
#import cloudpickle
import pandas as pd 
import numpy as np
import json

# interact with FastAPI endpoint
backend = "http://fastapi:8000/"
embedding_servicer = "http://embeddings:8502"

# data
@st.cache
def load_info():
    data = pd.read_csv('./books_info.csv').dropna().drop_duplicates().drop_duplicates(subset = 'title').sample(200000)
    return data
    #TITLES = data['title'].values.tolist()
    #ISBNs = data['isbn10'].values.tolist()
    #return TITLES, ISBNs

data = load_info()
TITLES = data['title'].values.tolist()
ISBNs = data['isbn10'].values.tolist()



def process(query, server_url: str, n_recomendaciones:int):
    print(f"Sending to url {server_url}")
    print(f"ISBNs: {', '.join(query)}")
    r = requests.post(
        server_url, json = {'ISBN_array':query, 'top_k':n_recomendaciones}, timeout=8000
    )
    print("RETURNING FROM BACKEND")
    return json.loads(r.json())

def embed(query, server_url : str, n_recomendaciones:int):
    print(f"Sending to url {server_url}")
    print(f"Description: {query}")
    r = requests.post(
        server_url, json = {'text':query, 'top_k':n_recomendaciones}, timeout = 8000
    )
    return json.loads(r.json())

# construct UI layout
st.sidebar.title("Recomendador de libros v0")

st.sidebar.write(
    """Obtenga sugerencias según sus gustos de lectura. 
         Seleccione varios títulos o ISBNs y pulse el botón."""
)  # description and instructions

# sidebar
st.sidebar.markdown("<h4>Libros por título</h4>", unsafe_allow_html = True)
titles = st.sidebar.multiselect("Título", TITLES, default=None)
st.sidebar.markdown("<h4>Libros por ISBN</h4>", unsafe_allow_html = True)
isbns = st.sidebar.multiselect("ISBN", ISBNs, default=None)

st.sidebar.markdown("<h4>Dinos brevemente qué te apetece leer</h4>", unsafe_allow_html = True)
desc = st.sidebar.text_area("Describe lo que te apetece leer", value = "")

n_recos = st.sidebar.number_input("Número de recomendaciones", min_value=1, max_value=20, value=5)

if st.sidebar.button("Recomiéndame"):
    book_titles = None
    col1, col2 = st.columns(2)
    if len(desc) ==0:
        query = []
        query_titles = data.loc[data['title'].isin(titles),:]['isbn10'].values.tolist()
        query.extend(isbns)
        query.extend(query_titles)
        query = list(set(query))
        book_titles = data.loc[data['isbn10'].isin(query)]['title'].values.tolist()
        

        results = process(query, f"{backend}get_recommendations", n_recos)
        
    else:
        results = embed(desc, f"{backend}get_recommendation_from_description", n_recos)

    recommendations = pd.DataFrame.from_dict(results)
    recommendations['rank'] = [i+1 for i in range(recommendations.shape[0])]
    col1.header("Sugerencias para")
    all_books =  ',<br>'.join(book_titles) if book_titles is not None else desc
    st.markdown(f"<h4>{all_books}</h4>", unsafe_allow_html = True)
    st.dataframe(recommendations)
    #st.dataframe(pd.DataFrame({'rank':[i+1 for i in range(len(recommendations))],'books':recommendations}))

#