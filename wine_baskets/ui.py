import requests
import streamlit as st
from streamlit_tags import st_tags_sidebar
import jsonlines
from tqdm import tqdm 
from src.knapsack import create_knapsack, solve_knapsack
import pandas as pd 

st.set_page_config(page_title="Wine basket v0", page_icon=None, initial_sidebar_state="expanded", layout='wide')

@st.experimental_singleton
def load_wine_data():
    data_path = '../wine/data/winedata_bodeboca_db.jsonl'
    Wines = []
    Regions = []
    Types = []
    Ratings = []
    Prices = []
    with jsonlines.open(data_path, 'r') as reader:
        for obj in tqdm(reader):
            try:
                w = obj['wine_name']
                r = obj['info']['do']
                t = obj['info']['type']
                p = obj['price']
                rr = obj['ratings']
                Wines.append(w)
                Regions.append(r)
                Types.append(t)
                Prices.append(p)
                Ratings.append(rr)
            except:
                pass


    df = pd.DataFrame({'wine':Wines, 'region':Regions, 'type':Types, 'price':Prices, 'rating':Ratings})

    return df 

wine_data = load_wine_data()

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def do_wine_basket():
    df_ = wine_data.copy()
    if len(wine_regions)>0:
        df_ = df_.merge(pd.DataFrame({'region':st.session_state.wine_regions}), on = 'region', how = 'inner')

    if len(wine_types)>0:
        df_ = df_.merge(pd.DataFrame({'type':st.session_state.wine_types}), on = 'type', how = 'inner')
        

    model = create_knapsack(df_, 
        maxPrice = st.session_state.max_price, 
        maxWines = st.session_state.max_bottles)

    result = solve_knapsack(model, solver = {'name':'cbc', 'path':'./bin/cbc.exe'})
    st.dataframe(result)



# SIDEBAR
st.sidebar.markdown("<h2>Wine basket builder</h2>", unsafe_allow_html = True)

wine_types = st.sidebar.multiselect(label = "Wine types", 
        options = list(set(wine_data.type)),  key = 'wine_types')

wine_regions = st.sidebar.multiselect(label = "Wine regions", 
        options = list(set(wine_data.region)),  key = 'wine_regions')

max_price = st.sidebar.number_input(label="max total price", key = "max_price")
max_bottles = st.sidebar.number_input(label="max different wines", key = "max_bottles")

run_basket = st.sidebar.button(label = "create basket", on_click = do_wine_basket)