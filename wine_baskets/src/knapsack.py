import pyomo.environ as pyo
from typing import List, Dict, Any 
import pandas as pd 

def create_knapsack(df, maxPrice: float = 50.0, maxWines: int = 12):
    
    model = pyo.ConcreteModel()
    Wines = df.wine.values.tolist()
    Types = df.type.values.tolist()
    Regions = df.region.values.tolist()
    Prices = df.price.values.tolist()
    Ratings = df.rating.values.tolist()
    

    # sets --------------------
    model.sWines = pyo.Set(initialize = Wines) # here come all wines
    model.sTypes = pyo.Set(initialize = list(set(Types))) # type of wine "red", "white", ...
    model.sRegion = pyo.Set(initialize = list(set(Regions))) # wine region

    # variables ---------------
    model.vbWineInKnapsack = pyo.Var(model.sWines, domain=pyo.Binary)

    # parameters ---------------
    model.pWinePrice = pyo.Param(model.sWines, initialize = dict(zip(Wines, Prices)))
    model.pWineRating = pyo.Param(model.sWines, initialize= dict(zip(Wines, Ratings)))
    model.pWineType = pyo.Param(model.sWines, initialize = dict(zip(Wines, Types)))
    model.pWineRegion = pyo.Param(model.sWines, initialize = dict(zip(Wines, Regions)))

    # constraints --------------
    model.ctMaxPrice = pyo.Constraint(expr = sum(model.vbWineInKnapsack[w]*model.pWinePrice[w] for w in model.sWines)<=maxPrice) # max basket price
    model.ctMaxWines = pyo.Constraint(expr = sum(model.vbWineInKnapsack[w] for w in model.sWines)<=maxWines) # max different wines

    # objective function --------
    model.ofBasket = pyo.Objective(expr = -sum(model.vbWineInKnapsack[w]*model.pWineRating[w] for w in model.sWines))

    return model

def solve_knapsack(model, 
    solver:Dict={'name':'cbc', 'path':"../bin/cbc.exe"}, 
    verbosity:int = 0
    ):
    print(solver)
    opt = pyo.SolverFactory(solver['name'], executable = solver['path'])
    res = opt.solve(model)
    print(res) if verbosity > 0 else ''

    results = {
    'wine' : [],
    'type' :[],
    'region' : [],
    'price' : [],
    'rating' : []
   }
   
    for w in model.sWines:
        if model.vbWineInKnapsack[w].value>0:
            results['wine'].append(w)
            results['type'].append(model.pWineType[w])
            results['region'].append(model.pWineRegion[w])
            results['price'].append(model.pWinePrice[w])
            results['rating'].append(model.pWineRating[w])
        

    return pd.DataFrame(results)
