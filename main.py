from sklearn.preprocessing import MinMaxScaler
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import numpy as np
from typing import List
import json
import calendar
from fastapi import FastAPI
from datetime import datetime
import pandas as pd

app = FastAPI()

# Leer el archivo csv
df = pd.read_csv("movies_dataset_final1.csv", parse_dates=["release_date"])


@app.get("/peliculas_mes")
async def peliculas_mes(mes: str):
    
    datetime_obj = datetime.strptime(mes, "%m") #Convertimos a datatime 
    
    filtered_movies = df[df["release_date"].dt.month == datetime_obj.month]#filtramos peliculas de ese mes 
    
    cantidad = len(filtered_movies)  # Contamos la cantidad de películas
    return {"mes": datetime_obj.strftime("%B"), "cantidad": cantidad}


@app.get("/peliculas_dia")
async def peliculas_dia(dia: int):
    
    dia_semana = calendar.day_name[dia] # Convertir el número de día a un nombre de día de la semana

    filtered_movies = df[df["release_date"].dt.day_name() == dia_semana]#filtramos peliculas por dia de estreno
   
    cantidad = len(filtered_movies)  # Contamos la cantidad de películas
    
    return {"dia": dia_semana, "cantidad": cantidad}




@app.get("/peliculas_pais/{pais}")
async def peliculas_pais(pais: str):
    
    peliculas = df[df['production_countries_name'].str.contains(
        pais, na=False)]  # Filtramos las películas por país

    
    cantidad = len(peliculas)

    
    respuesta = {"pais": pais, "cantidad": cantidad}

    
    return respuesta



@app.get("/productoras/{productora}")
async def productoras(productora: str):
    
    df_productora = df[df['production_companies_name'] == productora]# Filtramos columna productora

    ganancia_total = df_productora['revenue'].sum()  # Calculamos la ganancia total y la cantidad de películas
    cantidad = len(df_productora)

    return {'productora': productora, 'ganancia_total': ganancia_total, 'cantidad': cantidad}


@app.get('/retorno/{pelicula}')
async def retorno(pelicula: str):
   

    filtro_titulo = df[df['title'] == pelicula]

    filtr_inversion = filtro_titulo["budget"].values[0].item()
    filtro_ganancia = filtro_titulo["revenue"].values[0].item()
    filtro_retorno = filtro_titulo["return"].values[0].item()
    filtro_año = filtro_titulo["release_date"].values[0].astype(
        'M8[D]').astype(str)

    return {"pelicula": pelicula, "inversion": filtr_inversion, "ganancia": filtro_ganancia, "retorno": filtro_retorno, "fecha de lanzamiento": filtro_año}



