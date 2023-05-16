from sklearn.preprocessing import OneHotEncoder
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
    meses: Dict[str, int] = {'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4, 'mayo': 5, 'junio': 6,
                             'julio': 7, 'agosto': 8, 'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12}
    
    datetime_objeto = datetime.strptime(mes, "%m") #Convertimos a datatime 
    
    filtered_movies = df[df["release_date"].dt.month == datetime_objeto.month]#filtramos peliculas de ese mes 
    
    cantidad = len(filtered_movies)  # Contamos la cantidad de películas
    return {"mes": datetime_objeto.strftime("%B"), "cantidad": cantidad}


@app.get("/peliculas_dia")
async def peliculas_dia(dia: int):
    
    dia_semana = calendar.day_name[dia] # Convertir el número de día a un nombre de día de la semana

    filtro_movies = df[df["release_date"].dt.day_name() == dia_semana]#filtramos peliculas por dia de estreno
   
    cantidad = len(filtro_movies)  # Contamos la cantidad de películas
    
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


@app.get("/Franquisia")
async def franquicia(franquicia):

    franquicia_df = df[df['collection_name'] == franquicia]  # Filtramos las filas que pertenecen a la franquicia.

    cantidad = franquicia_df['title'].nunique()  # Contar la cantidad de películas de la franquicia

    ganancia_total = franquicia_df['revenue'].sum()  # Calcular la ganancia total de la franquicia

    ganancia_promedio = ganancia_total / cantidad

    respuesta_data = {"franquicia": franquicia, "cantidad": cantidad,
                     "ganancia_total": ganancia_total, "ganancia_promedio": ganancia_promedio}
    
    response_str = json.dumps(respuesta_data) # Convertir el objeto JSON a una cadena compatible con JSON

    return response_str

movies_1 = df[['production_companies_name', 'title', 'genre_name']]

movies_1 = movies_1.dropna()


movies_1 = movies_1.head(5000)


movies_1 = pd.get_dummies(movies_1, columns=[
                               'production_companies_name', 'genre_name'])


model = NearestNeighbors(n_neighbors=6, metric='cosine', algorithm='brute')

model.fit(movies_1.drop('title', axis=1))# Entrenar modelo de vecinos cercanos


scaler = StandardScaler()
movies_norm = scaler.fit_transform(
    movies_1.drop('title', axis=1))  # Normalizar los datos


joblib.dump(model, 'model.joblib')
joblib.dump(scaler, 'scaler.joblib')  # Guardar modelo y scaler entrenados


@app.get("/recomendacion")
async def recomendacion(titulo: str):

    model = joblib.load('model.joblib')
    scaler = joblib.load('scaler.joblib')  # Cargar modelo y scaler entrenados

    title_features = movies_1[movies_1['title'] == titulo].drop(
        'title', axis=1)
    title_features = scaler.transform(title_features)

    distances, indices = model.kneighbors(title_features, n_neighbors=6)

    titles = []
    for i in range(1, len(distances.flatten())):
        titles.append(
            df[df.index == indices.flatten()[i]]['title'].values[0])

    return {'lista recomendada': titles}

