import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
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
    # Convertir el mes a un objeto datetime
    datetime_obj = datetime.strptime(mes, "%m")
    # Obtener las películas que se estrenaron ese mes
    filtered_movies = df[df["release_date"].dt.month == datetime_obj.month]
    # Contar la cantidad de películas
    cantidad = len(filtered_movies)
    return {"mes": datetime_obj.strftime("%B"), "cantidad": cantidad}


@app.get("/peliculas_dia")
async def peliculas_dia(dia: int):
    # Convertir el número de día a un nombre de día de la semana
    dia_semana = calendar.day_name[dia]
    # Obtener las películas que se estrenaron ese día de la semana
    filtered_movies = df[df["release_date"].dt.day_name() == dia_semana]
    # Contar la cantidad de películas
    cantidad = len(filtered_movies)
    # Retornar el resultado en un diccionario
    return {"dia": dia_semana, "cantidad": cantidad}




@app.get("/peliculas_pais/{pais}")
async def peliculas_pais(pais: str):
    # Filtrar las películas por país
    peliculas = df[df['production_countries_name'].str.contains(
        pais, na=False)]

    # Contar el número de películas por país
    cantidad = len(peliculas)

    # Crear el diccionario de respuesta
    respuesta = {"pais": pais, "cantidad": cantidad}

    # Devolver la respuesta
    return respuesta



# Definir la ruta para la función productoras


@app.get("/productoras/{productora}")
async def productoras(productora: str):
    # Filtrar las filas que corresponden a la productora
    df_productora = df[df['production_companies_name'] == productora]

    # Calcular la ganancia total y la cantidad de películas
    ganancia_total = df_productora['revenue'].sum()
    cantidad = len(df_productora)

    # Retornar el resultado como un diccionario
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
    # Filtrar las filas que pertenecen a la franquicia especificada
    franquicia_df = df[df['collection_name'] == franquicia]

    # Contar la cantidad de películas de la franquicia
    cantidad = franquicia_df['title'].nunique()

    # Calcular la ganancia total de la franquicia
    ganancia_total = franquicia_df['revenue'].sum()

    # Calcular la ganancia promedio por película de la franquicia
    ganancia_promedio = ganancia_total / cantidad

    # Convertir el objeto JSON a una cadena compatible con JSON
    response_data = {"franquicia": franquicia, "cantidad": cantidad,
                     "ganancia_total": ganancia_total, "ganancia_promedio": ganancia_promedio}
    response_str = json.dumps(response_data)

    # Devolver la cadena como respuesta de la API
    return response_str


app = FastAPI()

# Cargar dataset
df = pd.read_csv('movies_dataset_final1.csv')

# Seleccionar solo las columnas necesarias
movies_subset = df[['production_companies_name', 'title', 'genre_name']]

# Eliminar filas con valores nulos
movies_subset = movies_subset.dropna()

# Limitar a 18000 filas
movies_subset = movies_subset.head(18000)

# Convertir valores categóricos en variables dummies
movies_subset = pd.get_dummies(movies_subset, columns=[
                               'production_companies_name', 'genre_name'])

# Entrenar modelo de vecinos cercanos
model = NearestNeighbors(n_neighbors=6, metric='cosine', algorithm='brute')
model.fit(movies_subset.drop('title', axis=1))

# Normalizar los datos
scaler = StandardScaler()
movies_norm = scaler.fit_transform(movies_subset.drop('title', axis=1))

# Guardar modelo y scaler entrenados
joblib.dump(model, 'model.joblib')
joblib.dump(scaler, 'scaler.joblib')

# Función de recomendación


@app.get("/recomendacion")
async def recomendacion(titulo: str):
    # Cargar modelo y scaler entrenados
    model = joblib.load('model.joblib')
    scaler = joblib.load('scaler.joblib')

    # Crear vector de características del título proporcionado
    title_features = movies_subset[movies_subset['title'] == titulo].drop(
        'title', axis=1)
    title_features = scaler.transform(title_features)

    # Obtener índices de las películas similares
    distances, indices = model.kneighbors(title_features, n_neighbors=6)

    # Obtener los títulos de las películas similares
    titles = []
    for i in range(1, len(distances.flatten())):
        titles.append(
            df[df.index == indices.flatten()[i]]['title'].values[0])

    return {'lista recomendada': titles}








