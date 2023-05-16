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





# Cargar el conjunto de datos
movies = pd.read_csv('movies_dataset_final1.csv')

# Seleccionar las columnas relevantes para el modelo
movies_subset = movies[['original_language', 'collection_name', 'genre_name', 'production_companies_name', 'production_countries_name']]

# Eliminar filas con valores faltantes
movies_subset = movies_subset.dropna()

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    movies_subset['title'], movies_subset[['original_language', 'collection_name', 'genre_name', 'production_companies_name', 'production_countries_name']], test_size=0.2, random_state=42)

# Crear un vectorizador para convertir los títulos en características numéricas
vectorizer = CountVectorizer()

# Convertir los títulos en características numéricas
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Crear el modelo de árbol de decisión
clf = DecisionTreeClassifier()

# Entrenar el modelo con los datos de entrenamiento
clf.fit(X_train_vec, y_train)

# Predecir los géneros de las películas en el conjunto de prueba
y_pred = clf.predict(X_test_vec)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print("La precisión del modelo es:", accuracy)


@app.get("/recomendacion/{titulo}", tags=["Recomendación de películas"])
async def recomendacion(titulo: str) -> List[str]:
    # Busca el índice de la película con el título dado
    movie_index = movies[movies['title'] == titulo].index.values[0]

    # Realiza la predicción
    X = vectorizer.transform([titulo])
    y_pred = clf.predict(X)[0]

    # Filtra las películas por género y por la predicción del modelo
    peliculas_recomendadas = movies[(
        movies['genre_name'] == y_pred) & (movies['title'] != titulo)]
    X = vectorizer.transform(peliculas_recomendadas['title'])
    y_pred = clf.predict(X)
    peliculas_recomendadas['predicciones'] = y_pred
    peliculas_recomendadas = peliculas_recomendadas.sort_values(
        'predicciones', ascending=False).head(10)['title'].tolist()

    return peliculas_recomendadas



