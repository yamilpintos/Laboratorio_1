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
from typing import Dict

import pandas as pd
import ast


df = pd.read_csv('movies_dataset.csv')


df_belongs_to_collection = df[df['belongs_to_collection'].notnull()]
df_belongs_to_collection['belongs_to_collection'] = df_belongs_to_collection['belongs_to_collection'].apply(
    lambda x: x.replace("'s", "'"))
df_belongs_to_collection['belongs_to_collection'] = df_belongs_to_collection['belongs_to_collection'].apply(
    lambda x: ast.literal_eval(x))
df_belongs_to_collection['belongs_to_collection'] = df_belongs_to_collection['belongs_to_collection'].apply(
    lambda x: {k: v for k, v in x.items() if k != 'id'} if x is not None else None)


dict_to_remove = {"id": 10455, "name": "Child's Play Collection",
                  "poster_path": "/50aqbDvbOtdlZrje6Qk4ZvKM7dM.jpg", "backdrop_path": "/AAhYXBVIEl6WgQnzfBsauTIC25.jpg"}
df_belongs_to_collection = df_belongs_to_collection[
    df_belongs_to_collection['belongs_to_collection'] != dict_to_remove]

df_belongs_to_collection.to_csv('movies_unacolumna.csv', index=False)


df = pd.read_csv('movies_dataset.csv')


for i, row in df.iterrows(): 
    belongs_to_collection = row['belongs_to_collection']
    if isinstance(belongs_to_collection, str): 
        try:
            dict_data = ast.literal_eval(belongs_to_collection)
            if isinstance(dict_data, dict):  
                df.loc[i, 'collection_id'] = dict_data.get('id', None)
                df.loc[i, 'collection_name'] = dict_data.get('name', None)
                df.loc[i, 'collection_poster_path'] = dict_data.get(
                    'poster_path', None)
        except (ValueError, SyntaxError):  
            pass

        import ast
import pandas as pd

df = pd.read_csv('movies_unalistadediccionarios.csv')

df['genre_name'] = ''  
df['genre_id'] = ''

for i, row in df.iterrows():  
    my_list = row['genres']
    if isinstance(my_list, str):  
        try:
            list_data = ast.literal_eval(my_list)
            if isinstance(list_data, list):  
                for dict_data in list_data:
                    
                    if isinstance(dict_data, dict):

                        if 'name' in dict_data:  
                            df.at[i, 'genre_name'] = dict_data['name']
                        if 'id' in dict_data:
                            df.at[i, 'genre_id'] = dict_data['id']

        except (ValueError, SyntaxError):  
            pass


print(df)


df.to_csv('movies_unalistadediccionarios.csv',
          index=False)


print(df)


import ast
import pandas as pd

df = pd.read_csv('movies_unalistadediccionarios.csv')



df['production_companies_name'] = ''
df['production_companies_id'] = ''


for i, row in df.iterrows():  
    my_list = row['production_companies']
    if isinstance(my_list, str):  
        try:
            list_data = ast.literal_eval(my_list)
            if isinstance(list_data, list):  
                for dict_data in list_data:
                    if isinstance(dict_data, dict):  

                        if 'name' in dict_data:
                           
                            df.at[i, 'production_companies_name'] = dict_data['name']
                        if 'id' in dict_data:
                            df.at[i, 'production_companies_id'] = dict_data['id']

        except (ValueError, SyntaxError):  
            pass

print(df)


df.to_csv('movies_unalistadediccionarios.csv', index=False)
df.to_csv('movies_unalistadediccionarios.csv', index=False)

import ast
import pandas as pd

df = pd.read_csv('movies_unalistadediccionarios.csv')


df['production_countries_name'] = ''
df['production_countries_id'] = ''


for i, row in df.iterrows():
    my_list = row['production_countries']
    if isinstance(my_list, str):
        try:
            list_data = ast.literal_eval(my_list)
            if isinstance(list_data, list):
                for dict_data in list_data:
                    if isinstance(dict_data, dict):
                        if 'name' in dict_data:
                            df.at[i, 'production_countries_name'] = dict_data['name']
                        if 'id' in dict_data:
                            df.at[i, 'production_countries_id'] = dict_data['id']

        except (ValueError, SyntaxError):
            pass


print(df)

df.to_csv('movies_unalistadediccionarios.csv', index=False)
import ast
import pandas as pd

df = pd.read_csv('movies_unalistadediccionarios.csv')


df['spoken_languages_name'] = ''
df['spoken_languages_iso'] = ''


for i, row in df.iterrows():
    my_list = row['spoken_languages']
    if isinstance(my_list, str):
        try:
            list_data = ast.literal_eval(my_list)
            if isinstance(list_data, list):
                for dict_data in list_data:
                    if isinstance(dict_data, dict):

                        if 'name' in dict_data:
                            df.at[i, 'spoken_languages_name'] = dict_data['name']
                        if 'iso_639_1' in dict_data:
                            df.at[i, 'spoken_languages_iso'] = dict_data['iso_639_1']

        except (ValueError, SyntaxError):
            pass


print(df)

df.to_csv('movies_unalistadediccionarios.csv', index=False)


import pandas as pd


df = pd.read_csv('movies_dataset_final.csv')


df['revenue'].fillna(0, inplace=True)
df['budget'].fillna(0, inplace=True)


df.dropna(subset=['release_date'], inplace=True)

df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce') 
df['release_year'] = df['release_date'].dt.year.astype('Int64') 

 
df['return'] = df['revenue'] / df['budget']  


df.drop(columns=['video', 'imdb_id', 'adult', 'original_title',
        'vote_count', 'poster_path', 'homepage'], inplace=True)  


df.to_csv('movies_dataset_final1.csv', index=False)
app = FastAPI()


df = pd.read_csv("movies_dataset_final1.csv", parse_dates=["release_date"])


def peliculas_mes(mes: str) -> int:
    meses: Dict[str, int] = {'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4, 'mayo': 5, 'junio': 6,
                             'julio': 7, 'agosto': 8, 'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12}

    mes_limpio = mes.lower()
    if mes_limpio not in meses:
        print(f"Error: mes {mes} inválido")
        return 0

    mes_numero = meses[mes_limpio]
    mes_numero = meses[mes.lower()]
    peliculas = [p for p in df['release_date']
                 if isinstance(p, str) and datetime.strptime(p, '%Y-%m-%d').month == mes_numero]
    return len(peliculas)


@app.get("/peliculas_dia")
async def peliculas_dia(dia: int):
    
    dia_semana = calendar.day_name[dia] 

    filtro_movies = df[df["release_date"].dt.day_name() == dia_semana]
    cantidad = len(filtro_movies)  
    
    return {"dia": dia_semana, "cantidad": cantidad}




@app.get("/peliculas_pais/{pais}")
async def peliculas_pais(pais: str):
    
    peliculas = df[df['production_countries_name'].str.contains(
        pais, na=False)]  

    
    cantidad = len(peliculas)

    
    respuesta = {"pais": pais, "cantidad": cantidad}

    
    return respuesta



@app.get("/productoras/{productora}")
async def productoras(productora: str):
    
    df_productora = df[df['production_companies_name'] == productora]
    ganancia_total = df_productora['revenue'].sum()  
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

    franquicia_df = df[df['collection_name'] == franquicia]  

    cantidad = franquicia_df['title'].nunique()  

    ganancia_total = franquicia_df['revenue'].sum()  

    ganancia_promedio = ganancia_total / cantidad

    respuesta_data = {"franquicia": franquicia, "cantidad": cantidad,
                     "ganancia_total": ganancia_total, "ganancia_promedio": ganancia_promedio}
    
    response_str = json.dumps(respuesta_data) 

    return response_str

movies_1 = df[['production_companies_name', 'title', 'genre_name']]

movies_1 = movies_1.dropna()


movies_1 = movies_1.head(5000)


movies_1 = pd.get_dummies(movies_1, columns=[
                               'production_companies_name', 'genre_name'])


model = NearestNeighbors(n_neighbors=6, metric='cosine', algorithm='brute')

model.fit(movies_1.drop('title', axis=1))


scaler = StandardScaler()
movies_norm = scaler.fit_transform(
    movies_1.drop('title', axis=1)) 


joblib.dump(model, 'model.joblib')
joblib.dump(scaler, 'scaler.joblib')  


@app.get("/recomendacion")
async def recomendacion(titulo: str):

    model = joblib.load('model.joblib')
    scaler = joblib.load('scaler.joblib')  

    title_features = movies_1[movies_1['title'] == titulo].drop(
        'title', axis=1)
    title_features = scaler.transform(title_features)

    distances, indices = model.kneighbors(title_features, n_neighbors=6)

    titles = []
    for i in range(1, len(distances.flatten())):
        titles.append(
            df[df.index == indices.flatten()[i]]['title'].values[0])

    return {'lista recomendada': titles}

