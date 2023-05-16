
PROYECTO INDIVIDUAL Nº1

Operaciones de Aprendizaje Automático (MLOps)
Proyecto de análisis de datos y desarrollo de API de películas
Este proyecto tiene como objetivo realizar un análisis exploratorio de datos de películas, desarrollar una API para consultar información sobre películas y entrenar un modelo de recomendación de películas basado en similitud de puntuaciones.

Datos utilizados

Los datos se trabajaron en Visual Studio. Se utilizó para obtener los datos y limpiarlos, realizando las transformaciones necesarias para su análisis. Los datos fueron proporcionados en formato CSV y consistían en información sobre películas, incluyendo su título, año de estreno, presupuesto, ingresos y otros detalles relevantes.

Durante el proceso de análisis, se llevaron a cabo las siguientes transformaciones en los datos:

Se descompusieron las columnas belongs_to_collection, production_companies, genres, production_countries y spoken_lenguages que estaban anidadas.

Los valores nulos de los campos revenue y budget se reemplazaron por 0.

Se eliminaron los valores nulos del campo release date.

Las fechas en el campo release date se transformaron al formato AAAA-mm-dd y se creó la columna release_year para extraer el año de la fecha de estreno.

Se creó la columna return para calcular el retorno de inversión.

Se eliminaron las columnas innecesarias para el análisis, como video, imdb_id, adult, original_title, vote_count, poster_path, homepage y otras columnas extras.

Una vez que los datos fueron limpiados, se guardaron en el archivo "movies_data.csv" para comenzar a trabajar en los endpoints. Además, el proceso de transformación se documentó en el archivo "Transformaciones.ipynb".

Desarrollo de la API
Se utilizó el framework FastAPI para desarrollar una API con 6 endpoints que permiten consultar información sobre películas. Los endpoints son los siguientes:

/peliculas_mes(mes): devuelve la cantidad de películas estrenadas en un mes determinado.
/peliculas_dia(dia): devuelve la cantidad de películas estrenadas en un día de la semana determinado.
/franquicia(franquicia): devuelve la cantidad de películas, ganancia total y ganancia promedio de una franquicia determinada.
/peliculas_pais(pais): devuelve la cantidad de películas producidas en un país determinado.
/productoras(productora): devuelve la ganancia total y la cantidad de películas producidas por una productora determinada.
/retorno(pelicula): devuelve la inversión, ganancia, retorno y año de estreno de una película determinada.

También se desarrolló una función adicional /recomendacion(titulo) que utiliza el modelo de recomendación entrenado para sugerir películas similares a una película determinada.

Análisis exploratorio de datos
Se realizó un análisis exploratorio de datos utilizando diversas librerías de Python como pandas, seaborn y matplotlib. Se investigaron las relaciones entre las variables del dataset, se identificaron outliers y se seleccionaron las variables a utilizar en el modelo de machine learning. Los resultados de este análisis se presentan en el archivo "EDA.ipynb", donde se encuentran los gráficos y tablas generados. Los cambios se guardaron en el archivo "movies_dataML.csv".

Modelo de recomendación
Se entrenó un modelo de recomendación basado en similitud de puntuaciones utilizando la librería