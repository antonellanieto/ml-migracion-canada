from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import random

df = pd.read_csv('inmigracion_canada_prediccion.csv')


label_encoder = LabelEncoder()
df['localidad_encoded'] = label_encoder.fit_transform(df['Localidad'])

joblib.dump(label_encoder, 'label_encoder.pkl')


#Comienzo de el modelo

# Defino X features y Y features
X = df.drop(['Inmigrantes' , 'Localidad'], axis=1)
y = df['Inmigrantes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

r2 = r2_score(y_test, y_pred)

model_filename = 'modelo_inmigracion.pkl'

joblib.dump(model, model_filename)

# print(df['localidad_encoded'].unique())

# Lista de localidades codificadas (reemplaza esto con tus códigos reales)
# localidades_codificadas = [2, 5, 10, 7, 4, 11, 9, 3, 12 ,0, 1, 13, 6, 8]

# # Elegir una localidad aleatoria de la lista
# localidad_codificada = random.choice(localidades_codificadas)

# # Año aleatorio (reemplaza esto con tus años reales)
# anio_aleatorio = random.randint(2023, 2030)

# # Crear un DataFrame con el año y la localidad codificada
# data_to_predict = pd.DataFrame({'Year': [anio_aleatorio], 'localidad_encoded': [localidad_codificada]})

# # Realizar la predicción
# prediccion = model.predict(data_to_predict)

# # Decodificar la localidad si es necesario
# localidad_decodificada = label_encoder.inverse_transform([localidad_codificada])[0]

# # Imprimir la predicción
# print(f'Predicción para la localidad {localidad_decodificada} en el año {anio_aleatorio}: {prediccion[0]} inmigrantes')