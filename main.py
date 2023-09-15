import pandas as pd
import joblib
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse, HTMLResponse
import uvicorn

app = FastAPI()

# Cargar el DataFrame con los datos
df = pd.read_csv('inmigracion_canada_prediccion.csv')

# Cargar el modelo previamente entrenado
model = joblib.load('modelo_inmigracion.pkl')

# Cargar el archivo de codificación para las provincias
label_encoder = joblib.load('label_encoder.pkl')

# Inicializar las plantillas
templates = Jinja2Templates(directory="templates")

# Página principal
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/")
def predict_inmigrantes(
    selected_year: int = Form(...),
    selected_province: str = Form(...)
):
    def predict_inmigrants(year, localidad_encoded):
        # Crear un DataFrame con las características ingresadas
        data = pd.DataFrame({
            "Year": [year],
            "localidad_encoded": [localidad_encoded],
        })

        # Realizar la predicción utilizando el modelo
        prediction = model.predict(data)

        return prediction[0]

    # Verificar si selected_province es un valor numérico (codificado) o una cadena de texto (provincia escrita)
    if selected_province.isdigit():
        # Si es un valor numérico, asumimos que es una provincia codificada
        selected_province_encoded = int(selected_province)
    else:
        # Si es una cadena de texto, intentamos codificarla usando el label_encoder
        try:
            selected_province_encoded = label_encoder.transform([selected_province])[0]
        except ValueError:
            return JSONResponse(content={"error": "Provincia no válida"})

    # Realizar la predicción utilizando el modelo cargado
    prediction = predict_inmigrants(selected_year, selected_province_encoded)
    
    return JSONResponse(content={f"Predicción para {selected_province} el {selected_year}": prediction})


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
