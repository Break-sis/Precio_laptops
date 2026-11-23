# Predicción del precio de laptops usadas

Proyecto final de Álgebra Lineal: regresión lineal múltiple con la ecuación normal
**β = (XᵀX)⁻¹XᵀY**, con interfaz web interactiva (NiceGUI).

## Cómo ejecutarlo (VS Code, Windows)

1. Abre la carpeta `Proyecto_final` en VS Code (`Archivo > Abrir carpeta`).
2. Abre la terminal (`Ctrl + ñ`) y crea el entorno virtual:
   `python -m venv venv`  y luego  `venv\Scripts\activate`
3. Instala las librerías: `pip install -r requirements.txt`
4. Ejecuta: `python main.py`  (se abre el navegador en http://localhost:8080)
5. Para cerrar el programa, presiona `Ctrl + C` en la terminal.

Si `python` no funciona, usa `py` en su lugar.

## Orden de uso en la interfaz

1. **Datos** → "Cargar datos de ejemplo" (o sube tu propio CSV).
2. **Modelo** → "Entrenar modelo".
3. **Predicción** → elige las características y presiona "Predecir precio".
4. **Evaluación** → MAE, RMSE, R² y gráficas (botón para guardar el PNG).
5. **Álgebra Lineal** → matrices X, Xᵀ, Y y β con tus datos reales.

## Archivos

| Archivo | Función |
|---|---|
| `main.py` | Interfaz web (pestañas, botones, gráficos) |
| `modelo.py` | Carga del CSV, ecuación normal, predicción y métricas |
| `graficos.py` | Genera `graficas/resultados.png` con Matplotlib |
| `generar_datos.py` | Crea el CSV con datos ficticios (¡sobrescribe el CSV!) |
| `test_modelo.py` | 12 pruebas automáticas: `python -m unittest -v` |
| `datos/laptops_usadas.csv` | 150 laptops ficticias |
| `assets/logo.png` | Logo mostrado en la interfaz |

## Formato del CSV

Primera línea exacta: `ram_gb,almacenamiento_gb,anios_uso,bateria_pct,precio_bs`
(mínimo 10 filas, solo números, punto decimal).
