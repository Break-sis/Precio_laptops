"""Sistema de predicción del precio de laptops usadas.

Interfaz web interactiva con NiceGUI. Ejecutar con:  python main.py
"""
import io
from pathlib import Path

import numpy as np
from nicegui import app, ui

import graficos
import modelo as m

BASE = Path(__file__).resolve().parent
CSV_DEFECTO = BASE / "datos" / "laptops_usadas.csv"
CARPETA_GRAFICAS = BASE / "graficas"
AZUL, ROJO, VERDE = "#004CAA", "#D9480F", "#2E8B57"

app.add_static_files("/assets", BASE / "assets")


# =====================================================================
#  Estado de la aplicación
# =====================================================================
class Estado:
    def __init__(self):
        self.df = None
        self.origen = ""
        self.train = None
        self.test = None
        self.modelo = m.RegresionLinealMultiple()
        self.entrenado = False
        self.m_train = None
        self.m_test = None
        self.prediccion = None

    def cargar(self, origen, nombre):
        self.df = m.cargar_datos(origen)
        self.origen = nombre
        self.entrenado = False
        self.prediccion = None

    def entrenar(self):
        self.train, self.test = m.dividir_datos(self.df)
        self.modelo.entrenar(self.train[m.COLUMNAS_X], self.train[m.COLUMNA_Y])
        self.m_train = m.calcular_metricas(self.train[m.COLUMNA_Y], self.modelo.predecir(self.train[m.COLUMNAS_X]))
        self.m_test = m.calcular_metricas(self.test[m.COLUMNA_Y], self.modelo.predecir(self.test[m.COLUMNAS_X]))
        self.entrenado = True
        self.prediccion = None


estado = Estado()
entrada = {"ram": 8, "almac": 256, "anios": 3.0, "bateria": 80}


def bs(valor):
    return f"Bs {valor:,.0f}"


def signo(valor):
    return "+" if valor >= 0 else "−"


# =====================================================================
#  Acciones (botones)
# =====================================================================
def refrescar_todo():
    barra_estado.refresh()
    panel_datos.refresh()
    panel_modelo.refresh()
    panel_resultado.refresh()
    panel_evaluacion.refresh()
    panel_algebra.refresh()


def accion_cargar(origen=CSV_DEFECTO, nombre=None):
    try:
        estado.cargar(origen, nombre or Path(origen).name)
    except FileNotFoundError:
        ui.notify("No se encontró el archivo. Ejecuta primero: python generar_datos.py", type="negative")
    except Exception as e:  # CSV dañado, columnas faltantes, etc.
        ui.notify(f"Error al cargar el CSV: {e}", type="negative")
    else:
        ui.notify(f"Datos cargados: {len(estado.df)} laptops", type="positive", icon="check_circle")
    refrescar_todo()


async def accion_subir(e):
    try:
        if hasattr(e, "file"):                       # NiceGUI 3 o superior
            contenido = await e.file.read()
            nombre = getattr(e.file, "name", "archivo subido")
        else:                                        # NiceGUI 2
            contenido = e.content.read()
            nombre = e.name
        accion_cargar(io.BytesIO(contenido), nombre)
    except Exception as err:
        ui.notify(f"No se pudo leer el archivo: {err}", type="negative")


def asegurar_datos():
    if estado.df is None:
        ui.notify("Aún no habías cargado datos: se cargó el CSV de ejemplo.", type="info")
        accion_cargar()
    return estado.df is not None


def accion_entrenar():
    if not asegurar_datos():
        return
    estado.entrenar()
    ui.notify(f"Modelo entrenado con {len(estado.train)} laptops "
              f"({len(estado.test)} reservadas para prueba)", type="positive", icon="psychology")
    refrescar_todo()


def asegurar_modelo():
    if not estado.entrenado:
        if not asegurar_datos():
            return False
        ui.notify("El modelo aún no estaba entrenado: se entrenó automáticamente.", type="info")
        accion_entrenar()
    return estado.entrenado


def accion_predecir():
    if not asegurar_modelo():
        return
    x = [entrada["ram"], entrada["almac"], entrada["anios"], entrada["bateria"]]
    pred = max(float(estado.modelo.predecir(x)[0]), 0.0)
    contribuciones = estado.modelo.beta * np.array([1.0] + [float(v) for v in x])
    precios = estado.df[m.COLUMNA_Y]
    q1, q3 = precios.quantile(0.25), precios.quantile(0.75)
    gama = "baja" if pred < q1 else ("media" if pred < q3 else "alta")
    estado.prediccion = {"x": x, "precio": pred, "contrib": contribuciones.tolist(), "gama": gama}
    panel_resultado.refresh()


def accion_ejemplo(ram, almac, anios, bateria):
    entrada.update(ram=ram, almac=almac, anios=anios, bateria=bateria)
    accion_predecir()


def accion_guardar_png():
    if not asegurar_modelo():
        return
    ruta = graficos.guardar_figura(estado.modelo, estado.train, estado.test, CARPETA_GRAFICAS / "resultados.png")
    ui.notify(f"Gráfica guardada en: {ruta}", type="positive", icon="save")


# =====================================================================
#  Componentes reutilizables
# =====================================================================
def tarjeta_dato(titulo, valor, icono):
    with ui.card().classes("flex-1 items-center text-center").style("min-width: 190px"):
        ui.icon(icono, size="2rem").classes("text-primary")
        ui.label(valor).classes("text-2xl font-bold")
        ui.label(titulo).classes("text-sm text-gray-500")


def vacio(titulo, texto, icono):
    with ui.card().classes("w-full items-center text-center p-8"):
        ui.icon(icono, size="3rem").classes("text-gray-400")
        ui.label(titulo).classes("text-xl font-bold")
        ui.label(texto).classes("text-gray-500")


def grafico(opciones, alto="h-80"):
    return ui.echart(opciones).classes(f"w-full {alto}")


def opciones_barras(categorias, valores, titulo, nombre_y, colorear=True, decimales=1):
    datos = [{"value": round(float(v), decimales), "itemStyle": {"color": VERDE if v >= 0 else ROJO}} if colorear
             else round(float(v), decimales) for v in valores]
    return {
        "title": {"text": titulo, "left": "center", "textStyle": {"fontSize": 14}},
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "grid": {"left": 70, "right": 20, "top": 60, "bottom": 50},
        "xAxis": {"type": "category", "data": categorias, "axisLabel": {"interval": 0}},
        "yAxis": {"type": "value", "name": nombre_y},
        "series": [{"type": "bar", "data": datos, "label": {"show": True, "position": "top"}}],
    }


# =====================================================================
#  Paneles (se redibujan cuando cambia el estado)
# =====================================================================
@ui.refreshable
def barra_estado():
    with ui.row().classes("items-center gap-2"):
        if estado.df is None:
            ui.chip("Datos: sin cargar", icon="folder_open", color="white", text_color="grey-8")
        else:
            ui.chip(f"Datos: {len(estado.df)} laptops", icon="check_circle", color="white", text_color="green-8")
        if estado.entrenado:
            ui.chip("Modelo: entrenado", icon="psychology", color="white", text_color="green-8")
        else:
            ui.chip("Modelo: sin entrenar", icon="psychology_alt", color="white", text_color="grey-8")


@ui.refreshable
def panel_datos():
    if estado.df is None:
        vacio("Aún no hay datos cargados",
              "Presiona «Cargar datos de ejemplo» o sube tu propio archivo CSV.", "folder_open")
        return
    df = estado.df
    precios = df[m.COLUMNA_Y]
    with ui.row().classes("w-full gap-4"):
        tarjeta_dato("Laptops registradas", f"{len(df)}", "laptop_mac")
        tarjeta_dato("Precio promedio", bs(precios.mean()), "payments")
        tarjeta_dato("Precio mínimo", bs(precios.min()), "trending_down")
        tarjeta_dato("Precio máximo", bs(precios.max()), "trending_up")

    with ui.card().classes("w-full"):
        ui.label(f"Datos registrados ({estado.origen})").classes("text-lg font-bold")
        columnas = [{"name": c, "label": et, "field": c, "align": "center", "sortable": True}
                    for c, et in zip(m.COLUMNAS_X + [m.COLUMNA_Y], m.ETIQUETAS_X + [m.ETIQUETA_Y])]
        filas = [{"id": i + 1, **fila} for i, fila in enumerate(df.round(1).to_dict("records"))]
        ui.table(columns=columnas, rows=filas, row_key="id", pagination=8).classes("w-full")

    with ui.row().classes("w-full gap-4 items-stretch"):
        with ui.card().classes("flex-1").style("min-width: 340px"):
            ui.label("Análisis estadístico").classes("text-lg font-bold")
            cols = [{"name": k, "label": k, "field": k, "align": "center"}
                    for k in ("Variable", "Media", "Desv. est.", "Mínimo", "Máximo")]
            stats = [{"id": i, "Variable": et, "Media": f"{df[c].mean():,.1f}", "Desv. est.": f"{df[c].std():,.1f}",
                      "Mínimo": f"{df[c].min():,.1f}", "Máximo": f"{df[c].max():,.1f}"}
                     for i, (c, et) in enumerate(zip(m.COLUMNAS_X + [m.COLUMNA_Y], m.ETIQUETAS_X + [m.ETIQUETA_Y]))]
            ui.table(columns=cols, rows=stats, row_key="id").classes("w-full")
        with ui.card().classes("flex-1").style("min-width: 340px"):
            corr = df.corr()[m.COLUMNA_Y].drop(m.COLUMNA_Y)
            grafico(opciones_barras(["RAM", "Almac.", "Años", "Batería"], corr.values, "Correlación de cada variable con el precio", "r", decimales=3), "h-64")
            ui.label("Cerca de +1: sube junto con el precio. Cerca de −1: al subir, el precio baja.").classes("text-xs text-gray-500")


def interpretar(i, v):
    verbo = "sube" if v >= 0 else "baja"
    if i == 1:
        return f"Cada 1 GB más de RAM {verbo} el precio ≈ Bs {abs(v):,.0f}"
    if i == 2:
        return f"Cada 100 GB más de almacenamiento {verbo} el precio ≈ Bs {abs(v) * 100:,.0f}"
    if i == 3:
        return f"Cada año más de uso {verbo} el precio ≈ Bs {abs(v):,.0f}"
    return f"Cada 1 % más de batería {verbo} el precio ≈ Bs {abs(v):,.0f}"


@ui.refreshable
def panel_modelo():
    if not estado.entrenado:
        vacio("El modelo aún no está entrenado",
              "Carga los datos y presiona «Entrenar modelo».", "psychology_alt")
        return
    b = estado.modelo.beta
    with ui.row().classes("w-full gap-4"):
        tarjeta_dato("Laptops para entrenar (80 %)", f"{len(estado.train)}", "school")
        tarjeta_dato("Laptops para probar (20 %)", f"{len(estado.test)}", "fact_check")
        tarjeta_dato("Coeficientes calculados", f"{len(b)}", "functions")
    with ui.card().classes("w-full"):
        ui.label("Ecuación del modelo").classes("text-lg font-bold")
        ui.markdown(
            f"**Precio estimado = {b[0]:,.0f} {signo(b[1])} {abs(b[1]):.1f}·RAM {signo(b[2])} {abs(b[2]):.2f}·Almacenamiento "
            f"{signo(b[3])} {abs(b[3]):.1f}·Años {signo(b[4])} {abs(b[4]):.1f}·Batería**")
    with ui.row().classes("w-full gap-4 items-stretch"):
        with ui.card().classes("flex-1").style("min-width: 340px"):
            ui.label("Coeficientes e interpretación").classes("text-lg font-bold")
            cols = [{"name": "coef", "label": "Coeficiente", "field": "coef", "align": "left"},
                    {"name": "valor", "label": "Valor", "field": "valor", "align": "right"},
                    {"name": "interp", "label": "Interpretación", "field": "interp", "align": "left", "style": "white-space: normal"}]
            filas = [{"coef": m.NOMBRES_BETA[0], "valor": f"{b[0]:,.2f}",
                      "interp": "Precio base cuando todas las variables valen cero (solo referencia)"}]
            filas += [{"coef": m.NOMBRES_BETA[i], "valor": f"{b[i]:,.2f}", "interp": interpretar(i, b[i])} for i in range(1, 5)]
            ui.table(columns=cols, rows=filas, row_key="coef").classes("w-full")
        with ui.card().classes("flex-1").style("min-width: 340px"):
            grafico(opciones_barras(["RAM\n(por GB)", "Almac.\n(por 100 GB)", "Años\n(por año)", "Batería\n(por 1 %)"],
                                    [b[1], b[2] * 100, b[3], b[4]], "Cambio en el precio al aumentar cada variable (Bs)", "Bs", decimales=0), "h-72")


@ui.refreshable
def panel_resultado():
    p = estado.prediccion
    if p is None:
        vacio("Aquí aparecerá la predicción",
              "Elige las características de la laptop y presiona «Predecir precio».", "sell")
        return
    rmse = estado.m_test["RMSE"]
    with ui.card().classes("w-full items-center text-center"):
        ui.label("Precio estimado").classes("text-sm text-gray-500")
        ui.label(bs(p["precio"])).classes("text-5xl font-bold text-primary")
        ui.label(f"Margen de error típico: ± {bs(rmse)} (RMSE en datos de prueba)").classes("text-sm text-gray-600")
        ui.label(f"Rango probable: {bs(max(p['precio'] - rmse, 0))} a {bs(p['precio'] + rmse)}").classes("text-sm text-gray-600")
        icono = {"baja": "south", "media": "remove", "alta": "north"}[p["gama"]]
        ui.chip(f"Gama {p['gama']} del mercado de datos", icon=icono, color="primary", text_color="white")
    with ui.card().classes("w-full"):
        categorias = ["Base\n(b0)", "RAM", "Almac.", "Años\nde uso", "Batería"]
        grafico(opciones_barras(categorias, p["contrib"], "¿Cómo se forma el precio? (aporte de cada variable en Bs)", "Bs", decimales=0), "h-72")


@ui.refreshable
def panel_evaluacion():
    if not estado.entrenado:
        vacio("Aún no hay resultados para evaluar",
              "Entrena el modelo para ver MAE, RMSE, R² y las gráficas.", "analytics")
        return
    for titulo, mt, n in (("Datos de entrenamiento", estado.m_train, len(estado.train)),
                          ("Datos de prueba (no vistos por el modelo)", estado.m_test, len(estado.test))):
        ui.label(f"{titulo} · {n} laptops").classes("text-lg font-bold")
        with ui.row().classes("w-full gap-4"):
            tarjeta_dato("MAE · error promedio", bs(mt["MAE"]), "straighten")
            tarjeta_dato("RMSE · error cuadrático", bs(mt["RMSE"]), "show_chart")
            tarjeta_dato("R² · variación explicada", f"{mt['R2']:.4f}", "percent")

    y = estado.test[m.COLUMNA_Y].to_numpy()
    yp = estado.modelo.predecir(estado.test[m.COLUMNAS_X])
    err = y - yp
    lo, hi = float(min(y.min(), yp.min())) - 200, float(max(y.max(), yp.max())) + 200

    with ui.row().classes("w-full gap-4 items-stretch"):
        with ui.card().classes("flex-1").style("min-width: 340px"):
            grafico({
                "title": {"text": "Real vs. predicho (datos de prueba)", "left": "center", "textStyle": {"fontSize": 14}},
                "tooltip": {"trigger": "item"},
                "grid": {"left": 70, "right": 20, "top": 50, "bottom": 50},
                "xAxis": {"type": "value", "name": "Precio real (Bs)", "nameLocation": "middle", "nameGap": 30, "scale": True},
                "yAxis": {"type": "value", "name": "Precio predicho (Bs)", "scale": True},
                "series": [
                    {"type": "scatter", "symbolSize": 10, "itemStyle": {"color": AZUL},
                     "data": [[round(float(a)), round(float(b_))] for a, b_ in zip(y, yp)]},
                    {"type": "line", "showSymbol": False, "silent": True,
                     "lineStyle": {"color": ROJO, "type": "dashed"}, "data": [[lo, lo], [hi, hi]]},
                ],
            }, "h-80")
            ui.label("Los puntos cerca de la línea roja son predicciones casi perfectas.").classes("text-xs text-gray-500")
        with ui.card().classes("flex-1").style("min-width: 340px"):
            conteo, bordes = np.histogram(err, bins=10)
            centros = [f"{(bordes[i] + bordes[i + 1]) / 2:,.0f}" for i in range(len(conteo))]
            grafico({
                "title": {"text": "Distribución de errores (real − predicho)", "left": "center", "textStyle": {"fontSize": 14}},
                "tooltip": {"trigger": "axis"},
                "grid": {"left": 60, "right": 20, "top": 50, "bottom": 50},
                "xAxis": {"type": "category", "data": centros, "name": "Error (Bs)", "nameLocation": "middle", "nameGap": 30},
                "yAxis": {"type": "value", "name": "Laptops"},
                "series": [{"type": "bar", "data": [int(c) for c in conteo], "itemStyle": {"color": AZUL}}],
            }, "h-80")
            ui.label("Si el centro está cerca de 0, el modelo no se equivoca siempre hacia el mismo lado.").classes("text-xs text-gray-500")

    # verificación con Scikit-learn
    try:
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression().fit(estado.train[m.COLUMNAS_X], estado.train[m.COLUMNA_Y])
        sk = np.concatenate([[lr.intercept_], lr.coef_])
        dif = float(np.max(np.abs(estado.modelo.beta - sk)))
        with ui.card().classes("w-full"):
            ui.label("Verificación con Scikit-learn").classes("text-lg font-bold")
            ui.label(f"Diferencia máxima entre los coeficientes calculados con la ecuación normal y los de la librería: {dif:.2e}. "
                     "Un valor casi cero confirma que el cálculo manual es correcto.").classes("text-gray-600")
    except ImportError:
        pass


@ui.refreshable
def panel_algebra():
    ui.markdown(
        "### Ecuación normal\n"
        "**β = (X<sup>T</sup>X)<sup>−1</sup>X<sup>T</sup>Y**\n\n"
        "- **X** (n × 5): matriz de datos. Cada fila es una laptop; la primera columna son unos (intercepto) y las otras cuatro "
        "son RAM, almacenamiento, años de uso y batería.\n"
        "- **X<sup>T</sup>** (5 × n): matriz transpuesta de X (filas y columnas intercambiadas).\n"
        "- **Y** (n × 1): vector de resultados, los precios reales.\n"
        "- **β** (5 × 1): matriz de coeficientes [b0, b1, b2, b3, b4] que se desea calcular.\n"
        "- **X<sup>T</sup>X** (5 × 5) se invierte y se multiplica por **X<sup>T</sup>Y** para obtener β.\n"
        "- Predicción para una laptop nueva: **Ŷ = Xβ**.").classes("w-full")
    if not estado.entrenado:
        ui.label("Entrena el modelo para ver las matrices reales con los datos cargados.").classes("text-gray-500")
        return
    mod = estado.modelo
    with ui.card().classes("w-full"):
        ui.label("Dimensiones reales con tus datos").classes("text-lg font-bold")
        ui.label(f"X: {mod.X.shape[0]} × {mod.X.shape[1]}   ·   Xᵀ: {mod.Xt.shape[0]} × {mod.Xt.shape[1]}   ·   "
                 f"Y: {mod.Y.shape[0]} × 1   ·   β: {mod.beta.shape[0]} × 1")
        ui.label("Primeras 5 filas de X (con la columna de unos)").classes("font-bold mt-2")
        cols = [{"name": f"c{i}", "label": et, "field": f"c{i}", "align": "center"}
                for i, et in enumerate(["1"] + m.ETIQUETAS_X)]
        filas = [{"id": i, **{f"c{j}": f"{v:,.1f}".rstrip("0").rstrip(".") for j, v in enumerate(fila)}}
                 for i, fila in enumerate(mod.X[:5])]
        ui.table(columns=cols, rows=filas, row_key="id").classes("w-full")
        ui.label("Vector β calculado").classes("font-bold mt-2")
        ui.label("   ·   ".join(f"b{i} = {v:,.3f}" for i, v in enumerate(mod.beta)))


# =====================================================================
#  Página principal
# =====================================================================
@ui.page("/")
def pagina():
    ui.colors(primary=AZUL)

    with ui.header(elevated=True).classes("bg-primary items-center justify-between px-6 py-3"):
        with ui.row().classes("items-center gap-4"):
            with ui.element("div").classes("bg-white rounded-full p-1"):
                ui.image("/assets/logo.png").classes("w-12 h-12")
            with ui.column().classes("gap-0"):
                ui.label("Predicción del precio de laptops usadas").classes("text-xl font-bold")
                ui.label("Regresión lineal múltiple · Álgebra Lineal · Universidad Adventista de Bolivia").classes("text-xs opacity-80")
        barra_estado()

    with ui.column().classes("w-full max-w-6xl mx-auto p-4 gap-4"):
        with ui.tabs().classes("w-full") as tabs:
            t_datos = ui.tab("Datos", icon="table_chart")
            t_modelo = ui.tab("Modelo", icon="psychology")
            t_pred = ui.tab("Predicción", icon="sell")
            t_eval = ui.tab("Evaluación", icon="analytics")
            t_alg = ui.tab("Álgebra Lineal", icon="functions")

        with ui.tab_panels(tabs, value=t_datos).classes("w-full bg-transparent"):
            with ui.tab_panel(t_datos).classes("gap-4 p-0"):
                with ui.row().classes("items-center gap-4"):
                    ui.button("Cargar datos de ejemplo", icon="folder_open", on_click=lambda: accion_cargar()).props("size=lg")
                    cargador = ui.upload(on_upload=accion_subir, auto_upload=True).props("accept=.csv").classes("hidden")
                    ui.button("Subir mi propio CSV", icon="upload_file",
                              on_click=lambda: cargador.run_method("pickFiles")).props("outline size=lg")
                panel_datos()

            with ui.tab_panel(t_modelo).classes("gap-4 p-0"):
                ui.button("Entrenar modelo", icon="psychology", on_click=accion_entrenar).props("size=lg")
                panel_modelo()

            with ui.tab_panel(t_pred).classes("gap-4 p-0"):
                with ui.row().classes("w-full gap-4 items-start no-wrap"):
                    with ui.card().classes("flex-1").style("min-width: 320px"):
                        ui.label("Características de la laptop").classes("text-lg font-bold")
                        ui.select([4, 8, 16, 32], label="Memoria RAM (GB)").bind_value(entrada, "ram").classes("w-full")
                        ui.select([128, 256, 512, 1024], label="Almacenamiento (GB)").bind_value(entrada, "almac").classes("w-full")
                        ui.label().bind_text_from(entrada, "anios", lambda v: f"Años de uso: {v:.1f}").classes("mt-2")
                        ui.slider(min=0.5, max=6, step=0.5).bind_value(entrada, "anios").props("label")
                        ui.label().bind_text_from(entrada, "bateria", lambda v: f"Estado de la batería: {v:.0f} %")
                        ui.slider(min=40, max=100, step=1).bind_value(entrada, "bateria").props("label")
                        ui.button("Predecir precio", icon="sell", on_click=accion_predecir).props("size=lg").classes("w-full mt-2")
                        ui.separator()
                        ui.label("Ejemplos rápidos").classes("text-sm text-gray-500")
                        with ui.row().classes("gap-2"):
                            ui.button("Básica", on_click=lambda: accion_ejemplo(4, 128, 5.0, 55)).props("outline dense no-caps")
                            ui.button("Media", on_click=lambda: accion_ejemplo(8, 256, 3.0, 80)).props("outline dense no-caps")
                            ui.button("Potente", on_click=lambda: accion_ejemplo(16, 512, 1.5, 95)).props("outline dense no-caps")
                    with ui.column().classes("flex-1 gap-4").style("min-width: 340px"):
                        panel_resultado()

            with ui.tab_panel(t_eval).classes("gap-4 p-0"):
                ui.button("Guardar gráfica en PNG", icon="save", on_click=accion_guardar_png).props("outline")
                panel_evaluacion()

            with ui.tab_panel(t_alg).classes("gap-4 p-0"):
                panel_algebra()


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(title="Precio de laptops usadas", favicon=BASE / "assets" / "logo.png", reload=False, show=True, language="es")
