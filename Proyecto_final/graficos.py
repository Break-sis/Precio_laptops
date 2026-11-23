"""Genera la figura de resultados en PNG con Matplotlib."""
import matplotlib

matplotlib.use("Agg")  # sin ventana: el servidor web solo guarda el archivo
import matplotlib.pyplot as plt
import numpy as np

import modelo as m

AZUL, ROJO, VERDE = "#004CAA", "#D9480F", "#2E8B57"


def guardar_figura(modelo, train, test, ruta):
    y = test[m.COLUMNA_Y].to_numpy()
    yp = modelo.predecir(test[m.COLUMNAS_X])
    err = y - yp
    mt = m.calcular_metricas(train[m.COLUMNA_Y], modelo.predecir(train[m.COLUMNAS_X]))
    ms = m.calcular_metricas(y, yp)

    fig, ax = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")

    a = ax[0, 0]
    a.scatter(y, yp, s=70, color=AZUL, alpha=.8, edgecolor="white")
    lim = [min(y.min(), yp.min()) - 200, max(y.max(), yp.max()) + 200]
    a.plot(lim, lim, "--", color=ROJO, lw=2.5, label="Predicción perfecta")
    a.set_title("A. Real vs. predicho (prueba)", fontweight="bold", loc="left")
    a.set_xlabel("Precio real (Bs)")
    a.set_ylabel("Precio predicho (Bs)")
    a.legend(frameon=False)

    b = ax[0, 1]
    b.hist(err, bins=10, color=AZUL, edgecolor="white")
    b.axvline(0, color=ROJO, ls="--", lw=2.5)
    b.set_title("B. Distribución de errores", fontweight="bold", loc="left")
    b.set_xlabel("Error (real − predicho) en Bs")
    b.set_ylabel("Laptops")

    c = ax[1, 0]
    etiquetas = ["RAM\n(por GB)", "Almac.\n(por 100 GB)", "Años\n(por año)", "Batería\n(por 1 %)"]
    efectos = [modelo.beta[1], modelo.beta[2] * 100, modelo.beta[3], modelo.beta[4]]
    colores = [VERDE if v >= 0 else ROJO for v in efectos]
    barras = c.bar(etiquetas, efectos, color=colores)
    for r, v in zip(barras, efectos):
        c.text(r.get_x() + r.get_width() / 2, v + (8 if v >= 0 else -22), f"{v:.0f}",
               ha="center", fontweight="bold")
    c.axhline(0, color="gray", lw=1)
    c.set_title("C. Coeficientes β estimados", fontweight="bold", loc="left")
    c.set_ylabel("Cambio en el precio (Bs)")

    d = ax[1, 1]
    x = np.arange(2)
    w = .36
    d.bar(x - w / 2, [mt["MAE"], mt["RMSE"]], w, label=f"Entrenamiento (R² = {mt['R2']:.3f})", color="#7FA8DC")
    d.bar(x + w / 2, [ms["MAE"], ms["RMSE"]], w, label=f"Prueba (R² = {ms['R2']:.3f})", color=AZUL)
    d.set_xticks(x)
    d.set_xticklabels(["MAE", "RMSE"])
    d.set_ylabel("Error (Bs)")
    d.set_title("D. Métricas del modelo", fontweight="bold", loc="left")
    d.legend(frameon=False, loc="upper left")

    for a_ in ax.ravel():
        a_.spines[["top", "right"]].set_visible(False)

    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    return ruta
