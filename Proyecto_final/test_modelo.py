"""Pruebas automáticas del modelo. Ejecutar con: python -m unittest -v"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import graficos
import modelo as m
from generar_datos import generar

BASE = Path(__file__).resolve().parent
CSV = BASE / "datos" / "laptops_usadas.csv"


class PruebasModelo(unittest.TestCase):
    def test_recupera_coeficientes_conocidos(self):
        """Con datos sin ruido, beta debe ser exactamente el real."""
        rng = np.random.default_rng(0)
        X = rng.uniform(1, 100, (50, 4))
        beta_real = np.array([2000.0, 130.0, 2.2, -300.0, 14.0])
        y = m.construir_matriz_X(X) @ beta_real
        modelo = m.RegresionLinealMultiple()
        modelo.entrenar(X, y)
        np.testing.assert_allclose(modelo.beta, beta_real, atol=1e-6)

    def test_coincide_con_sklearn(self):
        from sklearn.linear_model import LinearRegression
        df = generar()
        modelo = m.RegresionLinealMultiple()
        modelo.entrenar(df[m.COLUMNAS_X], df[m.COLUMNA_Y])
        lr = LinearRegression().fit(df[m.COLUMNAS_X], df[m.COLUMNA_Y])
        esperado = np.concatenate([[lr.intercept_], lr.coef_])
        np.testing.assert_allclose(modelo.beta, esperado, atol=1e-6)

    def test_dimensiones_matrices(self):
        df = generar(n=30)
        modelo = m.RegresionLinealMultiple()
        modelo.entrenar(df[m.COLUMNAS_X], df[m.COLUMNA_Y])
        self.assertEqual(modelo.X.shape, (30, 5))
        self.assertEqual(modelo.Xt.shape, (5, 30))
        self.assertEqual(modelo.Y.shape, (30,))
        self.assertEqual(modelo.beta.shape, (5,))

    def test_metricas_valores_conocidos(self):
        r = m.calcular_metricas([10, 20, 30], [12, 18, 30])
        self.assertAlmostEqual(r["MAE"], 4 / 3)
        self.assertAlmostEqual(r["RMSE"], np.sqrt(8 / 3))
        self.assertAlmostEqual(r["R2"], 1 - 8 / 200)

    def test_prediccion_perfecta_da_r2_uno(self):
        r = m.calcular_metricas([1, 2, 3, 4], [1, 2, 3, 4])
        self.assertAlmostEqual(r["R2"], 1.0)
        self.assertAlmostEqual(r["MAE"], 0.0)

    def test_predecir_sin_entrenar_falla(self):
        with self.assertRaises(RuntimeError):
            m.RegresionLinealMultiple().predecir([[8, 256, 3, 80]])

    def test_division_entrenamiento_prueba(self):
        df = generar(n=100)
        train, test = m.dividir_datos(df)
        self.assertEqual(len(train), 80)
        self.assertEqual(len(test), 20)
        self.assertTrue(set(train.index).isdisjoint(test.index))

    def test_cargar_csv_valido(self):
        df = m.cargar_datos(CSV)
        self.assertEqual(len(df), 150)
        self.assertEqual(list(df.columns), m.COLUMNAS_X + [m.COLUMNA_Y])

    def test_csv_con_columnas_faltantes(self):
        with tempfile.TemporaryDirectory() as tmp:
            ruta = Path(tmp) / "malo.csv"
            pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(ruta, index=False)
            with self.assertRaises(ValueError):
                m.cargar_datos(ruta)

    def test_csv_inexistente(self):
        with self.assertRaises(FileNotFoundError):
            m.cargar_datos(BASE / "datos" / "no_existe.csv")

    def test_calidad_del_modelo_en_prueba(self):
        df = m.cargar_datos(CSV)
        train, test = m.dividir_datos(df)
        modelo = m.RegresionLinealMultiple()
        modelo.entrenar(train[m.COLUMNAS_X], train[m.COLUMNA_Y])
        r = m.calcular_metricas(test[m.COLUMNA_Y], modelo.predecir(test[m.COLUMNAS_X]))
        self.assertGreater(r["R2"], 0.9)
        self.assertLess(r["MAE"], 400)

    def test_genera_figura_png(self):
        df = m.cargar_datos(CSV)
        train, test = m.dividir_datos(df)
        modelo = m.RegresionLinealMultiple()
        modelo.entrenar(train[m.COLUMNAS_X], train[m.COLUMNA_Y])
        with tempfile.TemporaryDirectory() as tmp:
            ruta = graficos.guardar_figura(modelo, train, test, Path(tmp) / "resultados.png")
            self.assertTrue(ruta.exists())
            self.assertGreater(ruta.stat().st_size, 10_000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
