import unittest

import numpy as np

from Boltzmann import energia, probabilidad_activacion, simular


class PruebasBoltzmann(unittest.TestCase):
    def setUp(self):
        self.pesos = np.array([[0.0, 1.0], [1.0, 0.0]])
        self.sesgos = np.array([0.0, 0.0])

    def test_energia_estado_activo(self):
        self.assertEqual(energia(np.array([1, 1]), self.pesos, self.sesgos), -1.0)

    def test_probabilidad_sin_campo_es_media(self):
        self.assertAlmostEqual(
            probabilidad_activacion(0, np.array([0, 0]), self.pesos, self.sesgos, 1.0),
            0.5,
        )

    def test_semilla_reproduce_recorrido(self):
        primero = simular(np.array([0, 1]), self.pesos, self.sesgos, 1.0, 20, 7)
        segundo = simular(np.array([0, 1]), self.pesos, self.sesgos, 1.0, 20, 7)
        self.assertEqual(primero, segundo)

    def test_rechaza_diagonal_no_nula(self):
        with self.assertRaises(ValueError):
            energia(np.array([0, 1]), np.eye(2), self.sesgos)


if __name__ == "__main__":
    unittest.main()
