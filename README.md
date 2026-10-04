# Simulación de una Máquina de Boltzmann

Implementación educativa desde cero de una red de Boltzmann estocástica. El
objetivo es observar la evolución de sus estados y su energía, no entrenar un
modelo ni usar una clase externa que lo implemente.

## Requisitos

- Python 3.10 o superior
- NumPy
- Matplotlib (solo para la gráfica)

## Ejecución

```bash
python Boltzmann.py
```

La ejecución usa la semilla documentada `20261004` y deja los resultados en
`resultados/`.

## Modelo

Se usan unidades binarias `s_i ∈ {0, 1}` y la energía es:

`E(s) = -1/2 sᵀWs - bᵀs`

La probabilidad de activar la unidad `i` se calcula con la sigmoide del campo
local `(Σ_j W_ij s_j + b_i) / T`. La actualización de una unidad se hace por
muestreo Bernoulli; por tanto, dos ejecuciones con igual semilla son
reproducibles.

## Restricciones cumplidas

- La matriz de pesos y el vector de sesgos se representan explícitamente.
- Se valida simetría, diagonal nula, dimensiones y estados binarios.
- Cada actualización registra paso, neurona seleccionada, estado y energía.
- Se exportan el recorrido completo y las frecuencias empíricas de estados a
  CSV.

No se utiliza TensorFlow, Keras, PyTorch, Scikit-learn ni bibliotecas que
implementen una máquina de Boltzmann.
