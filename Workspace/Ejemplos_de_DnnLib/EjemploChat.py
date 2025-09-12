import numpy as np
import DnnLib

# Datos de entrada
x = np.array([[0.5, -0.2, 0.1]])

# Capa densa con 3 entradas, 2 salidas, activación ReLU
layer = DnnLib.DenseLayer(3, 2, DnnLib.ActivationType.RELU)

# Definir manualmente pesos y bias
layer.weights = np.array([[0.1, 0.2, 0.3],
                          [0.4, 0.5, 0.6]])
layer.bias = np.array([0.01, -0.02])

# Forward con activación
y = layer.forward(x)
print("Salida con activación:", y)

# Forward lineal (sin activación)
y_lin = layer.forward_linear(x)
print("Salida lineal:", y_lin)

# Usar funciones de activación
print("Sigmoid:", DnnLib.sigmoid(np.array([0.0, 2.0, -1.0])))
