import numpy as np
import DnnLib

# --------------------------
# Datos de entrenamiento (XOR)
# --------------------------
X = np.array([
    [0., 0.],
    [0., 1.],
    [1., 0.],
    [1., 1.]
], dtype=float)

y = np.array([
    [0.],
    [1.],
    [1.],
    [0.]
], dtype=float)

# --------------------------
# Crear la capa densa
# 2 entradas -> 1 salida con activación SIGMOID
# --------------------------
layer = DnnLib.DenseLayer(2, 1, DnnLib.ActivationType.SIGMOID)

# --------------------------
# Función de pérdida (Binary Cross-Entropy)
# --------------------------
def bce_loss(y_hat, y_true, eps=1e-9):
    y_hat = np.clip(y_hat, eps, 1 - eps)  # evitar log(0)
    return -np.mean(y_true*np.log(y_hat) + (1 - y_true)*np.log(1 - y_hat))

# --------------------------
# Entrenamiento
# --------------------------
lr = 0.5        # tasa de aprendizaje
epochs = 5000   # número de iteraciones

for epoch in range(epochs):
    # Forward
    z = layer.forward_linear(X)    # pre-activación
    a = DnnLib.sigmoid(z)          # salida con sigmoid
    loss = bce_loss(a, y)

    # Backward
    batch_size = X.shape[0]
    delta = (a - y) / batch_size         # gradiente en la salida
    dW = delta.T @ X                     # (1,2) -> coincide con weights
    db = delta.sum(axis=0)               # (1,)

    # Update
    layer.weights -= lr * dW
    layer.bias -= lr * db

    # Print progreso
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# --------------------------
# Prueba del modelo
# --------------------------
pred = layer.forward(X)
print("\nPredicciones finales:")
for xi, pi in zip(X, pred):
    print(f"Entrada {xi} -> {pi[0]:.4f}")
