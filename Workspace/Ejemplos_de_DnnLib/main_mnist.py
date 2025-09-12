import numpy as np
import DnnLib

# --------------------------
# Cargar dataset MNIST desde npz
# --------------------------
data = np.load("mnist_test.npz")

print("Archivos dentro del npz:", data.files)

# Usar las claves correctas
x_test = data["images"]
y_test = data["labels"]

print("x_test shape:", x_test.shape)  # Ej: (10000, 28, 28)
print("y_test shape:", y_test.shape)  # Ej: (10000,)

# --------------------------
# Preprocesamiento
# --------------------------
X = x_test.reshape(x_test.shape[0], -1).astype(np.float32) / 255.0
Y = y_test

print("X shape después de aplanar:", X.shape)  # (10000, 784)

# --------------------------
# Crear modelo DnnLib
# --------------------------
l1 = DnnLib.DenseLayer(784, 128, DnnLib.ActivationType.RELU)
l2 = DnnLib.DenseLayer(128, 10, DnnLib.ActivationType.SOFTMAX)

# --------------------------
# Forward con algunas imágenes
# --------------------------
sample = X[:5]
logits1 = l1.forward_linear(sample)
a1 = DnnLib.relu(logits1)
logits2 = l2.forward_linear(a1)
y_hat = DnnLib.softmax(logits2)

print("\nPredicciones para las 5 primeras imágenes:")
print("Probabilidades (Softmax):\n", y_hat)
print("Predicciones (argmax):", np.argmax(y_hat, axis=1))
print("Etiquetas reales:", Y[:5])
