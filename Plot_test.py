import numpy as np
import matplotlib.pyplot as plt
import DnnLib

# Cargar dataset desde la carpeta montada
data = np.load("/workspace/mnist_test.npz") 
images = data["images"]
labels = data["labels"]

print("Shape imágenes:", images.shape)
print("Shape etiquetas:", labels.shape)

# Tomar 3 ejemplos
X = images[:5]
y = labels[:5]
print("Etiquetas:", y)

# Mostrar las 3 imágenes
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(X[i], cmap="gray")
    plt.title(f"Label: {y[i]}")
    plt.axis("off")
plt.show()

# Aplanar imágenes (28x28 → 784) para pasarlas por la capa
X_flat = X.reshape(5, -1)

# Crear capa densa 784 → 10 con activación Softmax
layer = DnnLib.DenseLayer(784, 10, DnnLib.ActivationType.SOFTMAX)

# Forward
out = layer.forward(X_flat)
print("Salida de la red (3x10):")
print(out)