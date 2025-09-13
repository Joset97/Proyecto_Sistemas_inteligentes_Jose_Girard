import json
import numpy as np
import matplotlib.pyplot as plt
import time
import DnnLib
#Este codigo es un test con optimizadores con el dataset de test
#No es el usado para entrenar

# =============================
# Mapeo de activaciones
# =============================
ACTIVATION_MAP = {
    "sigmoid": DnnLib.ActivationType.SIGMOID,
    "tanh": DnnLib.ActivationType.TANH,
    "relu": DnnLib.ActivationType.RELU,
    "softmax": DnnLib.ActivationType.SOFTMAX
}

# =============================
# Mapeo de optimizadores
# =============================
OPTIMIZER_MAP = {
    "sgd": lambda lr=0.01: DnnLib.SGD(learning_rate=lr),
    "adam": lambda lr=0.001: DnnLib.Adam(learning_rate=lr),
    "rmsprop": lambda lr=0.001: DnnLib.RMSprop(learning_rate=lr)
}

# =============================
# Cargar red desde JSON
# =============================
def load_model_from_json(path):
    with open(path, "r") as f:
        config = json.load(f)

    input_shape = config.get("input_shape", [28, 28])
    input_size = int(np.prod(input_shape))

    preprocess_scale = config.get("preprocess", {}).get("scale", 1.0)

    layers = []
    in_features = input_size

    for i, layer_cfg in enumerate(config["layers"]):
        if layer_cfg["type"] != "dense":
            continue

        units = layer_cfg["units"]
        activation_str = layer_cfg.get("activation", "relu").lower()
        activation = ACTIVATION_MAP.get(activation_str, DnnLib.ActivationType.RELU)

        layer = DnnLib.DenseLayer(in_features, units, activation)

        # Cargar pesos/bias si existen
        W = layer_cfg.get("W", [])
        b = layer_cfg.get("b", [])
        if W and b:
            W = np.array(W, dtype=np.float64)
            b = np.array(b, dtype=np.float64)
            if W.shape == (units, in_features):
                layer.weights = W
                layer.bias = b
            elif W.shape == (in_features, units):
                layer.weights = W.T
                layer.bias = b
                print(f"ℹ️ Corrigiendo transposición de pesos en capa {i}")

        layers.append(layer)
        in_features = units

    return layers, preprocess_scale, input_shape

# =============================
# Entrenamiento
# =============================
def train_network(layers, X, Y, optimizer_name="adam", lr=0.001, epochs=5, batch_size=128):
    optimizer = OPTIMIZER_MAP[optimizer_name](lr)
    n_samples = X.shape[0]

    start_time = time.time()

    for epoch in range(epochs):
        indices = np.random.permutation(n_samples)
        X_shuff, Y_shuff = X[indices], Y[indices]

        total_loss = 0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            X_batch = X_shuff[i:i+batch_size]
            Y_batch = Y_shuff[i:i+batch_size]

            # Forward
            activations = [X_batch]
            for layer in layers:
                activations.append(layer.forward(activations[-1]))
            output = activations[-1]

            # Loss + gradiente
            loss = DnnLib.cross_entropy(output, Y_batch)
            grad = DnnLib.cross_entropy_gradient(output, Y_batch)

            # Backward + update
            for j in reversed(range(len(layers))):
                grad = layers[j].backward(grad)
                optimizer.update(layers[j])

            total_loss += loss
            n_batches += 1

        avg_loss = total_loss / n_batches
        preds = np.argmax(output, axis=1)
        true_labels = np.argmax(Y_batch, axis=1)
        acc = np.mean(preds == true_labels)

        print(f"Época {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Acc: {acc*100:.2f}%")

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\n⏱️ Tiempo total de entrenamiento: {elapsed:.2f} segundos")

    return layers

# =============================
# Main
# =============================
if __name__ == "__main__":
    dataset_path = "/workspace/mnist_test.npz"
    model_path = "/workspace/mnist_mlp_pretty.json"

    # Cargar red
    layers, scale, input_shape = load_model_from_json(model_path)
    input_size = int(np.prod(input_shape))

    # Dataset
    data = np.load(dataset_path)
    images = data["images"]
    labels = data["labels"]

    N = images.shape[0]
    X = images.reshape(N, input_size).astype(np.float64) / scale

    # One-hot encoding
    K = layers[-1].weights.shape[0]
    Y = np.zeros((N, K), dtype=np.float64)
    Y[np.arange(N), labels] = 1.0

    # Entrenar
    trained_layers = train_network(
        layers, X, Y,
        optimizer_name="sgd",  # cambia aquí: "sgd", "adam", "rmsprop"
        lr=0.001,
        epochs=5,
        batch_size=128
    )

    # Evaluación final
    activations = [X]
    for layer in trained_layers:
        activations.append(layer.forward(activations[-1]))
    final_output = activations[-1]
    preds = np.argmax(final_output, axis=1)
    accuracy = np.mean(preds == labels)

    print(f"\n✅ Precisión final en test: {accuracy*100:.2f}%")

    # Mostrar algunas imágenes
    plt.figure(figsize=(10, 5))
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.imshow(images[i], cmap="gray")
        plt.axis("off")

        probs = final_output[i] * 100
        pred_class = preds[i]
        conf = probs[pred_class]
        plt.title(f"Pred: {pred_class}\nReal: {labels[i]}\nConf: {conf:.1f}%", fontsize=8)

    plt.tight_layout()
    plt.show()
