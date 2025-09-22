import json
import numpy as np
import matplotlib.pyplot as plt
import time
import DnnLib

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


def cargarjson(path):
    with open(path, "r") as f:
        config = json.load(f)

    input_shape = config.get("input_shape", [28, 28])
    input_size = int(np.prod(input_shape))
    preprocess_scale = config.get("preprocess", {}).get("scale", 1.0)

    layers = []
    in_features = input_size

    for i, layer_cfg in enumerate(config["layers"]):
        if layer_cfg["type"] != "dense":
            raise ValueError(f"❌ Tipo de capa no soportado en capa {i}: {layer_cfg['type']}")

        units = layer_cfg["units"]
        activation_str = layer_cfg.get("activation", "").lower()
        if activation_str not in ACTIVATION_MAP:
            raise ValueError(f"❌ Activación desconocida en capa {i}: {activation_str}")

        activation = ACTIVATION_MAP[activation_str]
        layer = DnnLib.DenseLayer(in_features, units, activation)

        # Pesos y sesgos
        W = layer_cfg.get("W", [])
        b = layer_cfg.get("b", [])

        if W and b:
            W = np.array(W, dtype=np.float64)
            b = np.array(b, dtype=np.float64)

            if W.shape == (units, in_features) and b.shape == (units,):
                layer.weights = W
                layer.bias = b
            elif W.shape == (in_features, units) and b.shape == (units,):
                layer.weights = W.T
                layer.bias = b
                print(f"ℹ️ Corrigiendo transposición de pesos en capa {i}")
            else:
                raise ValueError(f"❌ Shapes inválidas en capa {i}")

        layers.append(layer)
        in_features = units

    return layers, preprocess_scale, input_shape


def train_network(
    layers, X, Y,
    optimizer_name="adam",
    lr=0.001,
    epochs=5,
    batch_size=128,
    regularizers_type=None,            # "L1", "L2" o None
    regularizers_lambda_val=0.001,
    Dropout_Training=False,
    Dropout_rate=0.5
):
    optimizer = OPTIMIZER_MAP[optimizer_name](lr)
    n_samples = X.shape[0]

    # =============================
    # Aplicar regularización si se pide
    # =============================
    if regularizers_type is not None:
        reg_type = regularizers_type.upper()
        for layer in layers:
            if isinstance(layer, DnnLib.DenseLayer):
                if reg_type == "L1":
                    layer.set_regularizer(DnnLib.RegularizerType.L1, regularizers_lambda_val)
                elif reg_type == "L2":
                    layer.set_regularizer(DnnLib.RegularizerType.L2, regularizers_lambda_val)
        print(f"✅ Regularización {regularizers_type} aplicada (λ={regularizers_lambda_val})")

    # =============================
    # Hacer copia de las capas originales
    # =============================
    original_layers = layers.copy()

    # =============================
    # Insertar dropout si se pide (después de cada capa oculta)
    # =============================
    dropout_layers = []
    dropout_map = {}
    if Dropout_Training:
        new_layers = []
        for i, layer in enumerate(layers):
            new_layers.append(layer)
            if isinstance(layer, DnnLib.DenseLayer) and i < len(layers) - 1:
                dropout_layer = DnnLib.Dropout(Dropout_rate)
                new_layers.append(dropout_layer)
                dropout_layers.append(dropout_layer)
                dropout_map[layer] = dropout_layer
        layers = new_layers
        print(f"✅ Dropout aplicado (rate={Dropout_rate})")

    # Última capa densa (para gradiente especial con softmax)
    last_dense_layer = original_layers[-1]
    use_special_gradient = (last_dense_layer.activation_type == DnnLib.ActivationType.SOFTMAX)

    for epoch in range(epochs):
        indices = np.random.permutation(n_samples)
        X_shuff, Y_shuff = X[indices], Y[indices]

        total_loss = 0
        total_reg_loss = 0
        n_batches = 0

        # Modo entrenamiento en dropout
        for d in dropout_layers:
            d.training = True

        for i in range(0, n_samples, batch_size):
            X_batch = X_shuff[i:i+batch_size]
            Y_batch = Y_shuff[i:i+batch_size]

            # Forward
            activation = X_batch
            for layer in layers:
                activation = layer.forward(activation)
            output = activation

            # Pérdida base
            data_loss = DnnLib.cross_entropy(output, Y_batch)
            loss = data_loss

            # Regularización
            if regularizers_type is not None:
                reg_loss = sum(
                    l.compute_regularization_loss() for l in original_layers if isinstance(l, DnnLib.DenseLayer)
                )
                loss += reg_loss
                total_reg_loss += reg_loss

            total_loss += data_loss

            # Gradiente
            if use_special_gradient:
                activation_before_last = X_batch
                for layer in layers[:-1]:
                    activation_before_last = layer.forward(activation_before_last)
                logits = last_dense_layer.forward_linear(activation_before_last)
                grad = DnnLib.softmax_crossentropy_gradient(logits, Y_batch)
            else:
                grad = DnnLib.cross_entropy_gradient(output, Y_batch)

            # Backward
            grad = last_dense_layer.backward(grad)
            optimizer.update(last_dense_layer)

            for j in reversed(range(len(original_layers) - 1)):
                if Dropout_Training and original_layers[j] in dropout_map:
                    grad = dropout_map[original_layers[j]].backward(grad)
                grad = original_layers[j].backward(grad)
                optimizer.update(original_layers[j])

            n_batches += 1

        # =============================
        # Métricas
        # =============================
        avg_loss = total_loss / n_batches
        avg_reg_loss = total_reg_loss / n_batches if regularizers_type is not None else 0

        for d in dropout_layers:
            d.training = False

        output_val = X
        for layer in original_layers:
            output_val = layer.forward(output_val)

        preds = np.argmax(output_val, axis=1)
        true_labels = np.argmax(Y, axis=1)
        acc = np.mean(preds == true_labels)

        reg_info = f" - RegLoss: {avg_reg_loss:.4f}" if regularizers_type is not None else ""
        print(f"Época {epoch+1}/{epochs} - Loss: {avg_loss:.4f}{reg_info} - Acc: {acc*100:.2f}%")

    return original_layers


def save_model_to_json(layers, path, input_shape=[28, 28], scale=255.0):
    model = {
        "input_shape": input_shape,
        "preprocess": {"scale": scale},
        "layers": []
    }

    activation_reverse = {v: k for k, v in ACTIVATION_MAP.items()}

    for layer in layers:
        act_str = activation_reverse.get(layer.activation_type, "relu")
        model["layers"].append({
            "type": "dense",
            "units": int(layer.bias.shape[0]),
            "activation": act_str,
            "W": layer.weights.tolist(),
            "b": layer.bias.tolist()
        })

    with open(path, "w") as f:
        json.dump(model, f, indent=2)

    print(f"💾 Modelo guardado en {path}")

def evaluate_accuracy(X, Y, layers, batch_size=128):
    """
    Evalúa accuracy de la red en dataset dado (X,Y).
    X debe estar preprocesado (reshape y normalizado).
    Y debe estar en formato one-hot.
    """
    n_samples = X.shape[0]
    correct = 0

    # Asegurarse de que todas las capas estén en modo evaluación
    for layer in layers:
        if hasattr(layer, 'training'):  # Si es una capa dropout
            layer.training = False

    for i in range(0, n_samples, batch_size):
        X_batch = X[i:i+batch_size]
        Y_batch = Y[i:i+batch_size]

        # Forward pass a través de TODAS las capas
        activation = X_batch
        for layer in layers:
            activation = layer.forward(activation)
        output = activation

        # Predicciones
        preds = np.argmax(output, axis=1)
        true_labels = np.argmax(Y_batch, axis=1)

        correct += np.sum(preds == true_labels)

    accuracy = correct / n_samples
    return accuracy


if __name__ == "__main__":

    opcion = 0
    while(opcion!=3):

        print("===== MENÚ PRINCIPAL =====")
        print("1. Entrenar red")
        print("2. Testear red")
        print("3. Salir")
        opcion = input("Seleccione opción (1/2): ")

        if opcion == "1":
            print("\n🚀 Entrenando la red...")

            # Preguntar rutas
            model_path = input("Ingrese ruta del modelo base (.json): ")
            train_path = input("Ingrese ruta del dataset de entrenamiento (.npz): ")
            trained_model_path = input("Ingrese ruta para guardar el modelo entrenado (.json): ")

            # Cargar red inicial
            layers, scale, input_shape = cargarjson(model_path)
            input_size = int(np.prod(input_shape))

            # Cargar dataset de entrenamiento
            data = np.load(train_path)
            images = data["images"]
            labels = data["labels"]

            N = images.shape[0]
            X = images.reshape(N, input_size).astype(np.float64) / scale

            # One-hot
            K = layers[-1].weights.shape[0]
            Y = np.zeros((N, K), dtype=np.float64)
            Y[np.arange(N), labels] = 1.0

            # Entrenar
            trained_layers = train_network(
                layers, X, Y,
                optimizer_name="adam",
                lr=0.001,
                epochs=20,
                batch_size=128
            )

            # Guardar modelo entrenado
            save_model_to_json(trained_layers, trained_model_path, input_shape, scale)
            print(f"✅ Modelo entrenado guardado en {trained_model_path}")

        elif opcion == "2":
            print("\n🔎 Evaluando red entrenada...")

            # Preguntar rutas
            trained_model_path = input("Ingrese ruta del modelo entrenado (.json): ")
            test_path = input("Ingrese ruta del dataset de prueba (.npz): ")

            # Cargar modelo entrenado
            layers, scale, input_shape = cargarjson(trained_model_path)
            input_size = int(np.prod(input_shape))

            # Cargar dataset de test
            data = np.load(test_path)
            images = data["images"]
            labels = data["labels"]

            N = images.shape[0]
            X = images.reshape(N, input_size).astype(np.float64) / scale

            # One-hot
            K = layers[-1].weights.shape[0]
            Y = np.zeros((N, K), dtype=np.float64)
            Y[np.arange(N), labels] = 1.0

            # Calcular accuracy
            acc = evaluate_accuracy(X, Y, layers)
            print(f"✅ Precisión en test: {acc*100:.2f}%")

        elif opcion == "3":

            break
            
        else:
            print("❌ Opción inválida")
