import json
import numpy as np
import matplotlib.pyplot as plt
import time
import DnnLib
import argparse  # ← Añadimos esta importación

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


def train_network(layers, X, Y, optimizer_name="adam", lr=0.001, epochs=5, batch_size=128):
    optimizer = OPTIMIZER_MAP[optimizer_name](lr)
    n_samples = X.shape[0]
    
    # Verificar si la última capa usa softmax
    last_layer = layers[-1]
    use_special_gradient = (last_layer.activation_type == DnnLib.ActivationType.SOFTMAX)
    
    if use_special_gradient:
        print("✅ Usando softmax_crossentropy_gradient para eficiencia")

    for epoch in range(epochs):
        indices = np.random.permutation(n_samples)
        X_shuff, Y_shuff = X[indices], Y[indices]

        total_loss = 0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            X_batch = X_shuff[i:i+batch_size]
            Y_batch = Y_shuff[i:i+batch_size]

            # =============================
            # Forward pass SIMPLIFICADO
            # =============================
            activations = [X_batch]
            for layer in layers:
                activations.append(layer.forward(activations[-1]))
            output = activations[-1]

            # =============================
            # Loss
            # =============================
            loss = DnnLib.cross_entropy(output, Y_batch)
            total_loss += loss
            
            # =============================
            # Gradiente (MANTENIENDO LA LÓGICA ESPECIAL)
            # =============================
            if use_special_gradient:
                # Para Softmax: usar forward_linear + softmax_crossentropy_gradient
                logits = last_layer.forward_linear(activations[-2])  # Capa anterior
                grad = DnnLib.softmax_crossentropy_gradient(logits, Y_batch)
            else:
                # Para otras activaciones: normal
                grad = DnnLib.cross_entropy_gradient(output, Y_batch)

            # =============================
            # Backward + update
            # =============================
            for j in reversed(range(len(layers))):
                grad = layers[j].backward(grad)
                optimizer.update(layers[j])

            n_batches += 1

        # =============================
        # Métricas
        # =============================
        avg_loss = total_loss / n_batches
        
        # Calcular accuracy
        output_val = X
        for layer in layers:
            output_val = layer.forward(output_val)
        preds = np.argmax(output_val, axis=1)
        true_labels = np.argmax(Y, axis=1)
        acc = np.mean(preds == true_labels)

        print(f"Época {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Acc: {acc*100:.2f}%")

    return layers


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


# =============================
# Main CON ARGPARSE
# =============================
if __name__ == "__main__":
    # Configurar parser de argumentos
    parser = argparse.ArgumentParser(description="Train Mnist Model.")
    parser.add_argument('--epochs', type=int, default=10, help="Numero de epocas")
    parser.add_argument('--batchsize', type=int, default=64, help="Tamano de batch")
    parser.add_argument('--model', default="/workspace/modelo.json", help="Ruta de modelo")
    parser.add_argument('--dataset', default="/workspace/mnist_train.npz", help="Ruta de dataset")
    parser.add_argument('--modelgen', default="/workspace/modeloGEN.json", help="Ruta de modelo generado")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--optimizer', default="adam", choices=["sgd", "adam", "rmsprop"], 
                       help="Optimizador a usar")

    args = parser.parse_args()

    # Mostrar configuración
    print("⚙️  Configuración de entrenamiento:")
    print(f"   - Épocas: {args.epochs}")
    print(f"   - Batch size: {args.batchsize}")
    print(f"   - Learning rate: {args.lr}")
    print(f"   - Optimizador: {args.optimizer}")
    print(f"   - Modelo entrada: {args.model}")
    print(f"   - Dataset: {args.dataset}")
    print(f"   - Modelo salida: {args.modelgen}")
    print("-" * 50)

    # Cargar red
    try:
        layers, scale, input_shape = cargarjson(args.model)
        input_size = int(np.prod(input_shape))
        print(f"✅ Red cargada: {len(layers)} capas, input_size={input_size}")
        
        # Mostrar información de la red
        for i, layer in enumerate(layers):
            act_type = "Softmax" if layer.activation_type == DnnLib.ActivationType.SOFTMAX else "Otra"
            print(f"  Capa {i}: {layer.weights.shape[1]} -> {layer.weights.shape[0]} - Activación: {act_type}")
            
    except Exception as e:
        print(f"❌ Error cargando el modelo: {e}")
        exit(1)

    # Dataset
    try:
        data = np.load(args.dataset)
        images = data["images"]
        labels = data["labels"]
        print(f"✅ Dataset cargado: {images.shape[0]} muestras")
    except Exception as e:
        print(f"❌ Error cargando dataset: {e}")
        exit(1)

    # Preprocesamiento
    N = images.shape[0]
    X = images.reshape(N, -1).astype(np.float64) / scale

    # One-hot encoding
    K = layers[-1].bias.shape[0]
    Y = np.zeros((N, K), dtype=np.float64)
    Y[np.arange(N), labels] = 1.0

    print(f"📊 Datos preparados: X={X.shape}, Y={Y.shape}")

    # Entrenar
    try:
        trained_layers = train_network(
            layers, X, Y,
            optimizer_name=args.optimizer,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batchsize
        )
    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    # Evaluación
    output_final = X
    for layer in trained_layers:
        output_final = layer.forward(output_final)
    preds = np.argmax(output_final, axis=1)
    accuracy = np.mean(preds == labels)

    print(f"\n✅ Precisión final: {accuracy*100:.2f}%")

    # Guardar modelo
    try:
        save_model_to_json(trained_layers, args.modelgen, input_shape, scale)
        print("🎉 Entrenamiento completado exitosamente!")
    except Exception as e:
        print(f"❌ Error guardando modelo: {e}")
        import traceback
        traceback.print_exc()

    # Estadísticas finales
    print("\n📈 Estadísticas finales:")
    print(f"   - Precisión: {accuracy*100:.2f}%")
    print(f"   - Número de parámetros: {sum(layer.weights.size + layer.bias.size for layer in trained_layers):,}")
    print(f"   - Épocas completadas: {args.epochs}")
    print(f"   - Modelo guardado en: {args.modelgen}")