import json
import numpy as np
import time
import DnnLib
import argparse


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
        if layer_cfg["type"] == "dense":
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

            # REGULARIZACIÓN: Convertir string a RegularizerType
            if "regularizer" in layer_cfg:
                reg_config = layer_cfg["regularizer"]
                reg_type_str = reg_config.get("type", "none").lower()
                reg_lambda = reg_config.get("lambda", 0.001)
                
                # Convertir string a RegularizerType
                if reg_type_str == "l1":
                    reg_type = DnnLib.RegularizerType.L1
                elif reg_type_str == "l2":
                    reg_type = DnnLib.RegularizerType.L2
                else:  # "none" o cualquier otro valor
                    reg_type = DnnLib.RegularizerType.NONE
                
                layer.set_regularizer(reg_type, float(reg_lambda))

            # DROPOUT: Si la capa densa tiene soporte para dropout
            if "dropout" in layer_cfg and hasattr(layer, 'set_dropout'):
                dropout_config = layer_cfg["dropout"]
                dropout_rate = dropout_config.get("rate", 0.0)
                layer.set_dropout(float(dropout_rate))

            layers.append(layer)
            in_features = units
            
        elif layer_cfg["type"] == "dropout":
            # Cargar capa dropout separada
            dropout_rate = layer_cfg.get("rate", 0.5)
            dropout_seed = layer_cfg.get("seed", 42)
            dropout_layer = DnnLib.Dropout(dropout_rate, seed=dropout_seed)
            layers.append(dropout_layer)
            
        else:
            raise ValueError(f"❌ Tipo de capa no soportado en capa {i}: {layer_cfg['type']}")

    return layers, preprocess_scale, input_shape


def train_network(
        
    layers, X, Y,
    optimizer_name="adam",
    lr=0.001,
    epochs=5,
    batch_size=128
    # Los parámetros de regularización y dropout ahora se obtienen del JSON
):
    optimizer = OPTIMIZER_MAP[optimizer_name](lr)
    n_samples = X.shape[0]

    # =============================
    # OBTENER CONFIGURACIÓN DE REGULARIZACIÓN Y DROPOUT DEL JSON
    # =============================
    
    # Buscar capas densas
    dense_layers = [layer for layer in layers if isinstance(layer, DnnLib.DenseLayer)]
    
    # Aplicar regularización desde la configuración del JSON
    for i, layer in enumerate(dense_layers):
        # Verificar si la capa tiene información de regularización (se asume que se cargó del JSON)
        if hasattr(layer, 'regularizer_config'):
            reg_config = layer.regularizer_config
            if reg_config and reg_config.get('type', 'none').lower() != 'none':
                reg_type = reg_config['type'].upper()
                lambda_val = reg_config.get('lambda', 0.001)
                
                if reg_type == "L1":
                    layer.set_regularizer(DnnLib.RegularizerType.L1, lambda_val)
                    print(f"✅ Capa {i}: Regularización L1 (λ={lambda_val})")
                elif reg_type == "L2":
                    layer.set_regularizer(DnnLib.RegularizerType.L2, lambda_val)
                    print(f"✅ Capa {i}: Regularización L2 (λ={lambda_val})")

    # =============================
    # CONFIGURAR DROPOUT DESDE JSON
    # =============================
    dropout_layers = []
    dropout_map = {}
    new_layers = []
    
    for i, layer in enumerate(layers):
        if isinstance(layer, DnnLib.DenseLayer):
            # Agregar la capa densa
            new_layers.append(layer)
            
            # Verificar si esta capa necesita dropout (excepto la última)
            if i < len(dense_layers) - 1 and hasattr(layer, 'dropout_config'):
                dropout_config = layer.dropout_config
                if dropout_config and dropout_config.get('rate', 0) > 0:
                    dropout_rate = dropout_config['rate']
                    dropout_seed = dropout_config.get('seed', 42 + i)
                    
                    dropout_layer = DnnLib.Dropout(dropout_rate, seed=dropout_seed)
                    new_layers.append(dropout_layer)
                    dropout_layers.append(dropout_layer)
                    dropout_map[layer] = dropout_layer
                    
                    print(f"✅ Capa {i}: Dropout {dropout_rate:.1%}")
        else:
            # Mantener otras capas (si las hay)
            new_layers.append(layer)
    
    layers = new_layers
    original_layers = dense_layers.copy()

    # =============================
    # DETECTAR TIPO DE PROBLEMA
    # =============================
    last_dense_layer = original_layers[-1] if original_layers else None
    use_special_gradient = (last_dense_layer and 
                           last_dense_layer.activation_type == DnnLib.ActivationType.SOFTMAX)

    # =============================
    # BUCLE DE ENTRENAMIENTO
    # =============================
    for epoch in range(epochs):
        indices = np.random.permutation(n_samples)
        X_shuff, Y_shuff = X[indices], Y[indices]

        total_loss = 0
        total_reg_loss = 0
        n_batches = 0

        # Modo entrenamiento para dropout
        for layer in layers:
            if hasattr(layer, 'training'):  # Es capa dropout
                layer.training = True

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

            # Regularización (siempre calcular, puede que algunas capas tengan regularización)
            reg_loss = 0.0
            for layer in original_layers:
                reg_loss += layer.compute_regularization_loss()
            
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
            # Actualizar última capa
            grad = last_dense_layer.backward(grad)
            optimizer.update(last_dense_layer)

            # Actualizar capas restantes
            for j in reversed(range(len(original_layers) - 1)):
                # Aplicar backward en capa dropout si existe
                if original_layers[j] in dropout_map:
                    grad = dropout_map[original_layers[j]].backward(grad)
                
                # Backward en capa densa
                grad = original_layers[j].backward(grad)
                optimizer.update(original_layers[j])

            n_batches += 1

        # =============================
        # MÉTRICAS
        # =============================
        avg_loss = total_loss / n_batches
        avg_reg_loss = total_reg_loss / n_batches

        # Modo inferencia para dropout
        for layer in layers:
            if hasattr(layer, 'training'):
                layer.training = False

        # Calcular accuracy en todos los datos
        output_val = X
        for layer in original_layers:  # Usar solo capas densas para inferencia
            output_val = layer.forward(output_val)

        preds = np.argmax(output_val, axis=1)
        true_labels = np.argmax(Y, axis=1)
        acc = np.mean(preds == true_labels)

        # Información de regularización y dropout activo
        active_dropouts = [d for d in dropout_layers if hasattr(d, 'dropout_rate') and d.dropout_rate > 0]
        dropout_info = ""
        if active_dropouts:
            rates = [d.dropout_rate for d in active_dropouts]
            dropout_info = f" - Dropout: {min(rates):.1%}-{max(rates):.1%}"

        print(f"Época {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - RegLoss: {avg_reg_loss:.4f} - Acc: {acc*100:.2f}%{dropout_info}")

        # Reducción adaptativa de dropout (opcional)
        if active_dropouts and epoch > epochs // 2:
            for dropout_layer in active_dropouts:
                current_rate = dropout_layer.dropout_rate
                new_rate = max(0.1, current_rate * 0.99)  # Reducir 1% por época
                dropout_layer.dropout_rate = new_rate

    print("✅ Entrenamiento completado")
    return original_layers


def save_model_to_json(layers, path, input_shape=[28, 28], scale=255.0):
    model = {
        "input_shape": input_shape,
        "preprocess": {"scale": scale},
        "layers": []
    }

    activation_reverse = {v: k for k, v in ACTIVATION_MAP.items()}

    for layer in layers:
        if isinstance(layer, DnnLib.DenseLayer):
            act_str = activation_reverse.get(layer.activation_type, "relu")
            layer_dict = {
                "type": "dense",
                "units": int(layer.bias.shape[0]),
                "activation": act_str,
                "W": layer.weights.tolist(),
                "b": layer.bias.tolist()
            }
            
            # GUARDAR INFORMACIÓN DE REGULARIZACIÓN
            if hasattr(layer, 'regularizer_config'):
                layer_dict["regularizer"] = layer.regularizer_config
            else:
                # Si no tiene configuración, guardar none por defecto
                layer_dict["regularizer"] = {"type": "none", "lambda": 0.0}
            
            # GUARDAR INFORMACIÓN DE DROPOUT
            if hasattr(layer, 'dropout_config'):
                layer_dict["dropout"] = layer.dropout_config
            else:
                # Si no tiene configuración, guardar rate 0 por defecto
                layer_dict["dropout"] = {"rate": 0.0, "seed": 42}
            
            model["layers"].append(layer_dict)
        else:
            # Para capas que no son DenseLayer (como Dropout), guardar su información
            if hasattr(layer, 'dropout_rate'):  # Es una capa Dropout
                dropout_dict = {
                    "type": "dropout",
                    "rate": float(layer.dropout_rate),
                    "seed": int(getattr(layer, 'seed', 42))
                }
                model["layers"].append(dropout_dict)

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

def main():
    parser = argparse.ArgumentParser(description="Entrenar una red neuronal")

    # Rutas
    parser.add_argument("--model", required=True, help="Ruta del modelo base (.json)")
    parser.add_argument("--train", required=True, help="Ruta del dataset de entrenamiento (.npz)")
    parser.add_argument("--output", required=True, help="Ruta para guardar el modelo entrenado (.json)")

    # Hiperparámetros (solo los esenciales)
    parser.add_argument("--optimizer", default="adam", choices=["adam", "sgd", "rmsprop"], help="Optimizador (default: adam)")
    parser.add_argument("--lr", type=float, default=0.001, help="Tasa de aprendizaje (default: 0.001)")
    parser.add_argument("--epochs", type=int, default=5, help="Número de épocas (default: 5)")
    parser.add_argument("--batch-size", type=int, default=128, help="Tamaño de batch (default: 128)")

    # Se quitan los parámetros de regularización y dropout (ahora vienen del JSON)
    # --regularizer, --lambda, --dropout, --dropout-rate

    args = parser.parse_args()

    print("\n🚀 Entrenando la red...")
    print("📋 Configuración de regularización y dropout se obtendrá del archivo del modelo")

    # Cargar red inicial (que incluye configuración de regularización y dropout)
    layers, scale, input_shape = cargarjson(args.model)
    input_size = int(np.prod(input_shape))

    # Mostrar información sobre la configuración cargada
    dense_layers = [layer for layer in layers if isinstance(layer, DnnLib.DenseLayer)]
    dropout_layers = [layer for layer in layers if hasattr(layer, 'dropout_rate')]
    
    print(f"📊 Modelo cargado: {len(dense_layers)} capas densas, {len(dropout_layers)} capas dropout")
    
    # Mostrar configuración de regularización
    for i, layer in enumerate(dense_layers):
        if hasattr(layer, 'regularizer_config'):
            reg_config = layer.regularizer_config
            if reg_config and reg_config.get('type', 'none') != 'none':
                print(f"   Capa {i}: Regularización {reg_config['type'].upper()} (λ={reg_config.get('lambda', 0.001)})")

    # Cargar dataset
    data = np.load(args.train)
    images = data["images"]
    labels = data["labels"]

    N = images.shape[0]
    X = images.reshape(N, input_size).astype(np.float64) / scale

    # One-hot
    # Encontrar la última capa densa para determinar el número de clases
    last_dense_layer = None
    for layer in reversed(layers):
        if isinstance(layer, DnnLib.DenseLayer):
            last_dense_layer = layer
            break
    
    if last_dense_layer:
        K = last_dense_layer.weights.shape[0]
    else:
        raise ValueError("❌ No se encontró ninguna capa densa en el modelo")

    Y = np.zeros((N, K), dtype=np.float64)
    Y[np.arange(N), labels] = 1.0

    print(f"📈 Datos: {N} muestras, {K} clases")
    print(f"⚙️  Hiperparámetros: {args.optimizer}, lr={args.lr}, épocas={args.epochs}, batch_size={args.batch_size}")

    # Entrenar (sin parámetros de regularización/dropout)
    trained_layers = train_network(
        layers, X, Y,
        optimizer_name=args.optimizer,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size
        # Los parámetros de regularización y dropout ahora vienen del JSON
    )

    # Guardar modelo entrenado (conservará la configuración de regularización/dropout)
    save_model_to_json(trained_layers, args.output, input_shape, scale)
    print(f"✅ Modelo entrenado guardado en {args.output}")

    # Evaluar el modelo entrenado
    print("\n📊 Evaluando modelo final...")
    accuracy = evaluate_accuracy(X, Y, trained_layers)
    print(f"🎯 Accuracy final: {accuracy*100:.2f}%")


if __name__ == "__main__":
    main()