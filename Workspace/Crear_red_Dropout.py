import json
import numpy as np

def create_random_model():
    input_size = 784
    layers = []

    # Capa oculta 1: 784 -> 128
    units1 = 128
    weights1 = np.random.randn(units1, input_size) * 0.01
    bias_1 = np.zeros(units1)
    layers.append({
        "type": "dense",
        "units": units1,
        "activation": "relu",
        "W": weights1.tolist(),
        "b": bias_1.tolist(),
        "regularizer": {
            "type": "l2",  # o "l1", "none"
            "lambda": 0.001
        },
        "dropout": {
            "rate": 0.2,   # tasa de dropout para esta capa
            "seed": 42     # semilla opcional para reproducibilidad
        }
    })

    # Capa oculta 2: 128 -> 64
    units2 = 64
    W2 = np.random.randn(units2, units1) * 0.01
    b2 = np.zeros(units2)
    layers.append({
        "type": "dense",
        "units": units2,
        "activation": "relu",
        "W": W2.tolist(),
        "b": b2.tolist(),
        "regularizer": {
            "type": "l2",
            "lambda": 0.001
        },
        "dropout": {
            "rate": 0.3,
            "seed": 42
        }
    })

    # Capa salida: 64 -> 10
    units3 = 10
    W3 = np.random.randn(units3, units2) * 0.01
    b3 = np.zeros(units3)
    layers.append({
        "type": "dense",
        "units": units3,
        "activation": "softmax",
        "W": W3.tolist(),
        "b": b3.tolist(),
        "regularizer": {
            "type": "none"  # normalmente no se regulariza la capa de salida
        },
        "dropout": {
            "rate": 0.0  # normalmente no se aplica dropout a la salida
        }
    })

    # Capas de dropout adicionales (si se desean como capas separadas)
    layers.insert(1, {
        "type": "dropout",
        "rate": 0.2,
        "seed": 42
    })
    
    layers.insert(3, {
        "type": "dropout", 
        "rate": 0.3,
        "seed": 42
    })

    model = {
        "input_shape": [28, 28],
        "preprocess": {"scale": 255.0},
        "layers": layers
    }

    return model

if __name__ == "__main__":
    model = create_random_model()

    with open("workspace/RedDrop.json", "w") as f:
        json.dump(model, f, indent=2)

    print("✅ Modelo con dropout y regularización guardado en RedDrop.json")