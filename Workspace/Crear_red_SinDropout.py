import json
import numpy as np

def create_random_model():

    input_size = 784
    layers = []

    # Vamos a crear una capa oculta de 728 ->128 

    units1 = 128

    weights1 = np.random.randn(units1, input_size) * 0.01
    bias_1 = np.zeros(units1)
    layers.append({
        "type": "dense",
        "units": units1,
        "activation": "relu",
        "W" : weights1.tolist(),
        "b": bias_1.tolist()

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
        "b": b2.tolist()
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
        "b": b3.tolist()
    })

    model = {
        "input_shape": [28, 28],
        "preprocess": {"scale": 255.0},
        "layers": layers
    }

    return model


if __name__ == "__main__":
    model = create_random_model()

    with open("workspace/Red_Neuro2.json", "w") as f:
        json.dump(model, f, indent=2)

    print("✅ Modelo inicializado aleatoriamente guardado en Red_simple.json")