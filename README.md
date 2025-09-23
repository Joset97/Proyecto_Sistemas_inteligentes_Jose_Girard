

Como ejecutar el proyecto:

Primero hay que crear una red neuronal nueva ya sea con dropout o sin dropout con los archivos .py existentes

luego puede realizar el entrenamiento con normalidad con el siguiente comando

docker run -it --rm -v C:\Users\jrgir\Desktop\Sistemas_Inteligentes\Proyecto\workspace:/app -w /app iderashn/dnn-q32025:latest python Parte5.py --model Red_Neuro.json --train fashion_mnist_train.npz --output training.json --optimizer adam --lr 0.0005 --epochs 20 --batch-size 128

