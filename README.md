# TrabajoCNN
# 1. Extensiones:
.py = Python

.ipynb = Jupyter Notebook


# 2. Direfencias Low y High:

a) La estructura de la CNN, la primera tiene menos capas por lo que tiene una menor precision que la segunda.

Estructura de .Low:
- Capa de entrada convolucional, 32 mapas de características con un tamaño de 3 x 3, una activación de rectier y una restricción de peso de la norma máxima establecida en 3.
- Abandono fijado en el 20%.
- Capa convolucional, 32 mapas de características con un tamaño de 3 x 3, una función de activación de rectier y una restricción de peso de la norma máxima establecida en 3.
- Capa máxima de pool con el tamaño 2 x 2.
- Capa Flatten.
- Capa totalmente conectada con 512 unidades y función de activación de rectier. 21.3. CNN simple para CIFAR-10 152.
- Abandono al 50%.
- Capa de salida totalmente conectada con 10 unidades y función de activación softmax.

Estructura de .High:
- Capa de entrada convolucional, 32 mapas de características con un tamaño de 3 x 3 y una activación de rectier.
- Abandono al 20%.
- Capa convolucional, 32 mapas de características con un tamaño de 3 x 3 y una función de activación de rectier.
- Capa máxima de pooling con el tamaño 2 x 2.
- Capa convolucional, 64 mapas de características con un tamaño de 3 x 3 y una función de activación de rectier.
- Abandono al 20%.
- Capa convolucional, 64 mapas de características con un tamaño de 3 x 3 y una función de activación de rectier.
- Capa máxima de pooling con el tamaño 2 x 2.
- Capa convolucional, 128 mapas de características con un tamaño de 3 x 3 y una función de activación de rectier.
- Abandono al 20%.
- Capa convolucional, 128 mapas de características con un tamaño de 3 x 3 y una función de activación de rectier.
- Capa máxima de pooling con el tamaño 2 x 2.
- Capa flatten.
- Abandono al 20%.
- Capa totalmente conectada con 1.024 unidades y función de activación de rectier.
- Abandono al 20%.
- Capa totalmente conectada con 512 unidades y función de activación de rectier.
- Abandono al 20%.
- Capa de salida totalmente conectada con 10 unidades y función de activación softmax.

b) High tiene más épocas, a mayor epocas mayor es el acurracy.
