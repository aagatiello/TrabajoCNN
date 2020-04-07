#!/usr/bin/env python
# coding: utf-8

# 1. Importacion de los paquetes necesarios:

# In[1]:


import matplotlib.pyplot as plt

from keras.datasets import cifar10

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.constraints import maxnorm

from keras.optimizers import SGD

from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.utils import np_utils


# 2. Cargar el conjunto de datos CIFAR-10 usando la función de ayuda de Keras

# In[2]:


(X_train, y_train), (X_test, y_test) = cifar10.load_data()


# 3. Normalizacion: Píxeles en rango 0-255 para canales rojo, verde y azul. Como los valores de entrada son conocidos podemos normalizar a rango 0-1 dividiendo cada valor por observación máxima (255). Para dividir cambiamos datos enteros a flotantes.

# In[3]:


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0


# 4. Codificacion de la salida: Variables de salida se denotan como vector de números enteros 0-1. Transformamos en matriz binaria para modelar mejor el problema de la clasificación.

# In[4]:


y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]


# 5. Creacion y compilacion del modelo CNN

# In[5]:


model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D())
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D())
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

epochs = 20
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.summary()


# 6. Ajustar modelo: 20 epocas y tamaño de lote 64. Numero pequeño de epocas para hacerlo mas rapido.

# In[6]:


model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=64)


# 7. Prediccion

# In[7]:


y_pred = model.predict(X_test)
fig, axis = plt.subplots(4, 4, figsize=(12, 14))
for i, ax in enumerate(axis.flat):
    ax.imshow(X_test[i], cmap='binary')
    ax.set(title = f"This class is {y_test[i].argmax()}\nPredict class is {y_pred[i].argmax()}");


# Classes - 0:airplane - 1:automobile - 2:bird - 3:cat - 4:deer - 5:dog - 6:frog - 7:horse - 8:ship - 9:truck

# In[8]:


scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

