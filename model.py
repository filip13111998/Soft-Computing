#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.constraints import maxnorm
import numpy as np
import sys

import pickle


# In[2]:


X_Training = pickle.load(open("X_Training.pickle", 'rb'))
y_Training = pickle.load(open("y_Training.pickle", 'rb'))

X_Valid = pickle.load(open("X_Valid.pickle", 'rb'))
Y_Valid = pickle.load(open("y_Valid.pickle", 'rb'))

y_Test = pickle.load(open("y_Test.pickle", 'rb'))
X_Test = pickle.load(open("X_Test.pickle", 'rb'))
    


# In[3]:


y_Training = np.array(y_Training)
X_Training= X_Training.astype('float32')

X_Training = X_Training/255.0

Y_Valid = np.array(Y_Valid)

X_Valid = X_Valid.astype('float32')
X_Valid = X_Valid/255.0

y_Test = np.array(y_Test)
# Da li ovo sme
y_Test = y_Test.astype('float32')

X_Test = X_Test.astype('float32')
X_Test = X_Test / 255.0


# In[9]:


model = Sequential()

model.add(Conv2D(16, (7, 7), strides=(2, 2), padding="same", input_shape=(300, 300, 1)))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2))) 

# model.add(Conv2D(16, (3, 3), strides=1, padding="same"))
# model.add(LeakyReLU(alpha=0.1))
# model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(32, (3, 3), strides=1,padding="same"))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(32, (3, 3), strides=1,padding="same"))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2)))
# model.add(Dropout(0.1))


model.add(Conv2D(64, (3, 3), strides=1,padding="same"))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), strides=1,padding="same"))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2)))

# model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), strides=1,padding="same"))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), strides=1,padding="same"))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2)))


model.add(Dropout(0.3))


model.add(Flatten())

model.add(Dense(256))
model.add(LeakyReLU(alpha=0.1))

model.add(Dropout(0.2))
model.add(Dense(10, activation="softmax"))

# optimizer = keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer= 'Adam',
              metrics=['accuracy'])

model.fit(X_Training, y_Training, batch_size=128,
          epochs=40, validation_data=(X_Valid, Y_Valid) , shuffle=True)
model.save("model_finish")


# In[12]:


model = tf.keras.models.load_model("model_finish")
test_eval = model.evaluate(X_Test, y_Test, verbose=1, batch_size=16)

print("Test loss: " + str(test_eval[0]))
print("Test accuracy " + str(test_eval[1]))


# In[ ]:


print(classification_report(y_true, y_pred, target_names=target_names))

