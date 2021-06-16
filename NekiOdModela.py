import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
import numpy as np
import sys

import pickle

if __name__ == '__main__':

    X_Training = pickle.load(open("X_Training.pickle", 'rb'))
    y_Training = pickle.load(open("y_Training.pickle", 'rb'))

    X_Valid = pickle.load(open("X_Valid.pickle", 'rb'))
    Y_Valid = pickle.load(open("y_Valid.pickle", 'rb'))

    y_Test = pickle.load(open("y_Test.pickle", 'rb'))
    X_Test = pickle.load(open("X_Test.pickle", 'rb'))

    print("_--------------------------------------------------------------")
    print("_--------------------------------------------------------------")
    print("_--------------------------------------------------------------")
    print("_--------------------------------------------------------------")
    print("_--------------------------------------------------------------")

    y_Training = np.array(y_Training)

    X_Training = X_Training/255.0

    Y_Valid = np.array(Y_Valid)

    X_Valid = X_Valid/255.0

    y_Test = np.array(y_Test)
    X_Test = X_Test / 255.0

    unos = ''
    while (True):
        print('\n')
        print("Treninzi i modeli")
        print("1. Trainig one:...22%-LOSE ")
        print("2. FULL trining: 51%-Overfitting - 20 epoha")
        print("3. Three trining: 45%-Overfitting - 10 epoha")
        print("4. Seven Epohs trining: 48.2% - 7 epoha")
        print("5. Dropout trining: 51.4% - 7 epoha")
        print("6. Dropout-Shuffle trining:(acc = 84%) val = 70.5% - 7 epoha")
        print("7. Dropout-Shuffle trining: 53.9% - 7 epoha")

        #print('X  ZA IZLAZ')
        #unos = input(">> ")
        unos = '7'
        if (unos == '1'):

            # model = Sequential()
            # model.add(Conv2D(32, (3, 3), strides=1, padding="same", activation="relu", input_shape=(300, 300, 1)))
            # model.add(Dropout(0.4))
            # model.add(Conv2D(32, (3, 3), strides=1, padding="same", activation="relu"))
            # model.add(Dropout(0.4))
            # model.add(MaxPooling2D((2, 2)))

            # model.add(Conv2D(64, (3, 3), strides=1, padding="same", activation="relu"))
            # model.add(Dropout(0.4))
            # model.add(Conv2D(64, (3, 3), strides=1, padding="same", activation="relu"))
            # model.add(Dropout(0.4))
            # model.add(MaxPooling2D((2, 2)))

            # model.add(Conv2D(128, (3, 3), strides=1, padding="same", activation="relu"))
            # model.add(Dropout(0.4))
            # model.add(Conv2D(128, (3, 3), strides=1, padding="same", activation="relu"))
            # model.add(Dropout(0.4))
            # model.add(MaxPooling2D((2, 2)))

            # model.add(Flatten())

            # model.add(Dense(10, activation="softmax"))

            # model.compile(loss='sparse_categorical_crossentropy',
            #               optimizer='adam',
            #               metrics=['accuracy'])

            # model.fit(X_Training, y_Training, batch_size=64, epochs=20, validation_split=0.3)
            # model.save("model_vgg")

            model = tf.keras.models.load_model("model_vgg")
            test_eval = model.evaluate(
                X_Test, y_Test, verbose=1, batch_size=16)

            print("Test loss: " + str(test_eval[0]))
            print("Test accuracy " + str(test_eval[1]))

        if (unos == '2'):
            print("haha")
            # model.add(LeakyReLU(alpha = 0.1))

            # model.add(Conv2D(64, (4, 4), strides=1, padding="same", activation="relu"))
            # model.add(Dropout(0.5))
            # model.add(Conv2D(64, (4, 4), strides=1, padding="same", activation="relu"))
            # model.add(Dropout(0.5))
            # model.add(MaxPooling2D((3, 3)))

            # model.add(LeakyReLU(alpha = 0.1))

            # model.add(Conv2D(32, (4, 4), strides=1, padding="same", activation="relu"))
            # model.add(Dropout(0.5))
            # model.add(Conv2D(32, (4, 4), strides=1, padding="same", activation="relu"))
            # model.add(Dropout(0.5))
            # model.add(MaxPooling2D((3, 3)))

            # model.add(Dropout(0.5))
            # model.add(Dropout(0.5))

            ###############################################################################################################

            # model = Sequential()
            # model.add(Conv2D(32, (3, 3), strides=1, padding="same", activation="relu", input_shape=(200, 200, 1)))
            # model.add(MaxPooling2D((2, 2)))

            # model.add(Conv2D(64, (3, 3), strides=1, padding="same", activation="relu"))
            # model.add(MaxPooling2D((2, 2)))

            # model.add(Flatten())

            # model.add(Dense(64, activation="relu"))
            # model.add(Dense(10, activation="softmax"))

            # model.compile(loss='sparse_categorical_crossentropy',
            #               optimizer='adam',
            #               metrics=['accuracy'])

            # model.fit(X_Training, y_Training, batch_size=64, epochs=10, validation_split=0.3)
            # model.save("model_full")

            # model = tf.keras.models.load_model("model_full")
            # test_eval = model.evaluate(
            #     X_Test, y_Test)

            # print("Test loss: " + str(test_eval[0]))
            # print("Test accuracy " + str(test_eval[1]))

        elif (unos == '3'):
            # model = Sequential()
            # model.add(Conv2D(32, (3, 3), strides=1, padding="same",
            #                  activation="relu", input_shape=(200, 200, 1)))
            # model.add(MaxPooling2D((2, 2)))

            # model.add(Conv2D(64, (3, 3), strides=1,
            #                  padding="same", activation="relu"))
            # model.add(MaxPooling2D((2, 2)))

            # # model.add(Dropout(0.4))

            # model.add(Flatten())

            # model.add(Dense(64, activation="relu"))
            # model.add(Dense(10, activation="softmax"))

            # model.compile(loss='sparse_categorical_crossentropy',
            #               optimizer='adam',
            #               metrics=['accuracy'])

            # model.fit(X_Training, y_Training, batch_size=64,
            #           epochs=10, validation_split=0.3)
            # model.save("model_new_three")

            model = tf.keras.models.load_model("model_new_three")
            test_eval = model.evaluate(
                X_Test, y_Test)

            print("Test loss: " + str(test_eval[0]))
            print("Test accuracy " + str(test_eval[1]))
        elif (unos == '4'):
            # model = Sequential()
            # model.add(Conv2D(32, (3, 3), strides=1, padding="same",
            #                  activation="relu", input_shape=(200, 200, 1)))
            # model.add(MaxPooling2D((2, 2)))

            # model.add(Conv2D(64, (3, 3), strides=1,
            #                  padding="same", activation="relu"))
            # model.add(MaxPooling2D((2, 2)))

            # # model.add(Dropout(0.4))

            # model.add(Flatten())

            # model.add(Dense(64, activation="relu"))
            # model.add(Dense(10, activation="softmax"))

            # model.compile(loss='sparse_categorical_crossentropy',
            #               optimizer='adam',
            #               metrics=['accuracy'])

            # model.fit(X_Training, y_Training, batch_size=64,
            #           epochs=7, validation_split=0.3)
            # model.save("model_seven")

            model = tf.keras.models.load_model("model_seven")
            test_eval = model.evaluate(
                X_Test, y_Test, verbose=1, batch_size=16)

            print("Test loss: " + str(test_eval[0]))
            print("Test accuracy " + str(test_eval[1]))

        elif (unos == '5'):
            # model = Sequential()
            # model.add(Conv2D(32, (3, 3), strides=1, padding="same",
            #                  activation="relu", input_shape=(200, 200, 1)))
            # model.add(MaxPooling2D((2, 2)))

            # model.add(Conv2D(64, (3, 3), strides=1,
            #                  padding="same", activation="relu"))
            # model.add(MaxPooling2D((2, 2)))

            # model.add(Dropout(0.4))

            # model.add(Flatten())

            # model.add(Dense(64, activation="relu"))
            # model.add(Dense(10, activation="softmax"))

            # model.compile(loss='sparse_categorical_crossentropy',
            #               optimizer='adam',
            #               metrics=['accuracy'])

            # model.fit(X_Training, y_Training, batch_size=64,
            #           epochs=7, validation_split=0.3)
            # model.save("model_dropout")

            model = tf.keras.models.load_model("model_dropout")
            test_eval = model.evaluate(
                X_Test, y_Test, verbose=1, batch_size=16)

            print("Test loss: " + str(test_eval[0]))
            print("Test accuracy " + str(test_eval[1]))

        elif (unos == '6'):

            # model = Sequential()

            # model.add(Conv2D(32, (3, 3), strides=1, padding="same",activation="relu", input_shape=(200, 200, 1)))
            # model.add(MaxPooling2D((2, 2)))
            # model.add(Dropout(0.2))

            # model.add(Conv2D(64, (3, 3), strides=1, padding="same",activation="relu")
            # model.add(MaxPooling2D((2, 2)))
            # model.add(Dropout(0.2))

            # model.add(Conv2D(256, (3, 3), strides=1, padding="same",activation="relu")
            # model.add(MaxPooling2D((2, 2)))
            # model.add(Dropout(0.2))

            # model.add(Flatten())

            # model.add(Dense(256, activation="relu"))
            # model.add(Dropout(0.4))

            # model.add(Dense(10, activation="softmax"))

            # model.compile(loss='sparse_categorical_crossentropy',
            #             optimizer='adam',
            #             metrics=['accuracy'])

            # model.fit(X_Training, y_Training, batch_size=64,
            #         epochs=12, validation_data=(X_Valid, Y_Valid), shuffle=True)
            # model.save("model_dropout_shuffle")

            # model = Sequential()

            # model.add(Conv2D(32, (3, 3), strides=1, padding="same",activation="relu", input_shape=(200, 200, 1)))
            # model.add(MaxPooling2D((2, 2)))
            # model.add(Conv2D(32, (3, 3), strides=1, padding="same",activation="relu"))
            # model.add(MaxPooling2D((2, 2)))

            # model.add(Conv2D(64, (3, 3), strides=1,padding="same", activation="relu"))
            # model.add(MaxPooling2D((2, 2)))
            # model.add(Conv2D(64, (3, 3), strides=1,padding="same", activation="relu"))
            # model.add(MaxPooling2D((2, 2)))

            # model.add(Conv2D(128, (3, 3), strides=1,padding="same", activation="relu"))
            # model.add(MaxPooling2D((2, 2)))
            # model.add(Conv2D(128, (3, 3), strides=1,padding="same", activation="relu"))
            # model.add(MaxPooling2D((2, 2)))

            # model.add(Dropout(0.2))

            # model.add(Flatten())

            # model.add(Dense(128, activation="relu"))
            # model.add(Dropout(0.2))
            # model.add(Dense(10, activation="softmax"))

            # model.compile(loss='sparse_categorical_crossentropy',
            #               optimizer='adam',
            #               metrics=['accuracy'])

            # model.fit(X_Training, y_Training, batch_size=64,
            #           epochs=12, validation_data=(X_Valid, Y_Valid) , shuffle=True)
            # model.save("model_dropout_shuffle")

            model = tf.keras.models.load_model("model_dropout_shuffle")
            test_eval = model.evaluate(
                X_Test, y_Test, verbose=1, batch_size=16)

            print("Test loss: " + str(test_eval[0]))
            print("Test accuracy " + str(test_eval[1]))


        elif (unos == '7'):


            model = Sequential()
           
            #model.add(Conv2D(16, (3, 3), strides=1, padding="same",activation="relu", input_shape=(300, 300, 1)))
            #model.add(MaxPooling2D((2, 2))) 
            #model.add(Conv2D(16, (3, 3), strides=1, padding="same",activation="relu"))
            #model.add(MaxPooling2D((2, 2)))
        
            model.add(Conv2D(32, (3, 3), strides=1, padding="same",activation="relu", input_shape=(300, 300, 1))) #slucajno ostao 16
            model.add(MaxPooling2D((2, 2))) 
            model.add(Conv2D(32, (3, 3), strides=1, padding="same",activation="relu"))
            model.add(MaxPooling2D((2, 2)))

            model.add(Conv2D(64, (3, 3), strides=1,padding="same", activation="relu"))
            model.add(MaxPooling2D((2, 2)))
            model.add(Conv2D(64, (3, 3), strides=1,padding="same", activation="relu"))
            model.add(MaxPooling2D((2, 2)))
            
            model.add(Conv2D(128, (3, 3), strides=1,padding="same", activation="relu"))
            model.add(MaxPooling2D((2, 2)))
            model.add(Conv2D(128, (3, 3), strides=1,padding="same", activation="relu"))
            model.add(MaxPooling2D((2, 2)))
            

            model.add(Dropout(0.5))


            model.add(Flatten())

            model.add(Dense(128, activation="relu"))
            model.add(Dropout(0.5))
            model.add(Dense(10, activation="softmax"))

            model.compile(loss='sparse_categorical_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])

            model.fit(X_Training, y_Training, batch_size=64,
                      epochs=20, validation_data=(X_Valid, Y_Valid) , shuffle=True)
            model.save("model_finish")


            # model = tf.keras.models.load_model("model_finish")
            # test_eval = model.evaluate(
            #     X_Test, y_Test, verbose=1, batch_size=16)

            # print("Test loss: " + str(test_eval[0]))
            # print("Test accuracy " + str(test_eval[1]))

        elif (unos == 'x' or unos == 'X'):
            print("Dovidjenja")
            sys.exit()

        else:
            print("Neispravan unos")
