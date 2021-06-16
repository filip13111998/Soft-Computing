import numpy as np
from cv2 import cv2
import os
import random
import pickle
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from PIL import Image
import statistics
from keras.preprocessing import image
# def augmentation():
#     datagen = ImageDataGenerator(
#         rotation_range=20,
#         width_shift_range=0.2,
#         height_shift_range=0.2,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True,
#         fill_mode='reflect')

#     path = 'C:/Users/vaske/Desktop/ADR-copy/raw-img/scoiattolo/'

#     slike = os.listdir(path)
#     # print(slike)

#     new_width = 300
#     new_height = 300
#     # the .flow() command below generates batches of randomly transformed images
#     # and saves the results to the `preview/` directory
#     for image in slike:

#         img = load_img(path+image)  # this is a PIL image
#         img = img.resize((new_width, new_height), Image.ANTIALIAS)
#         x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
#         x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

#         i = 0
#         for batch in datagen.flow(x, batch_size=1,
#                                     save_to_dir=path, save_prefix='aug', save_format='jpg'):
            
#             if i > 2:
#                 break  # otherwise the generator would loop indefinitely

#             i += 1

img_x = []
img_y = []
mediana_x = []
mediana_y = []

def create_training_data(DATA_DIR):
    training_data = []
    validation_data = []
    test_data = []
    for animal_type in animal_species:
        path = os.path.join(DATA_DIR, animal_type)
        class_num = animal_species.index(animal_type)

        animal_counter = 0
        animals_img = os.listdir(path)



        for i in range(30):
            random.shuffle(animals_img)
      

        print("Number of pics: " , len(os.listdir(path)))

        # for img in animals_img:
        #     img_new = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        #     img_x.append(len(img_new))
        #     img_y.append(len(img_new[0]))
            
        # x_size = statistics.median(img_x)
        # y_size = statistics.median(img_y)

        # print(x_size)
        # print(y_size)

        # mediana_x.append(x_size)
        # mediana_y.append(y_size)

    

  
        for img in animals_img:
            animal_counter+=1
            if(animal_counter < len(animals_img)*0.7):
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                print(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])

            elif((animal_counter >= len(animals_img)*0.7) and (animal_counter < len(animals_img)*0.85)):
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                validation_data.append([new_array, class_num])
                
            else:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                test_data.append([new_array, class_num])

    # x_size = statistics.median(mediana_x)
    # y_size = statistics.median(mediana_y)

    return training_data,validation_data , test_data


# def create_validation_data(DATA_DIR):
#     validation_data = []
#     for animal_type in animal_species:
#         path = os.path.join(DATA_DIR, animal_type)
#         class_num = animal_species.index(animal_type)

#         animal_counter = 0

#         print("Number of valaid: " , len(os.listdir(path))*0.8)

#         for img in os.listdir(path):
            

#             animal_counter+=1
            
#             if( (animal_counter >= len(os.listdir(path))*0.8) and (animal_counter < len(os.listdir(path))*0.9) ):
#                 img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
#                 new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
#                 validation_data.append([new_array, class_num])

#     return validation_data

# def create_test_data(DATA_DIR):
#     test_data = []
#     for animal_type in animal_species:
#         path = os.path.join(DATA_DIR, animal_type)
#         class_num = animal_species.index(animal_type)

#         animal_counter = 0
#         # pic_counter = 0

#         print("Number of test: " , len(os.listdir(path)))

#         for img in os.listdir(path):
#             animal_counter +=1

#             # if animal_counter > 800:
                
#             if animal_counter >= len(os.listdir(path))*0.8:
#                 img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
#                 new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
#                 test_data.append([new_array, class_num])

#                 # pic_counter +=1
#                 # if(pic_counter >= 20):
#                 #     break
                
                

#     return test_data


if __name__ == '__main__':
    # DATA_DIR_CANE = "raw-img/cane"
    # DATA_DIR_CAVALLO = "raw-img/cavallo"
    # DATA_DIR_ELEFANTE = "raw-img/elefante"
    # DATA_DIR_FARFALLA = "raw-img/farfalla"
    # DATA_DIR_GALLINA = "raw-img/gallina"
    # DATA_DIR_GATTO = "raw-img/gatto"
    # DATA_DIR_MUCCA = "raw-img/mucca"
    # DATA_DIR_PECORA = "raw-img/pecora"
    # DATA_DIR_RAGNO = "raw-img/ragno"
    # DATA_DIR_SCOIATTOLO = "raw-img/scoiattolo"
    
    
    
    
    dir = ""

    #               dog     horse       elephant    butterfly   rooster     cat     cow...      sheep...    spider..    squirrel
    animal_species = ["pas", "konj", "slon" , "leptir" , "kokoska" , "macka" , "krava" , "ovca" , "pauk" , "veverica"] #,  
    IMG_SIZE = 300

    trainig_data ,validation_data , test_data= create_training_data(dir)
    # validation_data = create_training_data(dir)
    # test_data = create_test_data(dir)

    for i in range(30):
        random.shuffle(trainig_data)
        random.shuffle(validation_data)
        random.shuffle(test_data)

 


    X_Training = []
    y_Training = []

    for features, label in trainig_data:
        X_Training.append(features)
        y_Training.append(label)
        # print(label)

    X_Training = np.array(X_Training).reshape(-1, IMG_SIZE, IMG_SIZE, 1)


    picke_out = open("X_Training.pickle", "wb") 
    pickle.dump(X_Training, picke_out)
    picke_out.close()

    picke_out = open("y_Training.pickle", "wb")
    pickle.dump(y_Training, picke_out)
    picke_out.close()

    X_Valid = []
    Y_Valid = []

    for features, label in validation_data:
        X_Valid.append(features)
        Y_Valid.append(label)

    X_Valid = np.array(X_Valid).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    #x = np.array(X_Training).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    picke_out = open("X_Valid.pickle", "wb") 
    pickle.dump(X_Valid, picke_out)
    picke_out.close()

    picke_out = open("y_Valid.pickle", "wb")
    pickle.dump(Y_Valid, picke_out)
    picke_out.close()


    X_Test = []
    y_Test = []

    for features, label in test_data:
        X_Test.append(features)
        y_Test.append(label)

    X_Test = np.array(X_Test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    picke_out = open("X_Test.pickle", "wb")
    pickle.dump(X_Test, picke_out)
    picke_out.close()

    picke_out = open("y_Test.pickle", "wb")
    pickle.dump(y_Test, picke_out)
    picke_out.close()
    # augmentation()