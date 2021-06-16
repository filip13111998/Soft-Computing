#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import cv2 # OpenCV
from sklearn.svm import SVC # SVM klasifikator
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier # KNN
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
from random import randrange
import random
import pickle
import datetime
import joblib
from sklearn.decomposition import PCA
from matplotlib import pyplot
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image



def load_image(path):

    img = image.load_img(path, target_size=(300, 300), grayscale=True)
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    return img_tensor


def display_image(image):
    plt.imshow(image, 'gray')


def sliding_window(image, step_size, window_size=(300, 300)):
    image = image.reshape(300,300)
    # print(image.shape)
    # print(image)
    # image = image.reshape(300, 300)
    best_scores = []
    best_windows = []
    velicina = []
    sizes = [0.8, 0.85, 0.9, 0.92, 0.95]
    # sizes = [0.5, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.92, 0.95]
    y = 0
    x = 0
    IMG_SIZE = 300
    isecci = []
    resolution = []
    size_slike = []
    indeksi = []
    model = load_model("model_finish")
    test_eval = []
    # rez = model.predict(image)
    # print(rez)
    for size in sizes:
        for y in range(0, image.shape[0], step_size):
            # print("USAO")
            for x in range(0, image.shape[1], step_size):
                this_window = (x, y) # zbog formata rezultata
               
                window = image[y:int(y+window_size[0]*size), x:int(x+window_size[1]*size)]
                # print(window)
                if window.shape == (int(window_size[0]*size), int(window_size[1]*size)):

    #                 score = classify_window(window)
                    cropped = image[y:y+int(image.shape[1]*size), x:x+int(image.shape[0]*size)]
                    dimenzije = cropped
                    # print(cropped.shape)
                    try:
                        cropped = cv2.resize(cropped, (IMG_SIZE, IMG_SIZE), interpolation = cv2.INTER_AREA) 
                        # cropped = np.expand_dims(cropped, axis=0)  
                    except:
                        break  
                    # print(cropped.shape)
                    # cv2.imshow("bla", cropped)
                    # cv2.waitKey(0) 
                    # cv2.show()
                    # print(cropped.shape)
                    # isecci.append(cropped)
                    rez = model.predict(cropped.reshape(-1, IMG_SIZE, IMG_SIZE, 1))
                    test_eval.append(rez[0])
                    resolution.append((int(dimenzije.shape[0]), int(dimenzije.shape[1])))
                    size_slike.append(int(image.shape[1]*size))


    for i in range(len(test_eval)):

        index = 0
        for j in range(len(test_eval[i])):
            maximum = max(test_eval[i])
            if test_eval[i][j] == maximum:  
                index = j

        indeksi.append(index)

        maks = max(test_eval[i])
        duplicateFrequencies = {}
        for k in set(indeksi):
            duplicateFrequencies[k] = indeksi.count(k)
        # print(duplicateFrequencies)
        #print(maks)
        if(maks > 0.6) :
        #and (int(image.shape[0]*size) +  this_window[0] ) < int(image.shape[0]) )
           # print(maks)
            best_scores.append(maks)
            best_windows.append(resolution[i])
            prozor = int(size_slike[i])#*size
            velicina.append(prozor)
            #print(prozor)
            #print(score)
        
    
    
    # print(indeksi) 
    # print(velicina)
    return best_scores, best_windows, velicina, indeksi


def non_maximum_suppression(bounding_boxes, confidence_score, threshold, velicina, indexi):
    # Ako nema box-eva
    if len(bounding_boxes) == 0:
        return [], []

    boxes = np.array(bounding_boxes)

    #Koordinate box-a
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    score = np.array(confidence_score)


    picked_boxes = []
    picked_score = []
    picked_window = []
    picked_index = []
    #print(len(indexi))

    #Odredjujemo povrsinu
    povrsina = (end_x - start_x + 1) * (end_y - start_y + 1)

    #Sortiranje po velicini prozora
    order = np.argsort(velicina)
    # order = velicina
    maks = max(velicina)

  
    while len(order) > 0:
        # index najveceg confidence score-a
        index = order[-1]
        # print(index)
        # bira box-eve sa najvecim score-om
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])
        picked_window.append(velicina[index])
        picked_index.append(indexi[index])
        
        # Racuna koordinate od  intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # racuna povrsinu od intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        preklapanje = w * h

        # racuna odnos izmedju preklapanja i unije
        ratio = preklapanje / (povrsina[index] + povrsina[order[:-1]] - preklapanje)

        left = np.where(ratio < threshold)
        order = order[left]

    return picked_boxes, picked_score, picked_window, picked_index



#-----------------------------------------
slika = 'horse49-1.png'
#-----------------------------------------

# Bounding boxes
bounding_boxes = [] #(61, 20, 101, 120), (63, 140, 103, 240)
confidence_score = []
threshold = 0.03
step = 15


# start = datetime.datetime.now()

#Ucitavanje slike
itest = load_image(slika)

#Sliding window
score, score_window, velicina, indexi = sliding_window(itest, step_size=step)


#Odredjivanje box-eva
for i in range(len(score_window)):
     bounding_boxes.append((score_window[i][0]-int(score_window[i][0]*velicina[i]/300), score_window[i][1]-int(score_window[i][1]*velicina[i]/300), 
                           score_window[i][0], score_window[i][1]))

# print(bounding_boxes)

for scr in score:
    confidence_score.append(scr)

# print(bounding_boxes)



picked_boxes, picked_score, picked_window, picked_index = non_maximum_suppression(bounding_boxes, confidence_score, threshold, velicina, indexi)

# print(picked_boxes)
# print(picked_score)

# finish = datetime.datetime.now()
# print(finish-start)
# print(picked_index)


# start_point = [i[0] for i in picked_boxes[0]]
prvi = 0
drugi = 0
treci = 0
cetvrti = 0
brojac = 0

# itest = load_image('proba/ovca118.jpg')
slikaRGB = cv2.imread(slika)
# itest1 = cv2.cvtColor(itest1, cv2.COLOR_RGB2GRAY)
# itest1 = cv2.imread(slika, cv2.IMREAD_GRAYSCALE)
# dim = (itest.shape[1], itest.shape[0])
                        #(300,300)
# itest1 = cv2.resize(itest1, dim, interpolation = cv2.INTER_AREA)
itest1 = load_image(slika)
itest1 = itest1.reshape(300,300)
y=0
x=0
zivotinje = ["pas", "konj", "veverica", "leptir", "slon" ,  "kokoska" , "macka" , "krava" , "ovca" , "pauk"]  
crop_images = []
for box in picked_boxes:
    for num in box:
#         print(num)
        if(brojac == 0):
            prvi = num
        elif(brojac == 1):
            drugi = num
        elif(brojac == 2):
            treci = num
        else:
            cetvrti = num
            
            cropped = itest1[y:prvi+treci, x:drugi+cetvrti]
            #cropped = itest1[y:prvi+cetvrti, x:drugi+treci]
            crop_images.append(cropped)
            # cropped.reshape(300, 300)
            model = load_model("model_finish")
            test_eval = model.predict(cropped.reshape(-1, 300, 300, 1))
            
            itest1 = cv2.rectangle(slikaRGB, (prvi, drugi), (treci, cetvrti), (0,255,0) , 6) 
            cropped = slikaRGB[y:prvi+treci, x:drugi+cetvrti]

            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (prvi+5, prvi+25)
            fontScale = 0.5
            fontColor = (255,255,255)
            lineType = 2
            # print(picked_score[0])
            ispis = zivotinje[picked_index[0]]+ " " +str(picked_score[0])
            cv2.putText(cropped, ispis, 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                lineType)
            print(test_eval)
            cv2.imshow(ispis,cropped)
            cv2.waitKey(0) 
            # pyplot.imshow(cropped, cmap='gray')
            brojac = -1
        
        brojac = brojac + 1
            
# pyplot.show()        
#print(len(crop_images))



# itest = cv2.rectangle(itest, (150, 0), (500, 350), (0,255,0) , 6) 
#display_image(itest1)





