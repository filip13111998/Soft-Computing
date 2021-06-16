#!/usr/bin/env python
# coding: utf-8

# In[5]:



from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
from PIL import Image

datagen = ImageDataGenerator(
        rotation_range=25,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='reflect')
folder = 'ker1/'
slike = os.listdir(folder)

print(len(slike))
new_width = 300
new_height = 300
cnt = len(slike) 
# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
uslov = False
for image in slike:
    #print('usao')
    #if 'false' in image:
    
    img = load_img(folder+'/'+image)  # this is a PIL image
    img = img.resize((new_width, new_height), Image.ANTIALIAS)
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
     
    
    print(cnt)
     
    i = 0
    for batch in datagen.flow(x, batch_size=1, save_to_dir=folder, save_prefix='aug', save_format='jpg'):
        i += 1
        if i > 10:
            #cnt -= 1
            break  # otherwise the generator would loop indefinitel
    #     if cnt < 5001:
    #         cnt += 1
    #     else:
    #          uslov = True
        
    # if(uslov):
    #     break
            
    #if cnt > 5000:
        #print("USAO")
        #break   

# In[ ]:





# In[ ]:




