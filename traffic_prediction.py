#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import cv2
from sklearn.model_selection import train_test_split


# In[2]:


from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import layers,models


# In[3]:


#function to load and preprocess images
def load_and_preprocess_data(folder_path,label):
    images=[]
    labels=[]
    
    for filename in os.listdir(folder_path):
        img_path=os.path.join(folder_path,filename)
        img=cv2.imread(img_path)
        # Check if the image is not None
        if img is not None:
            img = cv2.resize(img, (64, 64))
            img = img / 255.0
            images.append(img)
            labels.append(label)
    return images,labels


# In[4]:


traffic_labels={
    'accident':0,
    'dense_traffic':1,
    'fire':2,
    'sparse_traffic':3,
}


# In[5]:


all_images=[]
all_labels=[]


# In[6]:


for traffic,label in traffic_labels.items():
    folder_path=os.path.join('dataset',traffic)
    images,labels=load_and_preprocess_data(folder_path,label)
    all_images.extend(images)
    all_labels.extend(labels)
    
#convert to Numpy arrrays
all_images=np.array(all_images)
all_labels=np.array(all_labels)


# In[7]:


all_labels


# In[8]:


all_images


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(all_images, all_labels, test_size = 0.2)


# In[14]:


batch_size = 16
nb_epochs = 20
nb_filters = 32
nb_pool = 2
nb_conv = 3


# In[15]:


from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model


# In[16]:


base_model = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
for layer in base_model.layers:
    layer.trainable = False


# In[17]:


# Create a new model on top
traffic_model_vgg16 = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(4, activation='softmax')
])


# In[18]:


traffic_model_vgg16.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
traffic_model_vgg16.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epochs, validation_data=(X_test, y_test))


# In[19]:


traffic_model_vgg16.save('traffic_model_vgg16.h5')


# In[20]:





# In[25]:





# In[32]:




# In[33]:




# In[ ]:




