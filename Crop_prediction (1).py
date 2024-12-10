#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
classes = os.listdir('../datasets/Crop prediction/crop_images')
classes


# In[7]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[8]:


data_gen = ImageDataGenerator(horizontal_flip=True,vertical_flip=True,zoom_range=0.2,shear_range=0.2,rescale=1/255)


# In[9]:


train_data = data_gen.flow_from_directory('../datasets/Crop prediction/crop_images',target_size=(224, 224),batch_size=8)


# In[10]:


train_data.class_indices


# In[11]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten
from tensorflow.keras.applications import VGG16


# In[12]:


vgg16 = VGG16(include_top=False,input_shape=(224,224,3))


# In[13]:


vgg16.summary()


# In[14]:


for layer in vgg16.layers:
    layer.trainable = False # 2) trainable parameter off


# In[15]:


final_layer = Dense(5,activation='softmax')(Flatten()(vgg16.output)) # 3)VGG16 output alyer


# In[16]:


from tensorflow.keras.models import Model


# In[17]:


model_vgg16 = Model(inputs = vgg16.input, outputs = final_layer) # 4) attaching output layer


# In[18]:


model_vgg16.compile(loss='categorical_crossentropy', metrics=['accuracy']) #model build


# In[19]:


model_vgg16.summary()


# In[20]:


model_vgg16.fit(train_data,epochs=5)


# In[ ]:





# In[ ]:




