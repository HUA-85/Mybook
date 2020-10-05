#!/usr/bin/env python
# coding: utf-8

# ## Cat or Not

# In[1]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from PIL import Image
from random import shuffle, choice
import numpy as np
import os
import keras
from matplotlib import pyplot as plt


# In[3]:


def label_img(name):   
    if name == 'cats': return np.array([1, 0])
    elif name == 'notcats' : return np.array([0, 1])
    
def load_data(IMAGE_DIRECTORY):
  print("Loading images...")
  data = []
  directories = next(os.walk(IMAGE_DIRECTORY))[1]
    
  for dirname in directories:
    print("Loading {0}".format(dirname))
    file_names = next(os.walk(os.path.join(IMAGE_DIRECTORY, dirname)))[2] 
    
    for i in range(200):
      image_name = choice(file_names)
      #載入圖片路徑
      image_path = os.path.join(IMAGE_DIRECTORY, dirname, image_name)
      label = label_img(dirname)
      if "DS_Store" not in image_path:
        img = Image.open(image_path)
        img = img.convert('L') #轉換灰色圖像
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS) 
        data.append([np.array(img),label])
  
  return data

def create_model():
  model = Sequential()
  model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(BatchNormalization())
  model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(BatchNormalization())
  model.add(Conv2D(filters=128, kernel_size=(5, 5), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(BatchNormalization())
  model.add(Conv2D(filters=256, kernel_size=(5, 5), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(BatchNormalization())
  model.add(Conv2D(filters=64, kernel_size=(5, 5),  activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(BatchNormalization())
  model.add(Dropout(0.2))#減少過度擬合
  model.add(Flatten())
  model.add(Dense(256, activation='relu'))
  model.add(Dropout(0.2))
  model.add(Dense(2, activation = 'softmax'))

  return model


# In[4]:


IMAGE_SIZE = 256
model = create_model()
model.summary()  


# In[5]:


IMAGE_DIRECTORY = './data/training_set' 
training_data=load_data(IMAGE_DIRECTORY)
training_images = np.array([i[0] for i in training_data]).reshape(-1, IMAGE_SIZE, IMAGE_SIZE,1)
training_labels = np.array([i[1] for i in training_data])


print('creating model')
model = create_model()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print('training model')
train_history = model.fit(training_images, training_labels, batch_size=50, epochs=10, verbose=1)
model.save("model3.h5")


# In[6]:


# 模型載入
import tensorflow as tf
from tensorflow import keras
print('Loading model...')
model1 = tf.keras.models.load_model('model3.h5') 


# In[7]:


IMAGE_DIRECTORY = './data/test_set'
testing_data=load_data(IMAGE_DIRECTORY)
testing_images = np.array([i[0] for i in testing_data]).reshape(-1, IMAGE_SIZE, IMAGE_SIZE,1)
testing_labels = np.array([i[1] for i in testing_data])
test_history = model1.fit(training_images, training_labels, batch_size=50, epochs=10, verbose=1)


# In[8]:


plt.plot(train_history.history['loss'],'r')
plt.plot(test_history.history['loss'],'g')
plt.show()


# In[9]:


plt.plot(train_history.history['accuracy'],'r')
plt.plot(test_history.history['accuracy'],'g')
plt.show()


# In[10]:


# 顯示訓練成果(分數)
loss, acc = model1.evaluate(testing_images, testing_labels, verbose=1)
print("accuracy: {0}".format(acc * 100))


# In[11]:


#繪製模型
tf.keras.utils.plot_model(model, to_file='model2.png')


# In[ ]:




