#!/usr/bin/env python
# coding: utf-8

# ## 使用Keres MNIST Fashion辨識穿著

# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils


# In[91]:


(x_train,y_train),(x_test,y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train.shape,y_test.shape,x_test.shape,y_test.shape


# In[92]:


x_train[0]


# In[93]:


y_train[:20]
# 0 T-shirt/top
# 1 Trouser
# 2 Pullover
# 3 Dress
# 4 Coat
# 5 Sandal
# 6 Shirt
# 7 Sneaker
# 8 Bag
# 9 Ankle boot


# In[94]:


img = x_train[0].reshape(28, 28)
plt.imshow(img, cmap='Greys')


# In[95]:


# 將非0的數字轉為1，顯示第1張圖片
data = x_train[0].copy()
data[data>0]=1

# 將轉換後二維內容顯示出來，隱約可以看出數字為 5
text_image=[]
for i in range(data.shape[0]):
    text_image.append(''.join(str(data[i])))
text_image


# In[96]:


data.shape[0]


# In[97]:


# 將 training 的 label 進行 one-hot encoding，例如數字 7 經過 One-hot encoding 轉換後是 0000000100，即第8個值為 1
y_TrainOneHot = np_utils.to_categorical(y_train) 
y_TestOneHot = np_utils.to_categorical(y_test) 


# In[98]:


# 將 training 的 input 資料轉為2維
x_train_2D = x_train.reshape(len(x_train), 28*28).astype('float32')  
x_test_2D = x_test.reshape(len(x_test), 28*28).astype('float32') 


# In[99]:


x_Train_norm = x_train_2D/255
x_Test_norm = x_test_2D/255


# In[100]:


# 建立簡單的線性執行的模型
model = tf.keras.models.Sequential()
# Add Input layer, 隱藏層(hidden layer) 有 320個輸出變數
model.add(tf.keras.layers.Dense(units=320, input_dim=784, kernel_initializer='normal', activation='relu')) 
model.add(tf.keras.layers.Dense(units=64, kernel_initializer='normal', activation='relu')) 
# Add output layer
model.add(tf.keras.layers.Dense(units=10, kernel_initializer='normal', activation='softmax'))

# 編譯: 選擇損失函數、優化方法及成效衡量方式
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 


# In[101]:


model.summary()


# In[102]:


# 進行訓練, 訓練過程會存在 train_history 變數中
train_history = model.fit(x=x_Train_norm, y=y_TrainOneHot, validation_split=0.2, epochs=10, batch_size=1000, verbose=2)  


# In[103]:


# 顯示訓練成果(分數)
scores = model.evaluate(x_Test_norm, y_TestOneHot)  
print(f"準確度 = {scores[1]*100.0:2.1f}")  


# In[104]:


#存模型
model.save('20200929fashion_mnist_keras.h5')


# In[105]:


#繪製模型
tf.keras.utils.plot_model(model, to_file='model.png')


# In[ ]:




