#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system(' pip install tensorflow')


# In[2]:


import tensorflow as tf 
tf.version


# In[4]:


get_ipython().system(' pip install pandas')


# In[5]:


get_ipython().system(' pip install numpy')


# In[6]:


get_ipython().system(' pip install  matplotlib')


# In[7]:


get_ipython().system(' pip install seaborn')


# In[9]:


get_ipython().system(' pip install sklearn')


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


# In[19]:


get_ipython().system(' pip install scikit-learn')


# In[20]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


# In[22]:


path=r'C:\Users\saisr\Downloads\NSE-TATAGLOBAL.csv'
frm=pd.read_csv(path)
frm.head()


# In[23]:


frm.info()


# In[24]:


frm.isnull().sum()


# In[25]:


df1=frm.reset_index()['Close']
#showing stats of data_frame
frm.describe()


# In[26]:


plt.plot(df1,color='b',label="Closing Price")
plt.legend(loc="upper right")
plt.title("Closing price")


# In[27]:


plt.plot(frm['Open'],color='g',label="Opening Price")
plt.legend(loc="upper right")
plt.title("Opening Price")


# In[28]:


sns.heatmap(frm.corr(numeric_only=True),annot=True)


# In[29]:


scale=MinMaxScaler(feature_range=(0,1))
df1=scale.fit_transform(np.array(df1).reshape(-1,1))
df1


# In[30]:


train_size=int(len(df1)*0.6)
test_size=len(df1)-train_size
train_data=df1[:train_size,:]
test_data=df1[train_size:len(df1),:1]
print(train_size,test_size)
print(train_data,test_data)


# In[31]:


def cr_dataset(dataset,time_step=1):
    dataX,dataY=[],[]
    for i in range(len(dataset)-time_step-1): 
        a=dataset[i:(i+time_step),0]
        dataX.append(a)
        dataY.append(dataset[i+time_step,0])
    return np.array(dataX),np.array(dataY)


# In[32]:


time_step=100
x_train,y_train=cr_dataset(train_data,time_step)
x_test,y_test=cr_dataset(test_data,time_step)


# In[33]:


print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)


# In[34]:


x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],1)


# In[35]:


#creating Stacked LSTM Model
model=Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(100,1)))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')


# In[36]:


model.summary()


# In[37]:


model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=100,batch_size=64,verbose=1)


# In[39]:


train_predict=model.predict(x_train)
test_predict=model.predict(x_test)


# In[40]:


import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict))


# In[41]:


## test data
math.sqrt(mean_squared_error(y_test,test_predict))


# In[42]:


plt.figure(figsize=(18,6))
plt.title("Stock Market Price Prediction")
plt.plot(frm['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Total Trade Quantity', fontsize=18)
plt.show()


# In[43]:


frm["Date"] = pd.to_datetime(frm.Date)
frm.index = frm['Date']

plt.figure(figsize=(20, 10))
plt.plot( frm["Open"], label='ClosePriceHist')


# In[44]:


plt.figure(figsize=(10,5))
plt.plot( frm['Date'])
plt.xlabel('Turnover (Lacks)', fontsize=15)
plt.ylabel('Total Trade Quantity', fontsize=15)
plt.show()


# In[46]:


frm["Turnover (Lacks)"] = pd.to_datetime( frm.Date)
frm.index =  frm['Turnover (Lacks)']

plt.figure(figsize=(20, 10))
plt.plot( frm["Turnover (Lacks)"], label='ClosePriceHist')
plt.show()


# In[47]:


sns.set(rc = {'figure.figsize': (20, 5)})
frm['Open'].plot(linewidth = 1,color='blue')


# In[48]:


frm.columns


# In[50]:


path=r'C:\Users\saisr\Downloads\NSE-TATAGLOBAL.csv'
frm=pd.read_csv(path)
frm.head()


# In[51]:


cols_plot = ['Open','High','Low','Last','Close']
axes = frm[cols_plot].plot(alpha = 1, figsize=(20, 30), subplots = True)

for ax in axes:
    ax.set_ylabel('Variation')


# In[ ]:




