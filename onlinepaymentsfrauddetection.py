#!/usr/bin/env python
# coding: utf-8

# In[15]:


pip install nbconvert


# In[16]:


import numpy as np 
import pandas as pd
data = pd.read_csv("C:\\Users\\USER\\Desktop\\data science\\onlinefraud.csv")


# In[17]:


data.isnull().sum()


# In[18]:


print(data.head())


# In[19]:


print(data.type.value_counts())


# In[20]:


type = data["type"].value_counts()
transactions = type.index
quantity = type.values

import plotly.express as px
figure = px.pie(data, 
             values=quantity, 
             names=transactions,hole = 0.5, 
            title="Distribution of Transaction Type")
figure.show()


# In[6]:


correlation = data.corr()
print(correlation["isFraud"].sort_values(ascending= False))


# In[11]:


data["type"] = data["type"].map({"CASH_out":1,"PAYMENT":2,"CASH_IN": 3, "TRANSFER": 4, "DEBIT":5})
data["isFraud"] = data["isFraud"].map({0: "NO Fraud", 1: "Fraud"})
print(data.head())


# In[12]:


# splitting the data
data.dropna(inplace = True)
from sklearn.model_selection import train_test_split
x = np.array(data[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]])
y = np.array(data[["isFraud"]])


# In[13]:


# training a machine learning model
from sklearn.tree import DecisionTreeClassifier
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.10, random_state=42)
model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)
print(model.score(xtest, ytest))


# In[10]:


# prediction
#features = [type, amount, oldbalanceOrg, newbalanceOrig]
features = np.array([[4, 9000.60, 9000.60, 0.0]])
print(model.predict(features))


# In[ ]:




