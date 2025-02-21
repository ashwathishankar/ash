#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np


# In[3]:


import pandas as pd


# In[4]:


from sklearn.model_selection import train_test_split


# In[5]:


from sklearn.metrics import confusion_matrix


# In[6]:


from sklearn.metrics import accuracy_score


# In[7]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# In[8]:


import numpy
import pandas 


# In[9]:


hair=pandas.read_csv('iphone_purchase_records.csv')
hair.info()


# In[10]:


hair.shape


# In[11]:


hair.info()


# In[12]:


hair.describe()


# In[13]:


X=hair.drop(['Gender'],axis=1)
y=hair['Age']


# In[15]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=42,stratify=y)


# In[16]:


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)


# In[17]:


y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Display the results
print(f'Accuracy: {accuracy:.2f}')
print('\nConfusion Matrix:')
print(conf_matrix)


# In[18]:


plt.figure(figsize=(10, 8))
sns.scatterplot(x='Age', y='Salary', hue='Purchase Iphone', size='Purchase Iphone', data=hair, palette='viridis', sizes=(50, 200))

# Customize the plot
plt.title('iPhone Purchases Scatter Plot')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.legend(title='Purchase Iphone')

# Show the plot
plt.show()


# In[19]:


sns.countplot(hair['Purchase Iphone'])


# In[21]:


plt.hist(hair['Purchase Iphone'])
plt.title('Histogram')
plt.show()


# In[ ]:




