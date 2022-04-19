#!/usr/bin/env python
# coding: utf-8

# # Hackaton Case: McRoberts

# Auteurs: Anna de Geeter, Daan Handgraaf, Chayenna Maas, Iris van de Velde

# Model geeft een score van 0.77972 op Kaggle!

# # 1 System setup

# **Inhoud case file**
# * Import and read data
# * Data description
# * Exploratory Data Analysis
# * Feature engineering
# * Model prediction

# In[1]:


#Importeer benodigden libraries
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier


# In[2]:


#Waar is notebook opgeslagen
print("Current working directory: {0}".format(os.getcwd()))


# # 2 Import and read data

# In[3]:


#Laad training dataset in
train = pd.read_csv('Train_Acitivity_Tracker_Labeled.csv')
train.head()


# In[4]:


train.info()


# In[5]:


train.describe()


# In[6]:


#Laad test dataset in
test = pd.read_csv('Test.csv')
test.head()


# In[7]:


test.info()


# In[8]:


test.describe()


# # 3 Data description

# #### Kolommen
# * Index - Index
# * 0 - x-waarde accelerator
# * 1 - y-waarde accelerator
# * 2 - z-waarde accelator
# * user - id voor gerbuikers
# * experiment - id voor experiment
# * index - tijdsindicator per experiment (data op 50Hz opgenomen)
# * Label - Label voor activiteit
# 
# #### Labels
# * 1 Lopen
# * 2 Lopen_Trap_op 
# * 3 Lopen_Trap_af 
# * 4 Zitten
# * 5 Staan
# * 6 Liggen
# * 7 Staan_Naar_Zitten
# * 8 Zitten_Naar_Staan
# * 9 Zitten_Naar_Liggen
# * 10 Liggen_Naar_Zitten
# * 11 Staan_Naar_Liggen
# * 12 Liggen_Naar_Staan

# ## 4 Exploratory Data Analysis

# In[9]:


plt.figure(figsize=(15,8))
sns.scatterplot(x=train['labels'], y=train['0'])

plt.title("Spreiding x-richting", weight="bold").set_fontsize(20)
plt.show()


# In[10]:


plt.figure(figsize=(15,8))

sns.scatterplot(x=train['labels'], y=train['1'])


plt.title("Spreiding y-richting", weight="bold").set_fontsize(20)
plt.show()


# In[11]:


plt.figure(figsize=(15,8))

sns.scatterplot(x=train['labels'], y=train['2'])

plt.title("Spreiding z-richting", weight="bold").set_fontsize(20)

plt.show()


# In[12]:


train[["labels", "0", "1", "2"]].groupby("labels", as_index=False).mean()


# In[13]:


Gemdf=train[["labels", "0", "1", "2"]].groupby("labels", as_index=False).mean()


# In[14]:


plt.figure(figsize=(15,7))

sns.set_theme(style="whitegrid", palette="pastel")
sns.barplot(x= "labels", y= '0', data=Gemdf)

plt.title("Gemiddelde beweging x", weight="bold").set_fontsize(20)

plt.show()


# In[15]:


plt.figure(figsize=(15,7))

sns.set_theme(style="whitegrid", palette="pastel")
sns.barplot(x= "labels", y= '1', data=Gemdf)

plt.title("Gemiddelde beweging y", weight="bold").set_fontsize(20)

plt.show()


# In[16]:


plt.figure(figsize=(15,7))

sns.set_theme(style="whitegrid", palette="pastel")
sns.barplot(x= "labels", y= '2', data=Gemdf)

plt.title("Gemiddelde beweging z", weight="bold").set_fontsize(20)
plt.show()


# ## 5 Feature engineering

# ### Train

# In[17]:


#Window functie 
train['xRolling']=train['0'].rolling(10).mean()
train['yRolling']=train['1'].rolling(10).mean()
train['zRolling']=train['2'].rolling(10).mean()


# In[18]:


#Vervang NaN waardes
train['xRolling'].fillna(train['xRolling'].mean(), inplace = True)
train['yRolling'].fillna(train['yRolling'].mean(), inplace = True)
train['zRolling'].fillna(train['zRolling'].mean(), inplace = True)


# In[51]:


train['vorige x'] = train['0'].shift(50)
train['vorige y'] = train['1'].shift(50)
train['vorige z'] = train['2'].shift(50)

train_test=train.iloc[50:100]
train_test=train_test.reset_index()

train['vorige x'].fillna(train_test['0'], inplace=True)
train['vorige y'].fillna(train_test['1'], inplace=True)
train['vorige z'].fillna(train_test['2'], inplace=True)


# In[20]:


train['A/P/T']=train['labels']

train.loc[(train['A/P/T'] <= 3), 'A/P/T'] = 1
train.loc[(train['A/P/T'] > 3) & (train['A/P/T'] <= 5), 'A/P/T'] = 2
train.loc[(train['A/P/T'] == 6 ), 'A/P/T']= 3
train.loc[(train['A/P/T'] > 6 ), 'A/P/T']= 4


# ### Test

# In[22]:


test['xRolling']=test['0'].rolling(10).mean()
test['yRolling']=test['1'].rolling(10).mean()
test['zRolling']=test['2'].rolling(10).mean()


# In[23]:


test['xRolling'].fillna(test['xRolling'].mean(), inplace = True)
test['yRolling'].fillna(test['yRolling'].mean(), inplace = True)
test['zRolling'].fillna(test['zRolling'].mean(), inplace = True)


# In[24]:


test['vorige x'] = test['0'].shift(50)
test['vorige y'] = test['1'].shift(50)
test['vorige z'] = test['2'].shift(50)

test_test=test.iloc[50:100]
test_test=test_test.reset_index()

test['vorige x'].fillna(test_test['0'], inplace=True)
test['vorige y'].fillna(test_test['1'], inplace=True)
test['vorige z'].fillna(test_test['2'], inplace=True)


# # 6 Model prediction

# ### 6.1 Sub groep

# In[26]:


#definieer je variabelen/inputs
X_sub_train=train.drop(labels=['Unnamed: 0','user','experiment','index','labels', 'A/P/T'], axis=1)
y_sub_train=train['A/P/T']


# In[27]:


X_sub_test=test.drop(labels=['user','experiment','Index'], axis=1)


# In[28]:


#predict eerst de sub groep voor de test dataset
#Creër een Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train het model, gebruik de training sets
clf.fit(X_sub_train, y_sub_train)

y_sub_pred=clf.predict(X_sub_test)


# In[29]:


#Voeg voorspelde label toe aan test dataset
test['A/P/T']=y_sub_pred


# ### 6.2 Label

# In[30]:


# #definieer je variabelen/inputs
# X_train=train.drop(labels=['Unnamed: 0', 'user','experiment','index', 'labels'], axis=1)
# y_train=train['labels']

# X_test=test.drop(labels=['Index', 'user', 'experiment'], axis=1)


# In[31]:


# #predict eerst sub groep A labels voor de test dataset
# #Creër een Gaussian Classifier
# clf=RandomForestClassifier(n_estimators=100)

# #Train het model, gebruik de training sets
# clf.fit(X_train, y_train)

# y_pred=clf.predict(X_test)


# In[32]:


# test['labels']=y_pred


# In[33]:


# submission=test[['Index', 'labels']]
# submission.to_csv('submissionV1.csv', index=False)


# #### Eerst sorteren per sub groep

# In[34]:


#Sorteer train set op subgroep
# A: Actief bewegen, P:Passief bewegen, T: in transit
sub_A_train=train[train['A/P/T']==1]
sub_P_train=train[train['A/P/T']==2]
sub_O_train=train[train['A/P/T']==3]
sub_T_train=train[train['A/P/T']==4]


# In[35]:


#Sorteer test set op subgroep
sub_A_test=test[test['A/P/T']==1]
sub_P_test=test[test['A/P/T']==2]
sub_O_test=test[test['A/P/T']==3]
sub_T_test=test[test['A/P/T']==4]


# #### 6.2.1 Model sub A

# In[36]:


#definieer je variabelen/inputs
X_A_train=sub_A_train.drop(labels=['Unnamed: 0', 'user','experiment','index', 'labels'], axis=1)
y_A_train=sub_A_train['labels']

X_A_test=sub_A_test.drop(labels=['Index', 'user', 'experiment'], axis=1)


# In[37]:


#predict eerst sub groep A labels voor de test dataset
#Creër een Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train het model, gebruik de training sets
clf.fit(X_A_train, y_A_train)

y_A_pred=clf.predict(X_A_test)


# In[38]:


#Voeg voorspelde label toe aan test dataset
sub_A_test['labels']=y_A_pred


# #### 6.2.2 Model sub P

# In[39]:


#definieer je variabelen/inputs
X_P_train=sub_P_train.drop(labels=['Unnamed: 0', 'user','experiment','index', 'labels'], axis=1)
y_P_train=sub_P_train['labels']

X_P_test=sub_P_test.drop(labels=['Index', 'user', 'experiment'], axis=1)


# In[40]:


#predict eerst sub groep P labels voor de test dataset
#Creër een Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train het model, gebruik de training sets
clf.fit(X_P_train, y_P_train)

y_P_pred=clf.predict(X_P_test)


# In[41]:


#Voeg voorspelde label toe aan test dataset
sub_P_test['labels']=y_P_pred


# #### 6.2.2 Model sub O

# In[44]:


#Voeg label toe aan test dataset
sub_O_test['labels']=6


# #### 6.2.3 Model sub T

# In[45]:


#definieer je variabelen/inputs
X_T_train=sub_T_train.drop(labels=['Unnamed: 0', 'user','experiment','index', 'labels'], axis=1)
y_T_train=sub_T_train['labels']

X_T_test=sub_T_test.drop(labels=['Index', 'user', 'experiment'], axis=1)


# In[46]:


#predict eerst sub groep T labels voor de test dataset
#Creër een Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train het model, gebruik de training sets
clf.fit(X_T_train, y_T_train)

y_T_pred=clf.predict(X_T_test)


# In[47]:


#Voeg voorspelde label toe aan test dataset
sub_T_test['labels']=y_T_pred


# #### 6.2.4 Samenvoegen datasets

# In[48]:


submission=pd.concat([sub_A_test, sub_P_test, sub_O_test,sub_T_test])
submission=submission.sort_index()

sub_hack=submission[['Index', 'labels']]

sub_hack


# In[49]:


#converteer dataframe naar csv file
sub_hack.to_csv('sub_hack_V16.csv', index=False)

