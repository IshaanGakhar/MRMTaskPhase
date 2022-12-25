#!/usr/bin/env python
# coding: utf-8

# In[41]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# In[42]:


cars = pd.read_csv('CarPrice_Assignment.csv')


# In[43]:


#Splitting company name from CarName column
CompanyName = cars['CarName'].apply(lambda x : x.split(' ')[0])
cars.insert(3,"CompanyName",CompanyName)
cars.drop(['CarName'],axis=1,inplace=True)
cars.head()


# In[44]:


cars.CompanyName = cars.CompanyName.str.lower()

def replace_name(a,b):
    cars.CompanyName.replace(a,b,inplace=True)

replace_name('maxda','mazda')
replace_name('porcshce','porsche')
replace_name('toyouta','toyota')
replace_name('vokswagen','volkswagen')
replace_name('vw','volkswagen')

cars.CompanyName.unique()


# In[45]:


#Fuel economy from transpoco.com
cars['fueleconomy'] = (0.55 * cars['citympg']) + (0.45 * cars['highwaympg'])


# In[46]:


#Binning the Car Companies based on avg prices of each Company.
cars['price'] = cars['price'].astype('int')
temp = cars.copy()
table = temp.groupby(['CompanyName'])['price'].mean()
temp = temp.merge(table.reset_index(), how='left',on='CompanyName')
bins = [0,10000,20000,40000]
cars_bin=['Budget','Medium','Highend']
cars['carsrange'] = pd.cut(temp['price_y'],bins,right=False,labels=cars_bin)


# In[47]:


#significant variables
cars_lr = cars[['price', 'fueltype', 'aspiration','carbody', 'drivewheel','wheelbase','curbweight', 'enginetype', 'cylindernumber', 'enginesize', 'boreratio','horsepower', 'fueleconomy', 'carlength','carwidth', 'carsrange']]


# In[57]:


# Defining the map function [using dummy variables as switches in equations]
def dummies(x,df):
    temp = pd.get_dummies(df[x], drop_first = True)
    df = pd.concat([df, temp], axis = 1)
    df.drop([x], axis = 1, inplace = True)
    return df
# Applying the function to the cars_lr

cars_lr = dummies('fueltype',cars_lr)
cars_lr = dummies('aspiration',cars_lr)
cars_lr = dummies('carbody',cars_lr)
cars_lr = dummies('drivewheel',cars_lr)
cars_lr = dummies('enginetype',cars_lr)
cars_lr = dummies('cylindernumber',cars_lr)
cars_lr = dummies('carsrange',cars_lr)


# In[56]:


np.random.seed(0)
df_train, df_test = train_test_split(cars_lr, train_size = 0.7, test_size = 0.3, random_state = 100)


# In[54]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
num_vars = ['wheelbase', 'curbweight', 'enginesize', 'boreratio', 'horsepower','fueleconomy','carlength','carwidth','price']
df_train[num_vars] = scaler.fit_transform(df_train[num_vars])


# In[51]:


#Dividing data into X and y variables [vectors]
y_train = df_train.pop('price')
X_train = df_train


# In[52]:


#RFE stansd for Recursive Feature Elimination - used to choose features which are actually affecting the result - remove multicollinearty
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm 
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[53]:


lm = LinearRegression()
lm.fit(X_train,y_train)
rfe = RFE(lm, 10)
rfe = rfe.fit(X_train, y_train)
X_train_rfe = X_train[X_train.columns[rfe.support_]]


# In[ ]:


X_train_rfe = X_train[X_train.columns[rfe.support_]]


# In[ ]:


def build_model(X,y):
    X = sm.add_constant(X) #Adding the constant
    lm = sm.OLS(y,X).fit() # fitting the model
    print(lm.summary()) # model summary
    return X
    
def checkVIF(X):
    vif = pd.DataFrame()
    vif['Features'] = X.columns
    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif['VIF'] = round(vif['VIF'], 2)
    vif = vif.sort_values(by = "VIF", ascending = False)
    return(vif)


# In[ ]:


X_train_new = build_model(X_train_rfe,y_train)


# In[ ]:


X_train_new = X_train_rfe.drop(["twelve"], axis = 1)


# In[ ]:


X_train_new = build_model(X_train_new,y_train)


# In[ ]:


X_train_new = X_train_new.drop(["fueleconomy"], axis = 1)


# In[ ]:


X_train_new = build_model(X_train_new,y_train)


# In[ ]:


#Calculating Variance INflation Factor
checkVIF(X_train_new)


# In[ ]:


#Dropping curbweight due to multicollinearity
X_train_new = X_train_new.drop(["curbweight"], axis = 1)


# In[ ]:


#Dropping sedan because of high VIF value.
X_train_new = X_train_new.drop(["sedan"], axis = 1)


# In[ ]:


#Dropping wagon because of high p-value [insignificant in other variables]
X_train_new = X_train_new.drop(["wagon"], axis = 1)


# In[ ]:


X_train_new = X_train_new.drop(["dohcv"], axis = 1)


# In[ ]:


#Residual Analysis - error analysis
lm = sm.OLS(y_train,X_train_new).fit()
y_train_price = lm.predict(X_train_new)


# In[ ]:


#Scaling the test set - Normalization
num_vars = ['wheelbase', 'curbweight', 'enginesize', 'boreratio', 'horsepower','fueleconomy','carlength','carwidth','price']
df_test[num_vars] = scaler.fit_transform(df_test[num_vars])


# In[ ]:


#Dividing into X and y
y_test = df_test.pop('price')
X_test = df_test
#Making predictions.
X_train_new = X_train_new.drop('const',axis=1)
# Creating X_test_new dataframe by dropping variables from X_test
X_test_new = X_test[X_train_new.columns]

# Adding a constant variable 
X_test_new = sm.add_constant(X_test_new)
# Making predictions
y_pred = lm.predict(X_test_new)

