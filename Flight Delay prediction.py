#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


flights=pd.read_csv('C:/Users/HARISUDHAN/Downloads/archive (11)/flights.csv')
flights=flights.sample(n=100000)
flights.head(11)


# In[3]:


flights.shape


# In[4]:


flights.isnull().sum()


# In[5]:


sns.countplot(x='CANCELLATION_REASON',data=flights)     # Reason for Cancellation of flight: A - Airline/Carrier; B - Weather; C - National Air System; D - Security


# In[6]:


sns.countplot(x="MONTH",hue="CANCELLATION_REASON",data=flights)


# In[7]:


plt.figure(figsize=(10, 10))
axis = sns.countplot(x=flights['ORIGIN_AIRPORT'], data =flights, order=flights['ORIGIN_AIRPORT'].value_counts().iloc[:20].index)
axis.set_xticklabels(axis.get_xticklabels(), rotation=90, ha="right")
plt.tight_layout()
plt.show()


# In[8]:



axis = plt.subplots(figsize=(20,14))
sns.heatmap(flights.corr(),annot = True)
plt.show()


# In[9]:


corr=flights.corr()
corr


# In[10]:


variables_to_remove=["YEAR","FLIGHT_NUMBER","TAIL_NUMBER","DEPARTURE_TIME","TAXI_OUT","WHEELS_OFF","ELAPSED_TIME","AIR_TIME","WHEELS_ON","TAXI_IN","ARRIVAL_TIME","DIVERTED","CANCELLED","CANCELLATION_REASON","AIR_SYSTEM_DELAY", "SECURITY_DELAY","AIRLINE_DELAY","LATE_AIRCRAFT_DELAY","WEATHER_DELAY","SCHEDULED_TIME","SCHEDULED_ARRIVAL"]
flights.drop(variables_to_remove,axis=1,inplace= True)
flights.columns


# In[11]:


flights.shape


# In[12]:


airport = pd.read_csv('C:/Users/HARISUDHAN/Downloads/archive (11)/airports.csv')
airport.head(5)


# In[13]:


flights.loc[~flights.ORIGIN_AIRPORT.isin(airport.IATA_CODE.values),'ORIGIN_AIRPORT']='OTHER'
flights.loc[~flights.DESTINATION_AIRPORT.isin(airport.IATA_CODE.values),'DESTINATION_AIRPORT']='OTHER'
flights


# In[14]:


print(flights.ORIGIN_AIRPORT.nunique())
print(flights.DESTINATION_AIRPORT.nunique())
print(flights.AIRLINE.nunique()) 


# In[15]:


flights=flights.dropna()
flights.head(10)


# In[16]:


flights.shape


# In[17]:


df=pd.DataFrame(flights)
df['DAY_OF_WEEK']= df['DAY_OF_WEEK'].apply(str)
df["DAY_OF_WEEK"].replace({"1":"SUNDAY", "2": "MONDAY", "3": "TUESDAY", "4":"WEDNESDAY", "5":"THURSDAY", "6":"FRIDAY", "7":"SATURDAY"},inplace=True)
flights


# In[18]:


dums = ['AIRLINE','ORIGIN_AIRPORT','DESTINATION_AIRPORT','DAY_OF_WEEK']
df_cat=pd.get_dummies(df[dums],drop_first=True)
df_cat


# In[19]:


df_cat.columns


# In[20]:


var_to_remove=["DAY_OF_WEEK","AIRLINE","ORIGIN_AIRPORT","DESTINATION_AIRPORT"]
df.drop(var_to_remove,axis=1,inplace=True)
df


# In[21]:


data=pd.concat([df,df_cat],axis=1)
data


# In[22]:


data.shape


# In[23]:


from sklearn.model_selection import train_test_split
from sklearn import metrics


# In[24]:


final_data = data.sample(n=60000)
final_data
X=final_data.drop("DEPARTURE_DELAY",axis=1)
Y=final_data.DEPARTURE_DELAY


# In[25]:


final_data.shape


# In[26]:


from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


# In[27]:


from sklearn.ensemble import RandomForestRegressor
reg_rf = RandomForestRegressor()
reg_rf.fit(X_train,y_train)


# In[28]:


y_pred = reg_rf.predict(X_test)


# In[29]:


reg_rf.score(X_train,y_train)


# In[30]:


reg_rf.score(X_test,y_test)


# In[31]:


metrics.r2_score(y_test,y_pred)


# In[32]:


pp=pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
pp


# In[33]:


from sklearn.model_selection import RandomizedSearchCV
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 200, num = 12)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10]


# In[34]:


random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}


# In[35]:


rf_random = RandomizedSearchCV(estimator = reg_rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)


# In[36]:


rf_random.fit(X_train,y_train)


# In[37]:


rf_random.best_params_


# In[38]:


p=rf_random.predict(X_test)


# In[39]:


metrics.r2_score(y_test,p)


# In[40]:



print('MAE:', metrics.mean_absolute_error(y_test,p))
print('MSE:', metrics.mean_squared_error(y_test,p))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test,p)))


# In[41]:


zz=pd.DataFrame({'Actual':y_test,'Predicted':p})
zz


# In[42]:


from sklearn.ensemble import GradientBoostingRegressor
gbr=GradientBoostingRegressor(random_state=0)


# In[43]:


GBR=gbr.fit(X_train,y_train)
pre=GBR.predict(X_test)


# In[44]:


metrics.r2_score(y_test,pre)


# In[45]:


gg=pd.DataFrame({'Actual':y_test,'Predicted':pre})
gg


# In[46]:


def predict(MONTH, DAY,SCHEDULED_DEPARTURE,DISTANCE, ARRIVAL_DELAY,AIRLINE,ORIGIN_AIRPORT,DESTINATION_AIRPORT,DAY_OF_WEEK):
    AIRLINE_index = np.where(X.columns==AIRLINE)[0][0]
    ORIGIN_index = np.where(X.columns==ORIGIN_AIRPORT)[0][0]
    DESTINATION_index = np.where(X.columns==DESTINATION_AIRPORT)[0][0]
    DAY_OF_WEEK_index = np.where(X.columns==DAY_OF_WEEK)[0][0]
    x= np.zeros(len(X.columns))
    x[0] = MONTH
    x[1] = DAY
    x[2] = SCHEDULED_DEPARTURE
    x[3] = DISTANCE
    x[4] = ARRIVAL_DELAY
    if AIRLINE_index >=0:
        x[AIRLINE_index] = 1
    if ORIGIN_index >=0:
        x[ORIGIN_index] = 1
    if DESTINATION_index >=0:
        x[DESTINATION_index] = 1
    if  DAY_OF_WEEK_index >= 0:
        x[ DAY_OF_WEEK_index] = 1

    return gbr.predict([x])[0]


# In[53]:


res= predict(5,6,1515,328,-8.0,'AIRLINE_OO','ORIGIN_AIRPORT_PHX','DESTINATION_AIRPORT_ABQ','DAY_OF_WEEK_TUESDAY')
print("Delay :",res)
if(res<=-15):
  print("Flight is delayed")
else:
  print("Flight is not delayed")


# In[66]:


import joblib


# In[70]:


f = "C:/Users/HARISUDHAN/Documents/ML/model_flight_delay.pkl"
joblib.dump(gbr,f)         


# In[ ]:




