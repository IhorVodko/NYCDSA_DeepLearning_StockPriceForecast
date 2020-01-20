#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from functools import reduce
import numpy as np
import re
from datetime import timedelta  
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 70)

# In[2]:

df1 = pd.read_csv('data/us-balance-annual.csv', sep=';')
print(df1.shape)
df1.head(5)

# In[3]:

df2 = pd.read_csv('data/us-income-annual.csv', sep=';')
print(df2.shape)
df1.head(5)

# In[4]:

df3 = pd.read_csv('data/us-cashflow-annual.csv', sep=';')
print(df3.shape)
df1.head(5)

# In[5]:

df4 = pd.read_csv('data/us-shareprices-daily.csv', sep=';')
print(df4.shape)
df4.head(5)

# In[6]:

df4['Date'] = pd.to_datetime(df4['Date'])

# In[7]:

df = [df1, df2, df3]
for x in df:
        x['Publish Date'] = pd.to_datetime(x['Publish Date'])

# In[8]: 

for x in df:
    x['Publish Year'] = x['Publish Date'].dt.year

# In[9]: Create a table key

for x in df:
    x['key_py_fy'] = x.apply(lambda x: x['Ticker']+'_'+str(x['Publish Year'])+'_'+
    str(x['Fiscal Year']), axis=1) 

# In[10]:

df_3 = reduce(lambda left, right: pd.merge(left, right, on=['key_py_fy'], how = 'inner',
            suffixes=('', '_y')), df)
print(df_3.shape)
df_3.drop(list(df_3.filter(regex='_y$')), axis=1, inplace=True)
print(df_3.shape)
df_3.head(4)

# In[11]:

key_py_fy = df_3['key_py_fy']
df_3.drop(labels=['key_py_fy'], axis=1, inplace=True)
df_3.insert(0, 'key_py_fy', key_py_fy)
df_3.head(3)

# In[12]:

df_3['key_py_fy'].unique().shape

# In[13]:

df_3['f1_Year_Date'] = df_3['Publish Date'] + timedelta(days=365)


# In[14]:

df_4 = df_3.merge(df4[['Ticker', 'Date', 'Adj. Close']], left_on = ['Ticker', 'f1_Year_Date'], 
                 right_on=['Ticker', 'Date'], how='left')

# In[15]:

df_4.shape

# In[16]:

df_2prices = df_4.merge(df4[['Ticker', 'Date', 'Adj. Close']], left_on = ['Ticker', 'Publish Date'], 
                 right_on=['Ticker', 'Date'], how='left')

# In[17]:

df_2prices.shape

# In[18]:

df_2prices.head(2)

# In[29]:

df_model = df_2prices.drop(['SimFinId', 'Currency', 'Fiscal Year', 'Fiscal Period',
                     'Report Date', 'Date_x', 'Date_y'],axis=1)
# In[30]:

df_model.rename(columns={'Adj. Close_x':'f1_price', 'Adj. Close_y':'pub_price'}, inplace=True)
df_model.rename(str.lower, axis='columns', inplace=True)
df_model.columns = df_model.columns.str.replace(' ', '_')

# In[31]:

print(df_model.shape)
df_model = df_model[pd.notnull(df_model['f1_price'])&pd.notnull(df_model['pub_price'])]
print(df_model.shape)

# In[32]:

df_model.head(20)

# In[33]:

df_model.isnull().sum(axis=0)

# In[34]:

df_model=df_model.drop(['long_term_investments_&_receivables', 'treasury_stock', 'research_&_development',
              'depreciation_&_amortization', 'net_extraordinary_gains_(losses)',
              'change_in_accounts_receivable', 'change_in_inventories', 
              'change_in_accounts_payable', 'change_in_other', 'net_change_in_long_term_investment',
              ], axis=1)
print(df_model.shape)
df_model.isnull().sum(axis=0)

# In[35]:

df_model.fillna(0, inplace=True)
df_model.isnull().sum(axis=0)

# In[36]:

for x in ['f1_year_date', 'f1_price', 'pub_price']:
        i = 0
        name = df_model[x]
        df_model.drop(labels=[x], axis=1, inplace=True)
        df_model.insert(i, x, name)
        i += 1

# In[37]:

df_model = df_model.drop_duplicates(keep='first')
print(df_model.shape)
df_model.head(2)

# In[38]:

df_model = df_model[(df_model['f1_price']>10) & (df_model['f1_price']<100)].sort_values('f1_price')

# In[39]:

df_model=df_model[(df_model['pub_price']>10) & (df_model['pub_price']<100)].sort_values('pub_price')

# Split for test and train

# In[40]:

test_all = []
for x in df_model.ticker.unique():
    ind = df_model.loc[(df_model['ticker']==x) & (df_model['publish_date']==
             max(df_model.loc[df_model['ticker']==x]['publish_date']))].index
    test_all.append(ind)
len(test_all)

# In[41]:

test_int = list(map(lambda x: x[0], test_all))
df_test_all = df_model[df_model.index.isin(test_int)]
print(df_test_all.shape)
df_test = df_test_all[df_test_all['publish_date'].dt.year ==
                      max(df_model['publish_date'].dt.year)]
print(df_test.shape)
print(df_test.drop_duplicates(keep='first').shape)

df_train = pd.concat([df_model,df_test]).drop_duplicates(keep=False)
print(df_train.shape)

# In[42]:df_model - no forecasted variable column

df_model.to_csv('data_model.csv')
df_test.to_csv('test.csv')
df_train.to_csv('train.csv')

