#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import preprocessing as prp
from sklearn.model_selection import train_test_split
import warnings

get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')
plt.style.use('dark_background')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 70)


# In[188]:


tf.__version__


# In[189]:


df_train_o = pd.read_csv('train.csv')
print('Shape train'+str(df_train_o.shape))
df_test_o = pd.read_csv('test.csv')
print('Shape test'+str(df_test_o.shape))
df_train = df_train_o.copy()
df_test = df_test_o.copy()


# In[190]:


df_train.head(3)


# In[191]:


df_train.drop(['Unnamed: 0', 'f1_year_date', 'key_py_fy', 'ticker', 
              'publish_date'], axis=1, inplace = True)
df_test.drop(['Unnamed: 0', 'f1_year_date', 'key_py_fy', 'ticker', 
              'publish_date'], axis=1, inplace = True)
print('Shape train'+str(df_train.shape)+'\n'+
     'Shape train'+str(df_test.shape))


# In[192]:


x = df_train.drop('f1_price', axis=1)
y_nolog = df_train['f1_price']
y = np.log(df_train['f1_price'])


# In[432]:


plt.hist(y_nolog, bins=50)
plt.xlabel('Price')
plt.ylabel('Number of Obdervstions')
plt.title('Original Form')


# In[433]:


print(y.mean())
print(y.std())
plt.hist(y, bins = 50)
plt.ylabel('Number of Observations')
plt.xlabel('Log_Price')
plt.title('Log-Transformed')


# In[195]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
print('x_train'+str(x_train.shape)+'\n'+
    'y_train'+str(y_train.shape)+'\n'+
    'x_test'+str(x_test.shape)+'\n'+
    'y_test'+str(y_test.shape) )


# In[196]:

#Standardize variables
scalerX = prp.StandardScaler().fit(x_train)
scalerY = prp.StandardScaler().fit(np.array(y_train).reshape(-1,1))
x_train = scalerX.transform(x_train)
y_train = scalerY.transform(np.array(y_train).reshape(-1,1))
x_test = scalerX.transform(x_test)
y_test = scalerY.transform(np.array(y_test).reshape(-1,1))
x_pred = scalerX.transform(df_test.drop('f1_price', axis=1) )
y_true = np.array(df_test['f1_price']).reshape(-1,1)
print('x_train'+str(x_train.shape)+'\n'+
    'y_train'+str(y_train.shape)+'\n'+
    'x_test'+str(x_test.shape)+'\n'+
    'y_test'+str(y_test.shape)+'\n'+
    'x_pred'+str(x_pred.shape)+'\n'+
    'y_true'+str(y_true.shape))


# Model 1

# In[417]:


tf.reset_default_graph()
#Number of observations in training data
n_time_dimensions = x_train.shape[1]

#Number of neurons
n_neurons_1 = 49
n_neurons_2 = 24
n_neurons_3 = 12
n_neurons_4 = 6

#Session
sess = tf.InteractiveSession(config=tf.ConfigProto(
  intra_op_parallelism_threads=12))

#Placeholder
X = tf.placeholder(dtype = tf.float32, shape = [None, n_time_dimensions]) #inputs
Y = tf.placeholder(dtype = tf.float32, shape = [None,1])    #f1 predicted price
Z = tf.placeholder(dtype = tf.float32, shape = [None,1]) #pub_price
#Initializers
sigma=1
weight_initializer = tf.variance_scaling_initializer(mode='fan_avg', distribution='uniform',
                                                    scale = sigma)
bias_initializer = tf.zeros_initializer()

#Hidden weights
w_hidden_1 = tf.Variable(weight_initializer([n_time_dimensions, n_neurons_1]))
bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))
w_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))
w_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))
w_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))
bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]))

#Output weights
w_out = tf.Variable(weight_initializer([n_neurons_4, 1]))
b_out = tf.Variable(bias_initializer([1]))

#Hidden layers
hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, w_hidden_1), bias_hidden_1))
#drop_out_1 = tf.nn.dropout(hidden_1, keep_prob = 0.8)
hidden_2 =tf.nn.relu(tf.add(tf.matmul(hidden_1, w_hidden_2), bias_hidden_2))

#drop_out_2 = tf.nn.dropout(hidden_2, keep_prob = 0.5)
hidden_3= tf.nn.relu(tf.add(tf.matmul(hidden_2, w_hidden_3), bias_hidden_3))
#drop_out_3 = tf.nn.dropout(hidden_3, keep_prob = 0.5)
hidden_4= tf.nn.relu(tf.add(tf.matmul(hidden_3, w_hidden_4), bias_hidden_4))
#drop_out_4 = tf.nn.dropout(hidden_4, keep_prob = 0.8)

#Output layer
out = tf.transpose(tf.add(tf.matmul(hidden_4, w_out), b_out))

#Cost function
mse = tf.reduce_mean(tf.squared_difference(out, Y))


#Optimizer
optimizer = tf.train.AdadeltaOptimizer()
opt = optimizer.minimize(loss=mse)


# In[418]:


#Init
sess.run(tf.global_variables_initializer())

#Setup plot
plt.ion()
fig = plt.figure()
ax1 =fig.add_subplot(111)
line1, = ax1.plot(y_test)
line2, = ax1.plot(y_test*0.5)

#Fit neural net
batch_size = 50
mse_train=[]
mse_test=[]

#Run
train_error=[]
test_error=[]
epochs=[3]
for epoch in epochs:
    for e in range(epoch):
        #Shuffle training data
        indices = np.random.permutation(np.arange(len(y_train)))
        x_train = x_train[indices]
        y_train = y_train[indices]

        #Minibatch training
        for i in range(0, len(y_train//batch_size)):
            start = i * batch_size
            batch_x = x_train[start : start+batch_size]
            batch_y = y_train[start : start+batch_size]
            #Run optimizer with batch
            sess.run(opt, feed_dict={X:batch_x, Y:batch_y})

            #Show progress
            if np.mod(i, 1000) == 0:
                #MSE train and test
                mse_train.append(sess.run(mse, feed_dict={X:x_train, Y:y_train}))
                mse_test.append(sess.run(mse, feed_dict={X:x_test, Y:y_test}))
                print('Train Error: '+str(round(mse_train[-1], 2)))
                print('Test Error: '+str(round(mse_test[-1], 2)))
                #Prediction
#                 pred=sess.run(out, feed_dict={X:x_test})
#                 line2.set_ydata(pred)
#                 plt.title('Epoch '+str(e)+', Batch '+str(i))')
                
                print('Epoch '+str(e)+', Batch '+str(i))
                plt.pause(0.01)  
                
        train_error.append(sess.run(mse, feed_dict={X:x_train, Y:y_train}))
        test_error.append(sess.run(mse, feed_dict={X:x_test, Y:y_test}))


# In[430]:


saver = tf.train.Saver()
save_path = saver.save(sess, "./model_3.ckpt")
print("Model saved in path: %s" % save_path)


# In[419]:


print(train_error)
print(test_error)


# In[269]:


# train_error_e = train_error.copy()
# test_error_e = test_error.copy()


# In[420]:


fig = plt.figure()
#plt.plot(train_error)
plt.plot(test_error, '-g', label = 'Validation')
plt.plot(train_error, '-r', label = 'Train')
plt.xlabel('Number of Epocs')
plt.ylabel('MSE')
plt.legend(loc = 'upper right')


# In[421]:


y_true_st_log = scalerY.transform(np.array(np.log(y_true)).reshape(-1,1))


# In[422]:


#Predict new data
y_pred = sess.run(out, feed_dict={X:x_pred})
mse_pred_std_log = np.mean((y_pred-y_true_st_log)**2)
mse_pred_std_log


# In[423]:


#transform prediction to original form
y_pred_tr = np.transpose(np.exp(scalerY.inverse_transform(y_pred)))
y_pred_tr


# In[424]:


mse_pred = np.mean((y_pred_tr-y_true)**2)
mse_pred


# In[425]:


df_test['f1_price_pred'] = y_pred_tr
f1_price_pred = df_test['f1_price_pred']
df_test.drop(labels=['f1_price_pred'], axis=1, inplace=True)
df_test.insert(0, 'f1_price_pred', f1_price_pred)
print(df_test['f1_price_pred'].describe())
df_test.head(3)


# In[426]:


df_test['f1_price'].describe()


# In[431]:


plt.hist(df_test['f1_price_pred'], bins = 50)
plt.xlabel('F1_Price_Predicted')
plt.ylabel('Number of Companies')
plt.title('3 Epochs')


# In[357]:


plt.hist(df_test['f1_price'], bins = 50)
plt.xlabel('F1_Price_Actual')
plt.ylabel('Number of Companies')
plt.title('Actual Forward Price')


# In[342]:


print('Train long: '+str(df_train.loc[df_train['f1_price']>df_train['pub_price']].shape[0]/
                        df_train.shape[0]))
print('Trainn short: '+str(df_train.loc[df_train['f1_price']<df_train['pub_price']].shape[0]/
                       df_train.shape[0]))
print('Test long: '+str(df_test.loc[df_test['f1_price_pred']>df_test['pub_price']].shape[0]/
                       df_test.shape[0]))
print('Test short: '+str(df_test.loc[df_test['f1_price_pred']<df_test['pub_price']].shape[0]/
                       df_test.shape[0]))


# In[428]:


print('Long gain: '+str(np.subtract(df_test.loc[df_test['f1_price_pred']>df_test['pub_price']]['f1_price'],
df_test.loc[df_test['f1_price_pred']>df_test['pub_price']]['pub_price']).sum()))
print('Short gain: '+str(np.subtract(df_test.loc[df_test['f1_price_pred']<df_test['pub_price']]['pub_price'],
df_test.loc[df_test['f1_price_pred']<df_test['pub_price']]['f1_price']).sum()))


# In[281]:


#if invest all portfolio long
print('Long all portfolio gain: '+str(np.subtract(df_test['f1_price'],
df_test['pub_price']).sum()))
print('Short all portfolio gain: '+str(np.subtract(df_test['pub_price'],
df_test['f1_price']).sum()))
print(str(df_test.loc[df_test['f1_price']>df_test['pub_price']].shape[0]/
                       df_test.shape[0]))


# In[429]:


#Percentage long correctly long, short correctly short
print(len(list(set(df_test.loc[df_test['f1_price_pred']>df_test['pub_price']].index).intersection(
    df_test.loc[df_test['f1_price']>df_test['pub_price']].index))) / len(list(
    df_test.loc[df_test['f1_price']>df_test['pub_price']].index)) )
print(len(list(set(df_test.loc[df_test['f1_price_pred']<df_test['pub_price']].index).intersection(
    df_test.loc[df_test['f1_price']<df_test['pub_price']].index))) / len(list(
    df_test.loc[df_test['f1_price']<df_test['pub_price']].index)) )
(len(list(set(df_test.loc[df_test['f1_price_pred']<df_test['pub_price']].index).intersection(
    df_test.loc[df_test['f1_price']<df_test['pub_price']].index)))+len(list(set
    (df_test.loc[df_test['f1_price_pred']>df_test['pub_price']].index).intersection(
    df_test.loc[df_test['f1_price']>df_test['pub_price']].index)))) / len(list(
    df_test.index)) 

