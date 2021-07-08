#!/usr/bin/env python
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, LSTM, SimpleRNN, Embedding
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import seaborn as sb
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.utils.vis_utils import plot_model
try:
    # pydot-ng is a fork of pydot that is better maintained.
    import pydot_ng as pydot
except ImportError:
    # pydotplus is an improved version of pydot
    try:
        import pydotplus as pydot
    except ImportError:
        # Fall back on pydot if necessary.
        try:
            import pydot
        except ImportError:
            pydot = None
import math
import datetime
from MuDataFrame import *
from scipy import stats
from sklearn.model_selection import train_test_split


# In[41]:



def TrainNN(df_train,df_test, targetList, inputList,emissions,epoch=1):
    
    target = [df_train[targetList[0]].to_numpy(), df_train[targetList[1]].to_numpy()]
    df_train.drop(columns=targetList,inplace=True)
 
    df_train = df_train[inputList]
    df_test = df_test[inputList]
    
    emissions = "{}_epoch_{}".format(emissions,epoch)

    tensorboard_callbacks = TensorBoard(log_dir=emissions + "/logs",
                                            histogram_freq=1)

    NN_model = Make_network(df_train.shape[1])

    checkpoint_name = emissions + '/Weights-{epoch:03d}--{val_loss:.5f}.hdf5'
    checkpoint = ModelCheckpoint(checkpoint_name,
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='auto')

    callbacks_list = [checkpoint, tensorboard_callbacks]

    # Fit network
    history_callback = NN_model.fit(x=df_train,
                                    y=target,
                                    shuffle=True,
                                    epochs=epoch,
                                    batch_size=256,
                                    validation_split=0.1,
                                    workers=2,
                                    callbacks=callbacks_list)
    file = open("NetworkHistory.txt", "a")

    CSV_Callbacks(history_callback, emissions)
    

def Make_network(in_shape):
    NN_model = Sequential()

    NN_model.add(
        Dense(1024,
              kernel_initializer='RandomNormal',
              input_dim=in_shape,
              activation="relu"))

    NN_model.add(
        Dense(1024, kernel_initializer='RandomNormal', activation="relu"))
    NN_model.add(Dense(1, kernel_initializer='RandomNormal', activation=None))
    NN_model = compile_NN(NN_model)
    NN_model.summary()

    return NN_model
    
def CSV_Callbacks(callbacks, directory):

    f = open("callback_history.csv", "w")
    print(type(callbacks.history))
    for title in callbacks.history:
        print(title)
        ##print(title, end==',')

    history_df = pd.DataFrame.from_dict(callbacks.history)
    history_df.to_csv(
        "{}/callback_history.csv".format(directory), index=False
    )  #MIH: the callback_history file gets moved to the right directory

def learning_schedule(epoch):
    learning_rate = .001 * math.exp(-epoch / 20)

    tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
    return learning_rate


def compile_NN(NN_model):
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        .001, decay_rate=.36, decay_steps=1e5)

    opt = Adam(learning_rate=lr_schedule)

    NN_model.compile(loss=tf.keras.losses.MeanAbsolutePercentageError(),
                     optimizer=opt,
                     metrics=['mse', 'mae', 'mape'])

    return NN_model


# ### Predicting from NN

# In[38]:


def PredictNN(df2,df1,targetList, inputList,dir):

    df_test = df1.copy(deep='all')
    df_train = df2

    target = [df_train[targetList[0]].to_numpy(), df_train[targetList[1]].to_numpy()]
    df_train.drop(columns=targetList,axis=1,inplace=True)

    NN_model = Make_network(df_train.shape[1])

    file = find_min_weights(dir)
    min_loss = None
    pwd = os.getcwd()

    wights_file = file  # choose the best checkpoint
    NN_model.load_weights(wights_file)  # load it

    NN_model = compile_NN(NN_model)


    predictions = (NN_model.predict(df_train))

    df_train['diffL2'] = target[0]
    df_train['diffL4'] = target[1]

    df1, df2 = split()
    
    df2['diffL2_predicted'] = predictions[0]
    df2['diffL4_predicted'] = predictions[1]

    write_csv_file(df2, dir)

    if debug_NN_predict == True:
        for col in df1.columns:
            print(col)
    tmp = NN_model.layers[0].get_weights()
    
def find_min_weights(directory):
    pwd = os.getcwd()
    min_loss = None
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('hdf5'):
                splt = file.split('--')
                if min_loss == None:
                    min_loss = float(splt[1][:-5])
                    min_file = os.path.join(root, file)
                elif min_loss > float(splt[1][:-5]):
                    min_loss = float(splt[1][:-5])
                    min_file = os.path.join(root, file)
                # print(file)
                # print(splt[1][:-5])
                # print(float(splt[1][:-5]))
    return min_file


def write_csv_file(df, dir, file_name='results.csv'):
    print("testing the write func:" + dir)
    df.to_csv(os.path.join(dir, file_name), header=True, index=False)
    print("CSV file written")


# # Run Test

# In[ ]:



#loading the input data 
path2csv = "/lustre/work/sshanto/PhotonTimeML/"
mdfo_calib = MuDataFrame("{}/data_sets/calibration_new_set_up/calibration_new_set_up.csv".format(path2csv))
mdf_calib = mdfo_calib.events_df
mdfo_calib.keep4by4Events()


# In[42]:



all_df = mdfo_calib.events_df
all_df.drop(columns=['Unnamed: 0', 'event_time','Run_Num','time_of_day','SmallCounter','speed','xx','yy'],inplace=True)
df_train_all, df_test_all = train_test_split(all_df, test_size=0.3)
targetList = ["diffL2","diffL4"]
inputList = ['event_num','L1','R1', 'L3', 'R3', 'TopCounter', 'BottomCounter', 'diffL1', 'diffL3', 'sumL1', 'sumL3']


# Train the Network

# In[ ]:


date = datetime.datetime.today().strftime('%Y-%m-%d-%H_%M_%S')
emissions = "Run_{}".format(date)

TrainNN(df_train_all, df_test_all, targetList, inputList, emissions, epoch=1)


# Test the Network

# In[ ]:


PredictNN(df_test_all,df_train_all,targetList, inputList, emissions)

