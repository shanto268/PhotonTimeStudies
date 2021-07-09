from NN import TrainNN, PredictNN
import datetime
import pandas as pd
import sys, os

df_train_all, df_test_all = pd.read_csv("train.csv"), pd.read_csv("test.csv")

targetList = ["diffL2", "diffL4"]
inputList = [
    'event_num', 'L1', 'R1', 'L3', 'R3', 'TopCounter', 'BottomCounter',
    'diffL1', 'diffL3', 'sumL1', 'sumL3'
]

date = datetime.datetime.today().strftime('%Y-%m-%d-%H_%M_%S')
emissions = "Run_{}".format(date)

TrainNN(df_train_all, df_test_all, targetList, inputList, emissions, epoch=1)
PredictNN(df_test_all, df_train_all, targetList, inputList, emissions)
