import datetime
from MuDataFrame import MuDataFrame
from NN import TrainNN, PredictNN
from sklearn.model_selection import train_test_split
import datetime
import sys, os

#loading the input data
path2csvq = "/lustre/work/sshanto/PhotonTimeML/data_sets/calibration_new_set_up/calibration_new_set_up.csv"

path2csvl = "/Volumes/mac_extended/Research/MT/proto1b/data_sets/calibration_new_set_up/calibration_new_set_up.csv"

mdfo_calib = MuDataFrame(path2csvl)
mdf_calib = mdfo_calib.events_df
mdfo_calib.keep4by4Events()

all_df = mdfo_calib.events_df
all_df.drop(columns=[
    'Unnamed: 0', 'event_time', 'Run_Num', 'time_of_day', 'SmallCounter',
    'speed', 'xx', 'yy'
],
            inplace=True)
df_train_all, df_test_all = train_test_split(all_df, test_size=0.3)
targetList = ["diffL2", "diffL4"]
inputList = [
    'event_num', 'L1', 'R1', 'L3', 'R3', 'TopCounter', 'BottomCounter',
    'diffL1', 'diffL3', 'sumL1', 'sumL3'
]

date = datetime.datetime.today().strftime('%Y-%m-%d-%H_%M_%S')
emissions = "Run_{}".format(date)

df_train_all.to_csv("train.csv")
df_test_all.to_csv("test.csv")

# TrainNN(df_train_all, df_test_all, targetList, inputList, emissions, epoch=1)
# PredictNN(df_test_all, df_train_all, targetList, inputList, emissions)
