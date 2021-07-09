import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from MultiOutputNN import norm
import pandas as pd
import sys

emissions = sys.argv[1]

# Split the data into train and test with 80 train / 20 test
test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv")

train_stats = train.describe()
# Normalize the training and test data
norm_test_X = np.array(norm(test,train_stats))

# Restore model
loaded_model = tf.keras.models.load_model('./{}/'.format(emissions))

# Run predict with restored model
predictions = loaded_model.predict(norm_test_X)
diffL2 = predictions[0]
diffL4 = predictions[1]

# Do smth
