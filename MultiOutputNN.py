# Importing the libraries
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datetime
import sys
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from keras.layers import Dense, Activation, Flatten, LSTM, SimpleRNN, Embedding
from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split


# Helper functions
def norm(x,train_stats):
    return (x - train_stats['mean']) / train_stats['std']


def plot_diff(y_true, y_pred, emissions, title=''):
    plt.scatter(y_true, y_pred)
    plt.title(title)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim(plt.xlim())
    plt.ylim(plt.ylim())
    plt.plot([-100, 100], [-100, 100])
    plt.savefig("{}/diff_{}.png".format(emissions, title))


def plot_metrics(metric_name, title, emissions):
    plt.title(title)
    plt.plot(history.history[metric_name], color='blue', label=metric_name)
    plt.plot(history.history['val_' + metric_name],
             color='green',
             label='val_' + metric_name)
    plt.savefig("{}/metrics_{}.png".format(emissions, title))


def format_output(data):
    y1 = data.pop('diffL2')
    y1 = np.array(y1)
    y2 = data.pop('diffL4')
    y2 = np.array(y2)
    return y1, y2


def build_model(train):
    input_layer = Input(shape=(len(train.columns), ))
    #NN for y1
    first_layer = Dense(1024,
                        kernel_initializer='RandomNormal',
                        activation="relu")(input_layer)
    second_layer = Dense(1024,
                         kernel_initializer='RandomNormal',
                         activation="relu")(first_layer)
    y1_output = Dense(units='1',
                      kernel_initializer='RandomNormal',
                      activation=None,
                      name='diffL2')(second_layer)
    #NN for y2
    third_layer = Dense(1024,
                        kernel_initializer='RandomNormal',
                        activation="relu")(second_layer)
    fourth_layer = Dense(1024,
                         kernel_initializer='RandomNormal',
                         activation="relu")(third_layer)
    y2_output = Dense(units='1',
                      kernel_initializer='RandomNormal',
                      activation=None,
                      name='diffL4')(fourth_layer)
    # Define the model with the input layer and a list of output layers
    model = Model(inputs=input_layer, outputs=[y1_output, y2_output])
    return model


# In[29]:
def TrainNN(train, test, emissions):
    # Split the data into train and test with 80 train / 20 test
    train, val = train_test_split(train, test_size=0.2, random_state=1)

    # Get Y1 and Y2 as the 2 outputs and format them as np arrays
    train_stats = train.describe()
    train_stats.pop('diffL2')
    train_stats.pop('diffL4')
    train_stats = train_stats.transpose()
    train_Y = format_output(train)
    test_Y = format_output(test)
    val_Y = format_output(val)

    # Normalize the training and test data
    norm_train_X = np.array(norm(train, train_stats))
    norm_test_X = np.array(norm(test  , train_stats))
    norm_val_X = np.array(norm(val    , train_stats))

    model = build_model(train)

    # Specify the optimizer, and compile the model with loss functions for both outputs
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(.001, decay_rate=.36, decay_steps=1e5)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer,
                  loss={
                      'diffL2': 'mse',
                      'diffL4': 'mse'
                  },
                  metrics={
                      'diffL2': tf.keras.metrics.RootMeanSquaredError(),
                      'diffL4': tf.keras.metrics.RootMeanSquaredError()
                  })

    # Train the model for 200 epochs
    history = model.fit(norm_train_X,
                        train_Y,
                        epochs=100,
                        batch_size=256,
                        use_multiprocessing=True,
                        workers=5,
                        validation_data=(norm_test_X, test_Y))

    # Test the model and print loss and rmse for both outputs
    loss, Y1_loss, Y2_loss, Y1_rmse, Y2_rmse = model.evaluate(x=norm_val_X,
                                                              y=val_Y)

    print()
    print(f'loss: {loss}')
    print(f'diffL2_loss: {Y1_loss}')
    print(f'diffL4_loss: {Y2_loss}')
    print(f'diffL2_rmse: {Y1_rmse}')
    print(f'diffL4_rmse: {Y2_rmse}')

    # Run predict
    Y_pred = model.predict(norm_test_X)
    diffL2_pred = Y_pred[0]
    diffL4_pred = Y_pred[1]

    plot_diff(test_Y[0], Y_pred[0], emissions, title='diffL2')
    plot_diff(test_Y[1], Y_pred[1], emissions, title='diffL4')

    # Plot RMSE
    plot_metrics(metric_name='diffL2_output_root_mean_squared_error',
                 title='diffL2 RMSE',
                 emssions=emissions)
    plot_metrics(metric_name='diffL4_output_root_mean_squared_error',
                 title='diffL4 RMSE',
                 emssions=emissions)

    # Plot loss
    plot_metrics(metric_name='diffL2_output_loss',
                 title='diffL2 LOSS',
                 emssions=emissions)
    plot_metrics(metric_name='diffL4_output_loss',
                 title='diffL4 LOSS',
                 emssions=emissions)

    # Save model
    model.save('./{}/'.format(emissions), save_format='tf')


if __name__ == "__main__":
    specifyer = sys.argv[1]
    date = datetime.datetime.today().strftime('%Y-%m-%d-%H_%M_%S')
    emissions = "Run_{}_{}".format(specifyer, date)
    train, test = pd.read_csv("train.csv"), pd.read_csv("test.csv")
    TrainNN(train=train, test=test, emissions=emissions)
