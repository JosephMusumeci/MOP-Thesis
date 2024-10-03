# Keras Optimiser test script for Thesis

"""
Created on Tue Sep 09 20:05:24 2024

@author: Joseph Musumeci
"""

"""
This script cycles through the imported keras optimisers and creates models for each one.
It goes through each optimizer ten times and gets an average / best model.
It also cycles with and without outlier data from the featuremap.npy files.
"""

"""
import all the necessary packages
Tested with:
    
Tensorflow 2.2.0
Keras 2.3.0
Python 3.7

"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import metrics
import os

from keras.optimizers import Adadelta
from keras.optimizers import Adagrad
from keras.optimizers import Adam
from keras.optimizers import Adamax
from keras.optimizers import Nadam
from keras.optimizers import RMSprop
from keras.optimizers import SGD

from keras.models import Model
from keras.layers import Dense 
from keras.layers import Input
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import Dropout


# set the directory
import os
path = os.getcwd()
os.chdir(path)


# Define batch size and epochs
batch_size = 128
epochs = 150

# Optimizers list
optimizers = {
    'Adadelta': Adadelta(),
    'Adagrad': Adagrad(),
    'Adam': Adam(),
    'Adamax': Adamax(),
    'Nadam': Nadam(),
    'RMSprop': RMSprop(),
    'SGD': SGD(),
}

# Directories to loop through
folders = ['featurewith', 'featurewout']

# Function to define CNN model
def define_CNN(in_shape, n_keypoints, optimizer):
    in_one = Input(shape=in_shape)
    conv_one_1 = Conv2D(16, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same')(in_one)
    conv_one_1 = Dropout(0.3)(conv_one_1)
    conv_one_2 = Conv2D(32, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same')(conv_one_1)
    conv_one_2 = Dropout(0.3)(conv_one_2)
    conv_one_2 = BatchNormalization(momentum=0.95)(conv_one_2)
    fe = Flatten()(conv_one_2)
    dense_layer1 = Dense(512, activation='relu')(fe)
    dense_layer1 = BatchNormalization(momentum=0.95)(dense_layer1)
    dense_layer1 = Dropout(0.4)(dense_layer1)
    out_layer = Dense(n_keypoints, activation='linear')(dense_layer1)
    model = Model(in_one, out_layer)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse', 'mape', tf.keras.metrics.RootMeanSquaredError()])
    return model

# Function to create directories for results
def create_output_dir(optimiser_name, folder):
    output_direct = f'results/{optimiser_name}/{folder}'
    if not os.path.exists(output_direct):
        os.makedirs(output_direct)
    return output_direct

# Loop through each optimiser and both feature folders
for optimiser_name, optimizer in optimizers.items():
    for folder in folders:
        # Load the feature and labels
        featuremap_train = np.load(f'{folder}/radar_data_train.npy', allow_pickle=True)
        featuremap_validate = np.load(f'{folder}/radar_data_validate.npy', allow_pickle=True)
        featuremap_test = np.load(f'{folder}/radar_data_test.npy')
        labels_train = np.load(f'{folder}/kinect_data_train.npy', allow_pickle=True)
        labels_validate = np.load(f'{folder}/kinect_data_validate.npy', allow_pickle=True)
        labels_test = np.load(f'{folder}/kinect_data_test.npy', allow_pickle=True)
        
        # Initialize result array
        paper_result_list = []
        
        # Output directory for saving results
        output_direct = create_output_dir(optimiser_name, folder)
        
        # Repeat i iterations to get the average result
        for i in range(10):
            # Instantiate the model
            keypoint_model = define_CNN(featuremap_train[0].shape, 57, optimizer)
            
            # Initial maximum error
            score_min = 10
            history = keypoint_model.fit(featuremap_train, labels_train, batch_size=batch_size, epochs=epochs, verbose=1, 
                                         validation_data=(featuremap_validate, labels_validate))
            
            # Evaluate model and save metrics
            score_train = keypoint_model.evaluate(featuremap_train, labels_train, verbose=1)
            score_test = keypoint_model.evaluate(featuremap_test, labels_test, verbose=1)
            result_test = keypoint_model.predict(featuremap_test)

            # Save the plots
            plt.plot(history.history['mae'])
            plt.plot(history.history['val_mae'])
            plt.title('Model accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validate'], loc='upper left')
            plt.savefig(os.path.join(output_direct, f'accuracy_plot_{i}.png'))
            plt.clf()

            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('Model loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validate'], loc='upper left')
            plt.savefig(os.path.join(output_direct, f'loss_plot_{i}.png'))
            plt.clf()

            # Error metrics for x, y, z
            x_mae = metrics.mean_absolute_error(labels_test[:, 0:19], result_test[:, 0:19], multioutput='raw_values')
            y_mae = metrics.mean_absolute_error(labels_test[:, 19:38], result_test[:, 19:38], multioutput='raw_values')
            z_mae = metrics.mean_absolute_error(labels_test[:, 38:57], result_test[:, 38:57], multioutput='raw_values')
            
            all_19_points_mae = np.concatenate((x_mae, y_mae, z_mae)).reshape(3, 19)
            avg_19_points_mae = np.mean(all_19_points_mae, axis=0)
            avg_19_points_mae_xyz = np.mean(all_19_points_mae, axis=1).reshape(1, 3)
            
            all_19_points_mae_Transpose = all_19_points_mae.T
            
            # Merge MAE and RMSE
            x_rmse = metrics.mean_squared_error(labels_test[:, 0:19], result_test[:, 0:19], multioutput='raw_values', squared=False)
            y_rmse = metrics.mean_squared_error(labels_test[:, 19:38], result_test[:, 19:38], multioutput='raw_values', squared=False)
            z_rmse = metrics.mean_squared_error(labels_test[:, 38:57], result_test[:, 38:57], multioutput='raw_values', squared=False)
            
            all_19_points_rmse = np.concatenate((x_rmse, y_rmse, z_rmse)).reshape(3, 19)
            avg_19_points_rmse = np.mean(all_19_points_rmse, axis=0)
            avg_19_points_rmse_xyz = np.mean(all_19_points_rmse, axis=1).reshape(1, 3)
            
            all_19_points_rmse_Transpose = all_19_points_rmse.T
            all_19_points_maermse_Transpose = np.concatenate((all_19_points_mae_Transpose, all_19_points_rmse_Transpose), axis=1) * 100
            avg_19_points_maermse_Transpose = np.concatenate((avg_19_points_mae_xyz, avg_19_points_rmse_xyz), axis=1) * 100
            
            paper_result_maermse = np.concatenate((all_19_points_maermse_Transpose, avg_19_points_maermse_Transpose), axis=0)
            paper_result_maermse = np.around(paper_result_maermse, 2)
            paper_result_maermse = paper_result_maermse[:, [0, 3, 1, 4, 2, 5]]
            
            # Append each iteration result
            paper_result_list.append(paper_result_maermse)
            
            # Save the best model so far
            if score_test[1] < score_min:
                keypoint_model.save(os.path.join(output_direct, f'model_{i}.h5'))
                score_min = score_test[1]

        # Average the result for all iterations
        mean_paper_result_list = np.mean(paper_result_list, axis=0)
        mean_mae = np.mean(np.dstack((mean_paper_result_list[:, 0], mean_paper_result_list[:, 2], mean_paper_result_list[:, 4])).reshape(20, 3), axis=1)
        mean_rmse = np.mean(np.dstack((mean_paper_result_list[:, 1], mean_paper_result_list[:, 3], mean_paper_result_list[:, 5])).reshape(20, 3), axis=1)
        mean_paper_result_list = np.concatenate((np.mean(paper_result_list, axis=0), mean_mae.reshape(20, 1), mean_rmse.reshape(20, 1)), axis=1)
        
        # Save the final accuracy and result file
        output_filename = os.path.join(output_direct, f'mop_accuracy_{optimiser_name}')
        np.save(output_filename + ".npy", mean_paper_result_list)
        np.savetxt(output_filename + ".txt", mean_paper_result_list, fmt='%.2f')
        
        print(f"Results saved for optimiser {optimiser_name} in folder {folder}")

print("All optimisations and saves are complete.")
