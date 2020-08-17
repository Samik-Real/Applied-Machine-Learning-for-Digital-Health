from tensorflow.keras.utils import model_to_dot
from sklearn.metrics import accuracy_score
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.utils import plot_model
import os
import matplotlib.pyplot as plt

from DataPreparationUtilityFunctions import*
import ModelTrainingUtilityFunctions
from ModelTrainingUtilityFunctions import*

from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier


def preprocessing_training_test(test_file_path='test/', training_file_path='train/', number_of_sensors=2, desired_data_points_per_second=4, downsampling_method='mean', amount_of_numbers_to_group_in_downsampling=2, upsampling_method='cubic_spline', window_size_in_seconds=2, window_overlap=0.5, cross_validation_evaluation=False, seed=11):
    test_file_path = test_file_path
    training_file_path = training_file_path

    test_class_label_dataframe = None
    test_accelerometer_dataframe = None
    test_meditag_dataframe = None
    training_class_label_dataframe = None
    training_accelerometer_dataframe = None
    training_meditag_dataframe = None

    if number_of_sensors == 0:  # 0 For accelerometer only
        print("Loading test data...")
        test_class_label_dataframe = load_class_label_dataframe(file_path=test_file_path + 'activities_test.csv')
        test_accelerometer_dataframe = load_accelerometer_dataframe(file_path=test_file_path + 'accelerometer_test.csv')
        print("Loading training data...")
        training_class_label_dataframe = load_class_label_dataframe(file_path=training_file_path + 'activities_train.csv')
        training_accelerometer_dataframe = load_accelerometer_dataframe(file_path=training_file_path + 'accelerometer_train.csv')
        if cross_validation_evaluation == True:
            print("Applying Z-score normalization on data...")
            training_accelerometer_dataframe, mean_arr_train_acc, std_arr_train_acc, col_name_arr_train_acc = z_score_normalization_TrainingSet(training_accelerometer_dataframe)
            test_accelerometer_dataframe = z_score_normalization_TestSet(test_accelerometer_dataframe, mean_arr_train_acc, std_arr_train_acc, col_name_arr_train_acc)
    elif number_of_sensors == 1:  # 1 For meditag only
        #importar accelerometer tambien SOLO para copiar el duration de accelerometer al de meditag, luego poner nulo a accelerometer
        print("Loading test data...")
        test_class_label_dataframe = load_class_label_dataframe(file_path=test_file_path + 'activities_test.csv')
        test_meditag_dataframe = load_meditag_dataframe(file_path=test_file_path + 'meditag_test.csv')
        print("Loading training data...")
        training_class_label_dataframe = load_class_label_dataframe(file_path=training_file_path + 'activities_train.csv')
        training_meditag_dataframe = load_meditag_dataframe(file_path=training_file_path + 'meditag_train.csv')
        #Reading accelerometer only for time_duration
        test_accelerometer_dataframe = load_accelerometer_dataframe(file_path=test_file_path + 'accelerometer_test.csv')
        training_accelerometer_dataframe = load_accelerometer_dataframe(file_path=training_file_path + 'accelerometer_train.csv')
        if cross_validation_evaluation == True:
            print("Applying Z-score normalization on data...")
            training_meditag_dataframe, mean_arr_train_medi, std_arr_train_medi, col_name_arr_train_medi = z_score_normalization_TrainingSet(training_meditag_dataframe)
            test_meditag_dataframe = z_score_normalization_TestSet(test_meditag_dataframe, mean_arr_train_medi, std_arr_train_medi, col_name_arr_train_medi)
    elif number_of_sensors == 2:  # 2 For sensors meditag and accelerometer
        print("Loading test data...")
        test_class_label_dataframe = load_class_label_dataframe(file_path=test_file_path + 'activities_test.csv')
        # print("test_class_label_dataframe: \n" + str(test_class_label_dataframe.head()))
        test_accelerometer_dataframe = load_accelerometer_dataframe(file_path=test_file_path + 'accelerometer_test.csv')
        # print("test_accelerometer_dataframe: \n" + str(test_accelerometer_dataframe.head()))
        test_meditag_dataframe = load_meditag_dataframe(file_path=test_file_path + 'meditag_test.csv')
        # print("test_meditag_dataframe: \n" + str(test_meditag_dataframe.head()))
        # print()
        print("Loading training data...")
        training_class_label_dataframe = load_class_label_dataframe(file_path=training_file_path + 'activities_train.csv')
        # print("training_class_label_dataframe: \n" + str(training_class_label_dataframe.head()))
        training_accelerometer_dataframe = load_accelerometer_dataframe(file_path=training_file_path + 'accelerometer_train.csv')
        # print("training_accelerometer_dataframe: \n" + str(training_accelerometer_dataframe.head()))
        training_meditag_dataframe = load_meditag_dataframe(file_path=training_file_path + 'meditag_train.csv')
        # print("training_meditag_dataframe: \n" + str(training_meditag_dataframe.head()))

        if cross_validation_evaluation == True:
            # Use Z-score normalization for the data (to represent data variations in a better scale and to manage outliers, similar to min-max normalization)
            # https://www.codecademy.com/articles/normalization#:~:text=Min%2DMax%20Normalization,decimal%20between%200%20and%201.
            print("Applying Z-score normalization on data...")
            # Can introduce different proportions when normalizing before data resampling (but seemes to work better since it reduces distance)
            # Normalize training set and then normalize test with values (mean and std) used for the training for each column
            training_accelerometer_dataframe, mean_arr_train_acc, std_arr_train_acc, col_name_arr_train_acc = z_score_normalization_TrainingSet(training_accelerometer_dataframe)
            training_meditag_dataframe, mean_arr_train_medi, std_arr_train_medi, col_name_arr_train_medi = z_score_normalization_TrainingSet(training_meditag_dataframe)

            test_accelerometer_dataframe = z_score_normalization_TestSet(test_accelerometer_dataframe, mean_arr_train_acc, std_arr_train_acc, col_name_arr_train_acc)
            test_meditag_dataframe = z_score_normalization_TestSet(test_meditag_dataframe, mean_arr_train_medi, std_arr_train_medi, col_name_arr_train_medi)

    else:
        print("Error in number of sensors: " + str(number_of_sensors) + " not a possible option")
        return 0



    print("Creating data dictionaries...")
    # Test data
    test_data_dictionary = create_data_dictionary(test_class_label_dataframe, test_accelerometer_dataframe, test_meditag_dataframe, number_of_sensors=number_of_sensors)
    test_data_dictionary, before, after = remove_incomplete_data(test_data_dictionary, number_of_sensors=number_of_sensors)
    print("Number of items in test_data_dictionary before removing incomplete data: " + str(before))
    print("Number of items in test_data_dictionary after removing incomplete data: " + str(after))
    test_data_dictionary = data_dictionary_time_elapsed_correction(test_data_dictionary, number_of_sensors=number_of_sensors)

    dictionary_summary(test_data_dictionary, number_of_sensors=number_of_sensors, segmentID=None, dictionary_name='Test Data Dictionary')
    dictionary_statistics(test_data_dictionary, number_of_sensors=number_of_sensors, dictionary_name='Test Data Dictionary')  # number_of_sensors=2 is for meditag and accelerometer

    # Training data
    training_data_dictionary = create_data_dictionary(training_class_label_dataframe, training_accelerometer_dataframe, training_meditag_dataframe, number_of_sensors=number_of_sensors)
    training_data_dictionary, before, after = remove_incomplete_data(training_data_dictionary, number_of_sensors=number_of_sensors)
    print("Number of items in training_data_dictionary before removing incomplete data: " + str(before))
    print("Number of items in training_data_dictionary after removing incomplete data: " + str(after))
    training_data_dictionary = data_dictionary_time_elapsed_correction(training_data_dictionary, number_of_sensors=number_of_sensors)
    dictionary_summary(training_data_dictionary, number_of_sensors=number_of_sensors, segmentID=None, dictionary_name='Training Data Dictionary')
    dictionary_statistics(training_data_dictionary, number_of_sensors=number_of_sensors, dictionary_name='Training Data Dictionary')  # number_of_sensors=2 is for meditag and accelerometer

    # Keep in mind that all data arrays of all segment id's do not have the same length (i.e. x_accelerometer between segmentID_12 and segmentID_129)
    # dictionary_summary(test_data_dictionary, number_of_sensors=2, segmentID='segmentID_12', dictionary_name='Test Data Dictionary')
    # dictionary_summary(test_data_dictionary, number_of_sensors=2, segmentID='segmentID_129', dictionary_name='Test Data Dictionary')

    #TODO: despues de acabar todo, plotear los datos para ver que tipo de agrupacion y de interpolacion es la mas adecuada
    # plotear unas 10 en una grafica (diferente grafica para meditag y para accelerometer)
    desired_data_points_per_second = desired_data_points_per_second
    training_data_dictionary = resample_data(training_data_dictionary, number_of_sensors=number_of_sensors, data_points_per_second=desired_data_points_per_second, upsampling_method=upsampling_method, downsampling_method=downsampling_method, amount_of_numbers_to_group=amount_of_numbers_to_group_in_downsampling, seed=seed)
    test_data_dictionary = resample_data(test_data_dictionary, number_of_sensors=number_of_sensors, data_points_per_second=desired_data_points_per_second, upsampling_method=upsampling_method, downsampling_method=downsampling_method, amount_of_numbers_to_group=amount_of_numbers_to_group_in_downsampling, seed=seed)

    dictionary_statistics(training_data_dictionary, number_of_sensors=number_of_sensors, dictionary_name='Training Data Dictionary')
    dictionary_summary(training_data_dictionary, number_of_sensors=number_of_sensors, segmentID=None, dictionary_name='Training Data Dictionary')

    dictionary_statistics(test_data_dictionary, number_of_sensors=number_of_sensors, dictionary_name='Test Data Dictionary')
    dictionary_summary(test_data_dictionary, number_of_sensors=number_of_sensors, segmentID=None, dictionary_name='Test Data Dictionary')

    #Create dataframes from data
    CSV.write_CSV(data_dictionary=training_data_dictionary, fileName='training_data.csv', number_of_sensors=number_of_sensors)
    CSV.write_CSV(data_dictionary=test_data_dictionary, fileName='test_data.csv', number_of_sensors=number_of_sensors)

    fileNameTrain = 'training_data.csv'
    fileNameTest = 'test_data.csv'
    if cross_validation_evaluation == True:
        #Normalize with z-transform (normalize the resampled data)
        fileNameTrain, mean_arr, std_arr, col_name_arr = CSV.Z_normalize_CSV_Training(fileName='training_data.csv', number_of_sensors=number_of_sensors)
        fileNameTest = CSV.Z_normalize_CSV_Test(fileName='test_data.csv', mean_arr=mean_arr, std_arr=std_arr, col_name_arr=col_name_arr, number_of_sensors=number_of_sensors)

    print("Reducing window size...")
    #Window data
    final_csv_training, length_training_window = CSV.window_data(fileName=fileNameTrain, number_of_sensors=number_of_sensors, data_points_per_second=desired_data_points_per_second, desired_window_size_in_seconds=window_size_in_seconds, overlap_percent=window_overlap)
    #No overlap for test data
    final_csv_test, length_test_window = CSV.window_data(fileName=fileNameTest, number_of_sensors=number_of_sensors, data_points_per_second=desired_data_points_per_second, desired_window_size_in_seconds=window_size_in_seconds, overlap_percent=0)
    if length_training_window != length_test_window:
        length_test_window = -1
    return final_csv_training, final_csv_test, length_test_window, number_of_sensors







def main():
    #For testing out windowing of data
    #CSV.window_data(fileName='test_data.csv', number_of_sensors=1, data_points_per_second=4, desired_window_size_in_seconds=2, overlap_percent=0.5)
    #return 0

    #Set random Seed
    seed = 11
    # The below is necessary for starting Numpy generated random numbers
    # in a well-defined initial state.
    np.random.seed(seed)
    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.
    random.seed(seed)
    # The below set_seed() will make random number generation
    # in the TensorFlow backend have a well-defined initial state.
    # For further details, see:
    # https://www.tensorflow.org/api_docs/python/tf/random/set_seed
    tf.random.set_seed(seed)


    # Variables to change:
    desired_data_points_per_second = 4  # so each second will have 4 data points
    # '''
    window_size_in_seconds = 60
    print("Window Size in Seconds: " + str(window_size_in_seconds))
    window_overlap = 0.1
    number_of_sensors = 2  # number_of_sensors for: accelerometer->0   |    meditag->1    |     accelerometer and meditag->2
    # '''

    batch_size = 35
    epochs = 300
    lr = 0.065 #0.065
    print("Learning Rate for SGD: " + str(lr))
    weights_path = None

    num_neighbors_for_KNN = 1

    name_prefix = 'multi_head_CNN_2'  # ['default_1D_CNN', 'multi_head_CNN_6', 'multi_head_CNN_2', 'multi_head_CNN_3'] Only works if classifier_name is cnn
    classifier_name = 'cnn'  # ['knn', 'cnn']
    cross_validation_evaluation = False
    num_folds = 5



    #TODO: ######################## TRAIN/TEST #############################
    if cross_validation_evaluation == False:

        # Preprocessing
        training_csv_string, test_csv_string, window_length, number_of_sensors = preprocessing_training_test(test_file_path='test/', training_file_path='train/', number_of_sensors=number_of_sensors, desired_data_points_per_second=desired_data_points_per_second, downsampling_method='mean', amount_of_numbers_to_group_in_downsampling=2, upsampling_method='cubic_spline', window_size_in_seconds=window_size_in_seconds, window_overlap=window_overlap, cross_validation_evaluation=cross_validation_evaluation, seed=seed)

        dataset = ModelTrainingUtilityFunctions.Dataset_Loader(training_file=training_csv_string, test_file=test_csv_string, window_length=window_length, number_of_sensors=number_of_sensors)
        print("Training X shape: " + str(dataset.training_X.shape))
        print("Training Y shape: " + str(dataset.training_Y.shape))
        print("Test X shape: " + str(dataset.test_X.shape))
        print("Training Y shape: " + str(dataset.test_Y.shape))

        #training
        num_instances_training, num_dataRows_training, num_features_training, num_outputClasses_training = dataset.training_X.shape[0], dataset.training_X.shape[1], dataset.training_X.shape[2], dataset.training_Y.shape[1]
        print("Training Dataset: " + "\n" + "   Number of instances: " + str(num_instances_training) + "\n" + "   Number of data rows/points per instance: " + str(num_dataRows_training) + "\n" + "   Number of features: " + str(num_features_training) + "\n" + "   Number of classes: " + str(num_outputClasses_training))

        #test
        num_instances_test, num_dataRows_test, num_features_test, num_outputClasses_test = dataset.test_X.shape[0], dataset.test_X.shape[1], dataset.test_X.shape[2], dataset.test_Y.shape[1]
        print("Test Dataset: " + "\n" + "   Number of instances: " + str(num_instances_test) + "\n" + "   Number of data rows/points per instance: " + str(num_dataRows_test) + "\n" + "   Number of features: " + str(num_features_test) + "\n" + "   Number of classes: " + str(num_outputClasses_test))


        print()
        print()

        #Model
        if classifier_name == 'cnn':

            output_folder = name_prefix + "_" + str(epochs) + "_" + str(batch_size) + "___" + training_csv_string.split("__training_data.csv")[0]

            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            model = Model_Library.ModelSelector(num_classes=num_outputClasses_training, timeSteps=num_dataRows_training, features=num_features_training, weights_path=weights_path, model_string=name_prefix, seed=seed)

            # To create a pdf file just change ending to pdf but the code will fail. It needs png to work but it will create pdf succesfully
            plot_model(model, to_file=output_folder + "/" + name_prefix + "_image.png", show_shapes=True, show_layer_names=True, dpi=96)
            model.summary()

            sgd = SGD(lr=lr, momentum=0.0005, nesterov=True, decay=0.00000000000005)
            adm = Adam(lr=0.1, beta_1=0.9, beta_2=0.999, amsgrad=False)
            rms_prop = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=5e-5)
            model.compile(loss=categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])


            #filepath = output_folder + "/weights-{epoch:02d}-{loss:.4f}-{accuracy:.4f}--{val_loss:.4f}-{val_accuracy:.4f}.h5"
            filepath = output_folder + "/best_Weights.h5"
            #checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=True, mode='auto', period=1)
            checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
            csvPath = output_folder + '/history.csv'
            csvLogger = CSVLogger(csvPath, separator=',', append=False)
            early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=60, verbose=1, mode='auto', baseline=None, restore_best_weights=True)
            callbacks_list = [checkpoint, csvLogger]
            #callbacks_list = [csvLogger]


            print("DEBUG: ")
            a = dataset.training_X[:, :, 1]
            b = np.reshape(a, (-1, (desired_data_points_per_second*window_size_in_seconds), 1))
            print(b.shape)


            if name_prefix == 'default_1D_CNN':
                model_history = model.fit(dataset.training_X, dataset.training_Y, batch_size=batch_size, initial_epoch=0, epochs=epochs, callbacks=callbacks_list, verbose=1, validation_data=(dataset.test_X, dataset.test_Y), shuffle=True)
            elif name_prefix == 'multi_head_CNN_6' and number_of_sensors == 2: #Only works with both sensors since it needs input of 6 vectors
                model_history = model.fit([np.reshape(dataset.training_X[:, :, 0], (-1, (desired_data_points_per_second*window_size_in_seconds), 1)),
                                   np.reshape(dataset.training_X[:, :, 1], (-1, (desired_data_points_per_second*window_size_in_seconds), 1)),
                                   np.reshape(dataset.training_X[:, :, 2], (-1, (desired_data_points_per_second*window_size_in_seconds), 1)),
                                   np.reshape(dataset.training_X[:, :, 3], (-1, (desired_data_points_per_second*window_size_in_seconds), 1)),
                                   np.reshape(dataset.training_X[:, :, 4], (-1, (desired_data_points_per_second*window_size_in_seconds), 1)),
                                   np.reshape(dataset.training_X[:, :, 5], (-1, (desired_data_points_per_second*window_size_in_seconds), 1))],
                                   dataset.training_Y, batch_size=batch_size, initial_epoch=0, epochs=epochs,
                                   callbacks=callbacks_list, verbose=1, validation_data=([
                                   np.reshape(dataset.test_X[:, :, 0], (-1, (desired_data_points_per_second*window_size_in_seconds), 1)),
                                   np.reshape(dataset.test_X[:, :, 1], (-1, (desired_data_points_per_second*window_size_in_seconds), 1)),
                                   np.reshape(dataset.test_X[:, :, 2], (-1, (desired_data_points_per_second*window_size_in_seconds), 1)),
                                   np.reshape(dataset.test_X[:, :, 3], (-1, (desired_data_points_per_second*window_size_in_seconds), 1)),
                                   np.reshape(dataset.test_X[:, :, 4], (-1, (desired_data_points_per_second*window_size_in_seconds), 1)),
                                   np.reshape(dataset.test_X[:, :, 5], (-1, (desired_data_points_per_second*window_size_in_seconds), 1))],
                                   dataset.test_Y), shuffle=True)
            elif name_prefix == 'multi_head_CNN_2' and number_of_sensors == 2: #Only works with both sensors since it needs input of 2 vectors
                model_history = model.fit([np.reshape(dataset.training_X[:, :, 0:3], (-1, (desired_data_points_per_second*window_size_in_seconds), 3)),
                                   np.reshape(dataset.training_X[:, :, 3:6], (-1, (desired_data_points_per_second*window_size_in_seconds), 3))],
                                   dataset.training_Y, batch_size=batch_size, initial_epoch=0, epochs=epochs,
                                   callbacks=callbacks_list, verbose=1, validation_data=([
                                   np.reshape(dataset.test_X[:, :, 0:3], (-1, (desired_data_points_per_second*window_size_in_seconds), 3)),
                                   np.reshape(dataset.test_X[:, :, 3:6], (-1, (desired_data_points_per_second*window_size_in_seconds), 3))],
                                   dataset.test_Y), shuffle=True)
            elif name_prefix == 'multi_head_CNN_3' and (number_of_sensors == 0 or number_of_sensors == 1):
                model_history = model.fit([np.reshape(dataset.training_X[:, :, 0], (-1, (desired_data_points_per_second*window_size_in_seconds), 1)),
                                   np.reshape(dataset.training_X[:, :, 1], (-1, (desired_data_points_per_second*window_size_in_seconds), 1)),
                                   np.reshape(dataset.training_X[:, :, 2], (-1, (desired_data_points_per_second*window_size_in_seconds), 1)),],
                                   dataset.training_Y, batch_size=batch_size, initial_epoch=0, epochs=epochs,
                                   callbacks=callbacks_list, verbose=1, validation_data=([
                                   np.reshape(dataset.test_X[:, :, 0], (-1, (desired_data_points_per_second*window_size_in_seconds), 1)),
                                   np.reshape(dataset.test_X[:, :, 1], (-1, (desired_data_points_per_second*window_size_in_seconds), 1)),
                                   np.reshape(dataset.test_X[:, :, 2], (-1, (desired_data_points_per_second*window_size_in_seconds), 1)),],
                                   dataset.test_Y), shuffle=True)


            #Useful for the model only (when saving all weights save_best_only=False)
            #model.save_weights(output_folder + "/" + name_prefix + "_last_weights.h5")


            #Model history metrics
            print("Model Metrics: " + str(model.metrics_names))


            #Load best model
            model.load_weights(output_folder + "/" + "best_Weights.h5")

            # Validation Set Evaluation
            if name_prefix == 'default_1D_CNN':
                test_eval = model.evaluate(dataset.test_X, dataset.test_Y, verbose=1)
            elif name_prefix == 'multi_head_CNN_6' and number_of_sensors == 2: #Only works with both sensors since it needs input of 6 vectors
                test_eval = model.evaluate([np.reshape(dataset.test_X[:, :, 0], (-1, (desired_data_points_per_second*window_size_in_seconds), 1)),
                                   np.reshape(dataset.test_X[:, :, 1], (-1, (desired_data_points_per_second*window_size_in_seconds), 1)),
                                   np.reshape(dataset.test_X[:, :, 2], (-1, (desired_data_points_per_second*window_size_in_seconds), 1)),
                                   np.reshape(dataset.test_X[:, :, 3], (-1, (desired_data_points_per_second*window_size_in_seconds), 1)),
                                   np.reshape(dataset.test_X[:, :, 4], (-1, (desired_data_points_per_second*window_size_in_seconds), 1)),
                                   np.reshape(dataset.test_X[:, :, 5], (-1, (desired_data_points_per_second*window_size_in_seconds), 1))], dataset.test_Y, verbose=1)
            elif name_prefix == 'multi_head_CNN_2' and number_of_sensors == 2: #Only works with both sensors since it needs input of 2 vectors
                test_eval = model.evaluate([np.reshape(dataset.test_X[:, :, 0:3], (-1, (desired_data_points_per_second*window_size_in_seconds), 3)),
                                   np.reshape(dataset.test_X[:, :, 3:6], (-1, (desired_data_points_per_second*window_size_in_seconds), 3))], dataset.test_Y, verbose=1)
            elif name_prefix == 'multi_head_CNN_3' and (number_of_sensors == 0 or number_of_sensors == 1):
                test_eval = model.evaluate([np.reshape(dataset.test_X[:, :, 0], (-1, (desired_data_points_per_second*window_size_in_seconds), 1)),
                                   np.reshape(dataset.test_X[:, :, 1], (-1, (desired_data_points_per_second*window_size_in_seconds), 1)),
                                   np.reshape(dataset.test_X[:, :, 2], (-1, (desired_data_points_per_second*window_size_in_seconds), 1)),], dataset.test_Y, verbose=1)
            print('Test loss:', test_eval[0])
            print('Test accuracy:', test_eval[1])

            model.save(output_folder + "/best_Weights_" + str(test_eval[0]) + "_" + str(test_eval[1]) + ".h5")

            accuracy = model_history.history['accuracy']
            val_accuracy = model_history.history['val_accuracy']
            loss = model_history.history['loss']
            val_loss = model_history.history['val_loss']
            epochs = range(len(accuracy))

            plt.plot(epochs, accuracy, label='Training accuracy', alpha=0.93, linestyle='', color='#8533ff', marker='o')
            plt.plot(epochs, val_accuracy, label='Validation accuracy', alpha=1.0, linestyle='solid', color='#4d0099')
            #plt.title('Training and validation accuracy')
            plt.legend()
            fig1 = plt.gcf()
            plt.show()
            plt.draw()
            fig1.savefig(output_folder + "/" + name_prefix + "_accuracy.pdf", dpi=1000)

            plt.figure()
            plt.plot(epochs, loss, label='Training loss', alpha=0.85, linestyle='', color='#8533ff', marker='o')
            plt.plot(epochs, val_loss, label='Validation loss', alpha=1.0, linestyle='solid', color='#4d0099')
            #plt.title('Training and validation loss')
            plt.legend()
            fig1 = plt.gcf()
            plt.show()
            plt.draw()
            fig1.savefig(output_folder + "/" + name_prefix + "_loss.pdf", dpi=1000)

            # Predict labels in test set
            if name_prefix == 'default_1D_CNN':
                PREDICTED_CLASSES = model.predict(dataset.test_X)
            elif name_prefix == 'multi_head_CNN_6' and number_of_sensors == 2:  # Only works with both sensors since it needs input of 6 vectors
                PREDICTED_CLASSES = model.predict([np.reshape(dataset.test_X[:, :, 0], (-1, (desired_data_points_per_second*window_size_in_seconds), 1)),
                                   np.reshape(dataset.test_X[:, :, 1], (-1, (desired_data_points_per_second*window_size_in_seconds), 1)),
                                   np.reshape(dataset.test_X[:, :, 2], (-1, (desired_data_points_per_second*window_size_in_seconds), 1)),
                                   np.reshape(dataset.test_X[:, :, 3], (-1, (desired_data_points_per_second*window_size_in_seconds), 1)),
                                   np.reshape(dataset.test_X[:, :, 4], (-1, (desired_data_points_per_second*window_size_in_seconds), 1)),
                                   np.reshape(dataset.test_X[:, :, 5], (-1, (desired_data_points_per_second*window_size_in_seconds), 1))])
            elif name_prefix == 'multi_head_CNN_2' and number_of_sensors == 2:  # Only works with both sensors since it needs input of 6 vectors
                PREDICTED_CLASSES = model.predict([np.reshape(dataset.test_X[:, :, 0:3], (-1, (desired_data_points_per_second*window_size_in_seconds), 3)),
                                   np.reshape(dataset.test_X[:, :, 3:6], (-1, (desired_data_points_per_second*window_size_in_seconds), 3))])
            elif name_prefix == 'multi_head_CNN_3' and (number_of_sensors == 0 or number_of_sensors == 1):
                PREDICTED_CLASSES = model.predict([np.reshape(dataset.test_X[:, :, 0], (-1, (desired_data_points_per_second*window_size_in_seconds), 1)),
                                   np.reshape(dataset.test_X[:, :, 1], (-1, (desired_data_points_per_second*window_size_in_seconds), 1)),
                                   np.reshape(dataset.test_X[:, :, 2], (-1, (desired_data_points_per_second*window_size_in_seconds), 1)),])
            #print("PREDICTED_CLASSES.max(): " + str(PREDICTED_CLASSES.max()))
            #print("Predicted classes: " + str(PREDICTED_CLASSES))


            #CHECK CLASSIFICATION PREDICTIONS
            string = "%%%%%%%%%%%%%%%% Predicted Classes %%%%%%%%%%%%%%%%"
            for item in PREDICTED_CLASSES:
                string = string + "\n " + str(item)
            string = string + "\n %%%%%%%%%%%%%%%% Test Labels %%%%%%%%%%%%%%%% \n"
            for item in dataset.test_Y:
                string = string + "\n " + str(item)
            text_file = open(output_folder + "/" + name_prefix + "_TEST_OUTPUT.txt", "w")
            #text_file.write(string)
            text_file.close()



            # To convert from one hot to label
            print('Class probabilities of first image: ' + str(PREDICTED_CLASSES[0]))
            print("True class of first image: " + str(dataset.test_Y[0]))

            print('Class probabilities of third image: ' + str(PREDICTED_CLASSES[2]))
            print("True class of third image: " + str(dataset.test_Y[2]))


            PREDICTED_CLASSES = np.argmax(PREDICTED_CLASSES, axis=1)
            print('Class of first predicted image (np.argmax): ', PREDICTED_CLASSES[0])
            print('Class of third predicted image (np.argmax): ', PREDICTED_CLASSES[2])

            print('Predicted class shape, test class Shape')
            print(PREDICTED_CLASSES.shape, dataset.test_Y.shape)
            print("Converting test_labels to classes (from one hot vector)...")


            TEST_LABELS = np.argmax(dataset.test_Y, axis=1)
            print('Predicted class shape, test class Shape')
            print(PREDICTED_CLASSES.shape, TEST_LABELS.shape)

            # Predictes test images and evaluated with binary relevance (only one class)
            correct_indexes = np.where(PREDICTED_CLASSES == TEST_LABELS)[0]
            print("Found %d correct labels" % len(correct_indexes))

            #print("Correct labels per class: " + str(Counter(correct)))
            #Debo iterar correct (estos son indices) y ver que labels son y en otro que true labels son
            predicted = []
            for i in range(len(correct_indexes)):
                predicted.append(PREDICTED_CLASSES[correct_indexes[i]])
            print("Number of correctly predicted per class" + str(Counter(predicted)))
            print("Number of instances per class in test set: " + str(Counter(TEST_LABELS)))
            TRAIN_LABELS = np.argmax(dataset.training_Y, axis=1)
            print("Number of instances per class in training set: " + str(Counter(TRAIN_LABELS)))
        elif classifier_name == 'knn':
            #https://towardsdatascience.com/dynamic-time-warping-3933f25fcdd
            #dynamic time warping para ver distancia entre dos vectore con varios puntos

            #transformar a 2D el de 3D array -> en vez de poner los sensores paralelamente, ponerlos uno tras el otro
            #En el caso de que no sean del mismo tamanio los arreglos (data vectors), se puede usar dynamic-time-warping para comparar entre


            #https://stackoverflow.com/questions/32034237/how-does-numpys-transpose-method-permute-the-axes-of-an-array
            # X vectors need to be singe vectors, so transposing the matrix into single vector is necessary
            # Vectors are put on below the other
            #Transposing so columns are put together
            num_instances_training = dataset.training_X.shape[0]
            print("num_instances_training: " + str(num_instances_training))
            dataset.training_X = np.transpose(dataset.training_X, axes=(0, 2, 1)) #Transpose inner matrix ()axes 1 and 2 are switched
            dataset.training_X = dataset.training_X.reshape(num_instances_training, -1)

            num_instances_test = dataset.test_X.shape[0]
            print("num_instances_test: " + str(num_instances_test))
            #Needs transposing so all points (in vertical line columns) are near each other sequentially when reshaped and not next to the horizontal points
            dataset.test_X = np.transpose(dataset.test_X, axes=(0, 2, 1))
            dataset.test_X = dataset.test_X.reshape(num_instances_test, -1)

            num_neighbors = num_neighbors_for_KNN
            print("Number of neighbors: " + str(num_neighbors))
            knn = KNeighborsClassifier(n_neighbors=num_neighbors)
            knn.fit(dataset.training_X, dataset.training_Y)
            predicted = knn.predict(dataset.training_X)
            acc = accuracy_score(dataset.training_Y, predicted)
            print("Number of neighbors used: " + str(num_neighbors))
            print("Train Accuracy of KNN: " + str(acc))
            predicted = knn.predict(dataset.test_X)
            acc = accuracy_score(dataset.test_Y, predicted)
            print("Number of neighbors used: " + str(num_neighbors))
            print("Test Accuracy of KNN: " + str(acc))



    #TODO: ######################## CROSS VALIDATION #############################

    elif cross_validation_evaluation == True:
        training_csv_string, test_csv_string, window_length, number_of_sensors = preprocessing_training_test(
            test_file_path='test/', training_file_path='train/', number_of_sensors=number_of_sensors,
            desired_data_points_per_second=desired_data_points_per_second, downsampling_method='mean',
            amount_of_numbers_to_group_in_downsampling=2, upsampling_method='cubic_spline',
            window_size_in_seconds=window_size_in_seconds, window_overlap=window_overlap,
            cross_validation_evaluation=cross_validation_evaluation, seed=seed)

        dataset = ModelTrainingUtilityFunctions.Dataset_Loader(training_file=training_csv_string,
                                                               test_file=test_csv_string, window_length=window_length,
                                                               number_of_sensors=number_of_sensors)

        print("All X data together: " + str(dataset.X.shape))
        print("All Y data together: " + str(dataset.Y.shape))

        num_instances, num_dataRows, num_features, num_outputClasses_training = \
        dataset.X.shape[0], dataset.X.shape[1], dataset.X.shape[2], dataset.Y.shape[
            1]
        print("Dataset: " + "\n" + "   Number of instances: " + str(
            num_instances) + "\n" + "   Number of data rows/points per instance: " + str(
            num_dataRows) + "\n" + "   Number of features: " + str(
            num_features) + "\n" + "   Number of classes: " + str(num_outputClasses_training))



        acc_per_fold = []

        kfold = KFold(n_splits=num_folds, shuffle=True)
        fold_index = 1
        for trainIndexes, testIndexes in kfold.split(dataset.X, dataset.Y):
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%% FOLD " + str(fold_index) + " %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

            dataset.X_crossVal_train = np.array(dataset.X[trainIndexes])
            dataset.Y_crossVal_train = np.array(dataset.Y[trainIndexes])
            dataset.X_crossVal_test = np.array(dataset.X[testIndexes])
            dataset.Y_crossVal_test = np.array(dataset.Y[testIndexes])

            print("Normalizing split...")
            dataset.normalizeCrossVal()

            print("(Train X, Train Y, Test X, Test Y) shapes: ")
            print(dataset.X_crossVal_train.shape, dataset.Y_crossVal_train.shape, dataset.X_crossVal_test.shape, dataset.Y_crossVal_test.shape)





            # Model
            if classifier_name == 'cnn':

                output_folder = name_prefix + "_" + str(epochs) + "_" + str(batch_size) + "___" + \
                                training_csv_string.split("__training_data.csv")[0] + "_crossValidation"

                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)

                # training
                num_instances_training, num_dataRows_training, num_features_training, num_outputClasses_training = \
                dataset.X_crossVal_train.shape[0], dataset.X_crossVal_train.shape[1], dataset.X_crossVal_train.shape[2], \
                dataset.training_Y.shape[1]
                print("Training Dataset: " + "\n" + "   Number of instances: " + str(
                    num_instances_training) + "\n" + "   Number of data rows/points per instance: " + str(
                    num_dataRows_training) + "\n" + "   Number of features: " + str(
                    num_features_training) + "\n" + "   Number of classes: " + str(num_outputClasses_training))

                # test
                num_instances_test, num_dataRows_test, num_features_test, num_outputClasses_test = \
                dataset.X_crossVal_test.shape[0], dataset.X_crossVal_test.shape[1], dataset.X_crossVal_test.shape[2], dataset.Y_crossVal_test.shape[1]
                print("Test Dataset: " + "\n" + "   Number of instances: " + str(
                    num_instances_test) + "\n" + "   Number of data rows/points per instance: " + str(
                    num_dataRows_test) + "\n" + "   Number of features: " + str(
                    num_features_test) + "\n" + "   Number of classes: " + str(num_outputClasses_test))

                model = Model_Library.ModelSelector(num_classes=num_outputClasses_training,
                                                    timeSteps=num_dataRows_training, features=num_features_training,
                                                    weights_path=weights_path, model_string=name_prefix, seed=seed)

                #To create a pdf file just change ending to pdf but the code will fail. It needs png to work but it will create pdf succesfully
                plot_model(model, to_file=output_folder + "/" + name_prefix + "_image.png", show_shapes=True,
                           show_layer_names=True, dpi=96)
                model.summary()

                sgd = SGD(lr=lr, momentum=0.0005, nesterov=True, decay=0.00000000000005)
                adm = Adam(lr=0.1, beta_1=0.9, beta_2=0.999, amsgrad=False)
                rms_prop = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=5e-5)
                model.compile(loss=categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])

                # filepath = output_folder + "/weights-{epoch:02d}-{loss:.4f}-{accuracy:.4f}--{val_loss:.4f}-{val_accuracy:.4f}.h5"
                filepath = output_folder + "/crossValidation" + str(fold_index) + "_best_Weights.h5"
                # checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=True, mode='auto', period=1)
                checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True,
                                             save_weights_only=False, mode='auto', period=1)
                csvPath = output_folder + "/crossValidation" + str(fold_index) +'_history.csv'
                csvLogger = CSVLogger(csvPath, separator=',', append=False)
                early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=60, verbose=1,
                                               mode='auto', baseline=None, restore_best_weights=True)
                callbacks_list = [checkpoint, csvLogger]
                # callbacks_list = [csvLogger]

                if name_prefix == 'default_1D_CNN':
                    model_history = model.fit(dataset.X_crossVal_train, dataset.Y_crossVal_train, batch_size=batch_size,
                                              initial_epoch=0, epochs=epochs, callbacks=callbacks_list, verbose=1,
                                              validation_data=(dataset.X_crossVal_test, dataset.Y_crossVal_test), shuffle=True)
                elif name_prefix == 'multi_head_CNN_6' and number_of_sensors == 2:  # Only works with both sensors since it needs input of 6 vectors
                    model_history = model.fit([np.reshape(dataset.X_crossVal_train[:, :, 0], (
                    -1, (desired_data_points_per_second * window_size_in_seconds), 1)),
                                               np.reshape(dataset.X_crossVal_train[:, :, 1], (
                                               -1, (desired_data_points_per_second * window_size_in_seconds), 1)),
                                               np.reshape(dataset.X_crossVal_train[:, :, 2], (
                                               -1, (desired_data_points_per_second * window_size_in_seconds), 1)),
                                               np.reshape(dataset.X_crossVal_train[:, :, 3], (
                                               -1, (desired_data_points_per_second * window_size_in_seconds), 1)),
                                               np.reshape(dataset.X_crossVal_train[:, :, 4], (
                                               -1, (desired_data_points_per_second * window_size_in_seconds), 1)),
                                               np.reshape(dataset.X_crossVal_train[:, :, 5], (
                                               -1, (desired_data_points_per_second * window_size_in_seconds), 1))],
                                              dataset.Y_crossVal_train, batch_size=batch_size, initial_epoch=0, epochs=epochs,
                                              callbacks=callbacks_list, verbose=1, validation_data=([
                                                                                                        np.reshape(
                                                                                                            dataset.X_crossVal_test[
                                                                                                            :, :, 0], (
                                                                                                            -1, (
                                                                                                                        desired_data_points_per_second * window_size_in_seconds),
                                                                                                            1)),
                                                                                                        np.reshape(
                                                                                                            dataset.X_crossVal_test[
                                                                                                            :, :, 1], (
                                                                                                            -1, (
                                                                                                                        desired_data_points_per_second * window_size_in_seconds),
                                                                                                            1)),
                                                                                                        np.reshape(
                                                                                                            dataset.X_crossVal_test[
                                                                                                            :, :, 2], (
                                                                                                            -1, (
                                                                                                                        desired_data_points_per_second * window_size_in_seconds),
                                                                                                            1)),
                                                                                                        np.reshape(
                                                                                                            dataset.X_crossVal_test[
                                                                                                            :, :, 3], (
                                                                                                            -1, (
                                                                                                                        desired_data_points_per_second * window_size_in_seconds),
                                                                                                            1)),
                                                                                                        np.reshape(
                                                                                                            dataset.X_crossVal_test[
                                                                                                            :, :, 4], (
                                                                                                            -1, (
                                                                                                                        desired_data_points_per_second * window_size_in_seconds),
                                                                                                            1)),
                                                                                                        np.reshape(
                                                                                                            dataset.X_crossVal_test[
                                                                                                            :, :, 5], (
                                                                                                            -1, (
                                                                                                                        desired_data_points_per_second * window_size_in_seconds),
                                                                                                            1))],
                                                                                                    dataset.Y_crossVal_test),
                                              shuffle=True)
                elif name_prefix == 'multi_head_CNN_2' and number_of_sensors == 2:  # Only works with both sensors since it needs input of 2 vectors
                    model_history = model.fit([np.reshape(dataset.X_crossVal_train[:, :, 0:3], (
                    -1, (desired_data_points_per_second * window_size_in_seconds), 3)),
                                               np.reshape(dataset.X_crossVal_train[:, :, 3:6], (
                                               -1, (desired_data_points_per_second * window_size_in_seconds), 3))],
                                              dataset.Y_crossVal_train, batch_size=batch_size, initial_epoch=0, epochs=epochs,
                                              callbacks=callbacks_list, verbose=1, validation_data=([
                                                                                                        np.reshape(
                                                                                                            dataset.X_crossVal_test[
                                                                                                            :, :, 0:3],
                                                                                                            (-1, (
                                                                                                                        desired_data_points_per_second * window_size_in_seconds),
                                                                                                             3)),
                                                                                                        np.reshape(
                                                                                                            dataset.X_crossVal_test[
                                                                                                            :, :, 3:6],
                                                                                                            (-1, (
                                                                                                                        desired_data_points_per_second * window_size_in_seconds),
                                                                                                             3))],
                                                                                                    dataset.Y_crossVal_test),
                                              shuffle=True)
                elif name_prefix == 'multi_head_CNN_3' and (number_of_sensors == 0 or number_of_sensors == 1):
                    model_history = model.fit([np.reshape(dataset.X_crossVal_train[:, :, 0], (-1, (desired_data_points_per_second*window_size_in_seconds), 1)),
                                   np.reshape(dataset.X_crossVal_train[:, :, 1], (-1, (desired_data_points_per_second*window_size_in_seconds), 1)),
                                   np.reshape(dataset.X_crossVal_train[:, :, 2], (-1, (desired_data_points_per_second*window_size_in_seconds), 1)),],
                                   dataset.Y_crossVal_train, batch_size=batch_size, initial_epoch=0, epochs=epochs,
                                   callbacks=callbacks_list, verbose=1, validation_data=([
                                   np.reshape(dataset.X_crossVal_test[:, :, 0], (-1, (desired_data_points_per_second*window_size_in_seconds), 1)),
                                   np.reshape(dataset.X_crossVal_test[:, :, 1], (-1, (desired_data_points_per_second*window_size_in_seconds), 1)),
                                   np.reshape(dataset.X_crossVal_test[:, :, 2], (-1, (desired_data_points_per_second*window_size_in_seconds), 1)),],
                                   dataset.Y_crossVal_test), shuffle=True)

                # Useful for the model only (when saving all weights save_best_only=False)
                # model.save_weights(output_folder + "/" + name_prefix + "_last_weights.h5")

                # Model history metrics
                print("Model Metrics: " + str(model.metrics_names))

                # Load best model
                model.load_weights(output_folder + "/" + "crossValidation" + str(fold_index) + "_best_Weights.h5")

                # Validation Set Evaluation
                if name_prefix == 'default_1D_CNN':
                    test_eval = model.evaluate(dataset.X_crossVal_test, dataset.Y_crossVal_test, verbose=1)
                elif name_prefix == 'multi_head_CNN_6' and number_of_sensors == 2:  # Only works with both sensors since it needs input of 6 vectors
                    test_eval = model.evaluate([np.reshape(dataset.X_crossVal_test[:, :, 0], (
                    -1, (desired_data_points_per_second * window_size_in_seconds), 1)),
                                                np.reshape(dataset.X_crossVal_test[:, :, 1], (
                                                -1, (desired_data_points_per_second * window_size_in_seconds), 1)),
                                                np.reshape(dataset.X_crossVal_test[:, :, 2], (
                                                -1, (desired_data_points_per_second * window_size_in_seconds), 1)),
                                                np.reshape(dataset.X_crossVal_test[:, :, 3], (
                                                -1, (desired_data_points_per_second * window_size_in_seconds), 1)),
                                                np.reshape(dataset.X_crossVal_test[:, :, 4], (
                                                -1, (desired_data_points_per_second * window_size_in_seconds), 1)),
                                                np.reshape(dataset.X_crossVal_test[:, :, 5], (
                                                -1, (desired_data_points_per_second * window_size_in_seconds), 1))],
                                               dataset.Y_crossVal_test, verbose=1)
                elif name_prefix == 'multi_head_CNN_2' and number_of_sensors == 2:  # Only works with both sensors since it needs input of 2 vectors
                    test_eval = model.evaluate([np.reshape(dataset.X_crossVal_test[:, :, 0:3], (
                    -1, (desired_data_points_per_second * window_size_in_seconds), 3)),
                                                np.reshape(dataset.X_crossVal_test[:, :, 3:6], (
                                                -1, (desired_data_points_per_second * window_size_in_seconds), 3))],
                                               dataset.Y_crossVal_test, verbose=1)
                elif name_prefix == 'multi_head_CNN_3' and (number_of_sensors == 0 or number_of_sensors == 1):
                    test_eval = model.evaluate([np.reshape(dataset.X_crossVal_test[:, :, 0], (-1, (desired_data_points_per_second*window_size_in_seconds), 1)),
                                   np.reshape(dataset.X_crossVal_test[:, :, 1], (-1, (desired_data_points_per_second*window_size_in_seconds), 1)),
                                   np.reshape(dataset.X_crossVal_test[:, :, 2], (-1, (desired_data_points_per_second*window_size_in_seconds), 1)),], dataset.Y_crossVal_test, verbose=1)
                print('Test loss:', test_eval[0])
                print('Test accuracy:', test_eval[1])

                model.save(output_folder + "/best_Weights_" + str(test_eval[0]) + "_" + str(test_eval[1]) + ".h5")

                accuracy = model_history.history['accuracy']
                val_accuracy = model_history.history['val_accuracy']
                loss = model_history.history['loss']
                val_loss = model_history.history['val_loss']
                epochs_range = range(len(accuracy))

                plt.plot(epochs, accuracy, label='Training accuracy', alpha=0.85, linestyle='', color='#8533ff', marker='o')
                plt.plot(epochs, val_accuracy, label='Validation accuracy', alpha=1.0, linestyle='solid', color='#4d0099')
                #plt.title('Training and validation accuracy')
                plt.legend()
                fig1 = plt.gcf()
                plt.show()
                plt.draw()
                fig1.savefig(output_folder + "/" + name_prefix + "_accuracy.pdf", dpi=1000)

                plt.figure()
                plt.plot(epochs, loss, label='Training loss', alpha=0.93, linestyle='', color='#8533ff', marker='o')
                plt.plot(epochs, val_loss, label='Validation loss', alpha=1.0, linestyle='solid', color='#4d0099')
                #plt.title('Training and validation loss')
                plt.legend()
                fig1 = plt.gcf()
                plt.show()
                plt.draw()
                fig1.savefig(output_folder + "/" + name_prefix + "_loss.pdf", dpi=1000)

                # Predict labels in test set
                if name_prefix == 'default_1D_CNN':
                    PREDICTED_CLASSES = model.predict(dataset.X_crossVal_test)
                elif name_prefix == 'multi_head_CNN_6' and number_of_sensors == 2:  # Only works with both sensors since it needs input of 6 vectors
                    PREDICTED_CLASSES = model.predict([np.reshape(dataset.X_crossVal_test[:, :, 0], (
                    -1, (desired_data_points_per_second * window_size_in_seconds), 1)),
                                                       np.reshape(dataset.X_crossVal_test[:, :, 1], (
                                                       -1, (desired_data_points_per_second * window_size_in_seconds),
                                                       1)),
                                                       np.reshape(dataset.X_crossVal_test[:, :, 2], (
                                                       -1, (desired_data_points_per_second * window_size_in_seconds),
                                                       1)),
                                                       np.reshape(dataset.X_crossVal_test[:, :, 3], (
                                                       -1, (desired_data_points_per_second * window_size_in_seconds),
                                                       1)),
                                                       np.reshape(dataset.X_crossVal_test[:, :, 4], (
                                                       -1, (desired_data_points_per_second * window_size_in_seconds),
                                                       1)),
                                                       np.reshape(dataset.X_crossVal_test[:, :, 5], (
                                                       -1, (desired_data_points_per_second * window_size_in_seconds),
                                                       1))])
                elif name_prefix == 'multi_head_CNN_2' and number_of_sensors == 2:  # Only works with both sensors since it needs input of 6 vectors
                    PREDICTED_CLASSES = model.predict([np.reshape(dataset.X_crossVal_test[:, :, 0:3], (
                    -1, (desired_data_points_per_second * window_size_in_seconds), 3)),
                                                       np.reshape(dataset.X_crossVal_test[:, :, 3:6], (
                                                       -1, (desired_data_points_per_second * window_size_in_seconds),
                                                       3))])
                elif name_prefix == 'multi_head_CNN_3' and (number_of_sensors == 0 or number_of_sensors == 1):
                    PREDICTED_CLASSES = model.predict([np.reshape(dataset.X_crossVal_test[:, :, 0], (-1, (desired_data_points_per_second*window_size_in_seconds), 1)),
                                   np.reshape(dataset.X_crossVal_test[:, :, 1], (-1, (desired_data_points_per_second*window_size_in_seconds), 1)),
                                   np.reshape(dataset.X_crossVal_test[:, :, 2], (-1, (desired_data_points_per_second*window_size_in_seconds), 1)),])
                # print("PREDICTED_CLASSES.max(): " + str(PREDICTED_CLASSES.max()))
                # print("Predicted classes: " + str(PREDICTED_CLASSES))

                # CHECK CLASSIFICATION PREDICTIONS
                string = "%%%%%%%%%%%%%%%% Predicted Classes %%%%%%%%%%%%%%%%"
                for item in PREDICTED_CLASSES:
                    string = string + "\n " + str(item)
                string = string + "\n %%%%%%%%%%%%%%%% Test Labels %%%%%%%%%%%%%%%% \n"
                for item in dataset.Y_crossVal_test:
                    string = string + "\n " + str(item)
                text_file = open(output_folder + "/" + name_prefix + "crossValidation" + str(fold_index) + "_TEST_OUTPUT.txt", "w")
                # text_file.write(string)
                text_file.close()

                # To convert from one hot to label
                print('Class probabilities of first image: ' + str(PREDICTED_CLASSES[0]))
                print("True class of first image: " + str(dataset.Y_crossVal_test[0]))

                print('Class probabilities of third image: ' + str(PREDICTED_CLASSES[2]))
                print("True class of third image: " + str(dataset.Y_crossVal_test[2]))

                PREDICTED_CLASSES = np.argmax(PREDICTED_CLASSES, axis=1)
                print('Class of first predicted image (np.argmax): ', PREDICTED_CLASSES[0])
                print('Class of third predicted image (np.argmax): ', PREDICTED_CLASSES[2])

                print('Predicted class shape, test class Shape')
                print(PREDICTED_CLASSES.shape, dataset.Y_crossVal_test.shape)
                print("Converting test_labels to classes (from one hot vector)...")

                TEST_LABELS = np.argmax(dataset.Y_crossVal_test, axis=1)
                print('Predicted class shape, test class Shape')
                print(PREDICTED_CLASSES.shape, TEST_LABELS.shape)

                # Predictes test images and evaluated with binary relevance (only one class)
                correct_indexes = np.where(PREDICTED_CLASSES == TEST_LABELS)[0]
                print("Found %d correct labels" % len(correct_indexes))

                # print("Correct labels per class: " + str(Counter(correct)))
                # Debo iterar correct (estos son indices) y ver que labels son y en otro que true labels son
                predicted = []
                for i in range(len(correct_indexes)):
                    predicted.append(PREDICTED_CLASSES[correct_indexes[i]])
                print("Number of correctly predicted per class" + str(Counter(predicted)))
                print("Number of instances per class in test set: " + str(Counter(TEST_LABELS)))
                TRAIN_LABELS = np.argmax(dataset.Y_crossVal_train, axis=1)
                print("Number of instances per class in training set: " + str(Counter(TRAIN_LABELS)))

                val_accuracy = np.array(val_accuracy)
                final_acc = val_accuracy.max()
                acc_per_fold.append(final_acc)

            elif classifier_name == 'knn':
                # https://towardsdatascience.com/dynamic-time-warping-3933f25fcdd
                # dynamic time warping para ver distancia entre dos vectore con varios puntos

                # transformar a 2D el de 3D array -> en vez de poner los sensores paralelamente, ponerlos uno tras el otro
                # En el caso de que no sean del mismo tamanio los arreglos (data vectors), se puede usar dynamic-time-warping para comparar entre

                # https://stackoverflow.com/questions/32034237/how-does-numpys-transpose-method-permute-the-axes-of-an-array
                # X vectors need to be singe vectors, so transposing the matrix into single vector is necessary
                # Vectors are put on below the other
                # Transposing so columns are put together
                num_instances_training = dataset.X_crossVal_train.shape[0]
                print("num_instances_training: " + str(num_instances_training))
                num_instances_test = dataset.X_crossVal_test.shape[0]
                print("num_instances_test: " + str(num_instances_test))
                dataset.trasposeX(num_instances_training, num_instances_test) #Get data flattened and trasposed for knn

                num_instances_training = dataset.X_crossVal_train_trasposed.shape[0]
                print("X_crossVal_train_trasposed: " + str(num_instances_training))
                num_instances_test = dataset.X_crossVal_train_trasposed.shape[0]
                print("X_crossVal_train_trasposed: " + str(num_instances_test))

                num_instances_training = dataset.Y_crossVal_train.shape[0]
                print("Y_crossVal_train: " + str(num_instances_training))
                num_instances_test = dataset.Y_crossVal_test.shape[0]
                print("Y_crossVal_test: " + str(num_instances_test))

                num_neighbors = num_neighbors_for_KNN
                knn = KNeighborsClassifier(n_neighbors=num_neighbors)
                knn.fit(dataset.X_crossVal_train_trasposed, dataset.Y_crossVal_train)
                predicted = knn.predict(dataset.X_crossVal_test_trasposed)
                acc = accuracy_score(dataset.Y_crossVal_test, predicted)
                print("Number of neighbors used: " + str(num_neighbors))
                print("Accuracy of KNN: " + str(acc))
                acc_per_fold.append(acc)
            fold_index = fold_index + 1
        print("All accuracies: " + str(acc_per_fold))
        acc_per_fold = np.array(acc_per_fold)
        mean_acc = np.mean(acc_per_fold)
        std_acc = np.std(acc_per_fold)
        print("Accuracy: " + str(mean_acc) + " +- " + str(std_acc))



#To add cnn models and svm or knn models use this example:
#https://www.kaggle.com/salokr/a-simple-cnn-knn-based-model-to-get-99-score
#but cnn needs to be pretrained first and then extract features

if __name__ == "__main__":
    main()
