import pandas as pd
import random
from collections import Counter
import math
import random
import numpy as np
import scipy
import scipy.interpolate
import csv


def load_class_label_dataframe(file_path='test/activities_test.csv'):
    dataframe = pd.read_csv(file_path)
    # Original claass numbers: Vital signs measurements=2   |   Blood Collection=3   |   Blood Glucose Measurement=4   |   Indwelling drip retention and connection=6   |   Oral care=9   |   Diaper exchange and cleaning of area=12
    # New class numbers: Vital signs measurements=0   |   Blood Collection=1   |   Blood Glucose Measurement=2   |   Indwelling drip retention and connection=3   |   Oral care=4   |   Diaper exchange and cleaning of area=5
    dataframe["activity_id"].replace(to_replace={2: 0, 3: 1, 4: 2, 6: 3, 9: 4, 12: 5},
                                     inplace=True)  # replace class values
    dataframe = dataframe.astype(dtype={"segment_id": str})  # convert segment_id column to string
    return dataframe


def load_accelerometer_dataframe(file_path='test/accelerometer_test.csv'):
    dataframe = pd.read_csv(file_path)
    dataframe = dataframe.astype(dtype={"segment_id": 'int32'})  # convert segment_id column to int
    dataframe = dataframe.astype(dtype={"segment_id": str})  # convert segment_id column to string
    return dataframe


def load_meditag_dataframe(file_path='test/meditag_test.csv'):
    dataframe = pd.read_csv(file_path)
    dataframe["pressure"].fillna(value=0, inplace=True)  # convert nan to 0
    dataframe = dataframe.astype(dtype={"segment_id": str})  # convert segment_id column to string
    return dataframe


#Easier: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
def z_score_normalization_TrainingSet(dataframe):
    column_names_to_normalize = ['x', 'y', 'z', 'pressure']
    std_arr = []
    mean_arr = []
    col_name_arr = []
    for column in dataframe.columns:
        if column in column_names_to_normalize:
            col_name_arr.append(column)
            new_column_name = column + '1'
            mean = dataframe[column].mean()
            mean_arr.append(mean)
            std = dataframe[column].std()
            std_arr.append(std)
            dataframe[new_column_name] = (dataframe[column] - mean) / std
            del dataframe[column]
            dataframe.rename(columns={new_column_name: column}, inplace=True)
    return dataframe, mean_arr, std_arr, col_name_arr

#Easier: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
def z_score_normalization_TestSet(dataframe, mean_arr, std_arr, col_name_arr):
    #mean_arr and std_arr have the same order as col_name_arr, so it will be for these columns
    column_names_to_normalize = col_name_arr
    for column in dataframe.columns:
        if column in column_names_to_normalize:
            index = column_names_to_normalize.index(column) #get index of column in column_names_to_normalize
            new_column_name = column + '1'
            dataframe[new_column_name] = (dataframe[column] - mean_arr[index]) / std_arr[index]
            del dataframe[column]
            dataframe.rename(columns={new_column_name: column}, inplace=True)
    return dataframe


def create_data_dictionary(class_label_dataframe, accelerometer_dataframe, meditag_dataframe, number_of_sensors=2):
    data_dictionary = {}
    segmentID_prefix = 'segmentID_'

    # Reading class labels
    for index, row in class_label_dataframe.iterrows():
        segment_id = row['segment_id']
        if segmentID_prefix + str(segment_id) not in data_dictionary:
            data_dictionary[segmentID_prefix + str(segment_id)] = {}
            data_dictionary[segmentID_prefix + str(segment_id)]['class_label'] = row['activity_id']
        else:
            raise Exception('Duplicate key found: .'.format(segmentID_prefix + str(segment_id)))

    if number_of_sensors == 0: # 0 For sensor accelerometer
        # Reading accelerometer data
        for index, row in accelerometer_dataframe.iterrows():
            segment_id = row['segment_id']
            if segmentID_prefix + str(segment_id) in data_dictionary:
                if 'time_elapsed_accelerometer' not in data_dictionary[segmentID_prefix + str(segment_id)]:
                    data_dictionary[segmentID_prefix + str(segment_id)]['time_elapsed_accelerometer'] = [row['time_elapsed']]
                else:
                    data_dictionary[segmentID_prefix + str(segment_id)]['time_elapsed_accelerometer'].append(
                        row['time_elapsed'])
                if 'x_accelerometer' not in data_dictionary[segmentID_prefix + str(segment_id)]:
                    data_dictionary[segmentID_prefix + str(segment_id)]['x_accelerometer'] = [row['x']]
                else:
                    data_dictionary[segmentID_prefix + str(segment_id)]['x_accelerometer'].append(row['x'])
                if 'y_accelerometer' not in data_dictionary[segmentID_prefix + str(segment_id)]:
                    data_dictionary[segmentID_prefix + str(segment_id)]['y_accelerometer'] = [row['y']]
                else:
                    data_dictionary[segmentID_prefix + str(segment_id)]['y_accelerometer'].append(row['y'])
                if 'z_accelerometer' not in data_dictionary[segmentID_prefix + str(segment_id)]:
                    data_dictionary[segmentID_prefix + str(segment_id)]['z_accelerometer'] = [row['z']]
                else:
                    data_dictionary[segmentID_prefix + str(segment_id)]['z_accelerometer'].append(row['z'])
            else:
                raise Exception('Segment ID key not found in dictionary: .'.format(segmentID_prefix + str(segment_id)))
    elif number_of_sensors == 1: # 1 For sensor meditag
        #Only adding time_elapsed for accelerometer
        for index, row in accelerometer_dataframe.iterrows():
            segment_id = row['segment_id']
            if segmentID_prefix + str(segment_id) in data_dictionary:
                if 'time_elapsed_accelerometer' not in data_dictionary[segmentID_prefix + str(segment_id)]:
                    data_dictionary[segmentID_prefix + str(segment_id)]['time_elapsed_accelerometer'] = [row['time_elapsed']]
                else:
                    data_dictionary[segmentID_prefix + str(segment_id)]['time_elapsed_accelerometer'].append(row['time_elapsed'])

        for index, row in meditag_dataframe.iterrows():
            segment_id = row['segment_id']
            if segmentID_prefix + str(segment_id) in data_dictionary:
                if 'time_elapsed_meditag' not in data_dictionary[segmentID_prefix + str(segment_id)]:
                    data_dictionary[segmentID_prefix + str(segment_id)]['time_elapsed_meditag'] = [
                        row['time_elapsed']]
                else:
                    data_dictionary[segmentID_prefix + str(segment_id)]['time_elapsed_meditag'].append(
                        row['time_elapsed'])
                if 'x_meditag' not in data_dictionary[segmentID_prefix + str(segment_id)]:
                    data_dictionary[segmentID_prefix + str(segment_id)]['x_meditag'] = [row['x']]
                else:
                    data_dictionary[segmentID_prefix + str(segment_id)]['x_meditag'].append(row['x'])
                if 'y_meditag' not in data_dictionary[segmentID_prefix + str(segment_id)]:
                    data_dictionary[segmentID_prefix + str(segment_id)]['y_meditag'] = [row['y']]
                else:
                    data_dictionary[segmentID_prefix + str(segment_id)]['y_meditag'].append(row['y'])
                if 'pressure_meditag' not in data_dictionary[segmentID_prefix + str(segment_id)]:
                    data_dictionary[segmentID_prefix + str(segment_id)]['pressure_meditag'] = [row['pressure']]
                else:
                    data_dictionary[segmentID_prefix + str(segment_id)]['pressure_meditag'].append(row['pressure'])
            else:
                raise Exception(
                    'Segment ID key not found in dictionary: .'.format(segmentID_prefix + str(segment_id)))
        # Get staring first and last points of accelerometer time_elapsed data and copy them to meditag time+elapsed
        id_to_remove = []
        for segment_id in data_dictionary.keys():
            if (len(data_dictionary[segment_id].keys()) == 6):
                data_dictionary[segment_id]['time_elapsed_meditag'][0] = data_dictionary[segment_id]['time_elapsed_accelerometer'][0]
                data_dictionary[segment_id]['time_elapsed_meditag'][-1] = data_dictionary[segment_id]['time_elapsed_accelerometer'][-1]
            else:
                id_to_remove.append(segment_id) #add id to array to delete all instances that dont have accelerometer data
        for id in id_to_remove:
            del data_dictionary[id]
        for segment_id in data_dictionary.keys():
            if (len(data_dictionary[segment_id].keys()) == 6):
                del data_dictionary[segment_id]['time_elapsed_accelerometer'] #delete time_elapsed_accelerometer array
    elif number_of_sensors == 2: # 2 For sensors meditag and accelerometer
        # Reading accelerometer data
        for index, row in accelerometer_dataframe.iterrows():
            segment_id = row['segment_id']
            if segmentID_prefix + str(segment_id) in data_dictionary:
                if 'time_elapsed_accelerometer' not in data_dictionary[segmentID_prefix + str(segment_id)]:
                    data_dictionary[segmentID_prefix + str(segment_id)]['time_elapsed_accelerometer'] = [row['time_elapsed']]
                else:
                    data_dictionary[segmentID_prefix + str(segment_id)]['time_elapsed_accelerometer'].append(row['time_elapsed'])
                if 'x_accelerometer' not in data_dictionary[segmentID_prefix + str(segment_id)]:
                    data_dictionary[segmentID_prefix + str(segment_id)]['x_accelerometer'] = [row['x']]
                else:
                    data_dictionary[segmentID_prefix + str(segment_id)]['x_accelerometer'].append(row['x'])
                if 'y_accelerometer' not in data_dictionary[segmentID_prefix + str(segment_id)]:
                    data_dictionary[segmentID_prefix + str(segment_id)]['y_accelerometer'] = [row['y']]
                else:
                    data_dictionary[segmentID_prefix + str(segment_id)]['y_accelerometer'].append(row['y'])
                if 'z_accelerometer' not in data_dictionary[segmentID_prefix + str(segment_id)]:
                    data_dictionary[segmentID_prefix + str(segment_id)]['z_accelerometer'] = [row['z']]
                else:
                    data_dictionary[segmentID_prefix + str(segment_id)]['z_accelerometer'].append(row['z'])
            else:
                raise Exception('Segment ID key not found in dictionary: .'.format(segmentID_prefix + str(segment_id)))
        # Reading meditag data
        for index, row in meditag_dataframe.iterrows():
            segment_id = row['segment_id']
            if segmentID_prefix + str(segment_id) in data_dictionary:
                if 'time_elapsed_meditag' not in data_dictionary[segmentID_prefix + str(segment_id)]:
                    data_dictionary[segmentID_prefix + str(segment_id)]['time_elapsed_meditag'] = [
                        row['time_elapsed']]
                else:
                    data_dictionary[segmentID_prefix + str(segment_id)]['time_elapsed_meditag'].append(
                        row['time_elapsed'])
                if 'x_meditag' not in data_dictionary[segmentID_prefix + str(segment_id)]:
                    data_dictionary[segmentID_prefix + str(segment_id)]['x_meditag'] = [row['x']]
                else:
                    data_dictionary[segmentID_prefix + str(segment_id)]['x_meditag'].append(row['x'])
                if 'y_meditag' not in data_dictionary[segmentID_prefix + str(segment_id)]:
                    data_dictionary[segmentID_prefix + str(segment_id)]['y_meditag'] = [row['y']]
                else:
                    data_dictionary[segmentID_prefix + str(segment_id)]['y_meditag'].append(row['y'])
                if 'pressure_meditag' not in data_dictionary[segmentID_prefix + str(segment_id)]:
                    data_dictionary[segmentID_prefix + str(segment_id)]['pressure_meditag'] = [row['pressure']]
                else:
                    data_dictionary[segmentID_prefix + str(segment_id)]['pressure_meditag'].append(row['pressure'])
            else:
                raise Exception(
                    'Segment ID key not found in dictionary: .'.format(segmentID_prefix + str(segment_id)))
        for segment_id in data_dictionary.keys():
            if (len(data_dictionary[segment_id].keys()) == 9):
                data_dictionary[segment_id]['time_elapsed_meditag'][0] = data_dictionary[segment_id]['time_elapsed_accelerometer'][0]
                data_dictionary[segment_id]['time_elapsed_meditag'][-1] = data_dictionary[segment_id]['time_elapsed_accelerometer'][-1]
    return data_dictionary


def data_dictionary_time_elapsed_correction(data_dictionary, number_of_sensors=2):
    # Use ceiling function to every last time_elapsed data to have a rounded number and floor for first value
    for segmentID in data_dictionary.keys():
        if number_of_sensors == 0: #for accelerometer
            if (len(data_dictionary[segmentID].keys()) == 5):
                data_dictionary[segmentID]['time_elapsed_accelerometer'][0] = math.floor(data_dictionary[segmentID]['time_elapsed_accelerometer'][0])
                data_dictionary[segmentID]['time_elapsed_accelerometer'][-1] = math.ceil(data_dictionary[segmentID]['time_elapsed_accelerometer'][-1])
        elif number_of_sensors == 1: #for meditag
            if (len(data_dictionary[segmentID].keys()) == 5):
                data_dictionary[segmentID]['time_elapsed_meditag'][0] = math.floor(data_dictionary[segmentID]['time_elapsed_meditag'][0])
                data_dictionary[segmentID]['time_elapsed_meditag'][-1] = math.ceil(data_dictionary[segmentID]['time_elapsed_meditag'][-1])
        elif number_of_sensors == 2: #for meditag and accelerometer
            if (len(data_dictionary[segmentID].keys()) == 9):
                data_dictionary[segmentID]['time_elapsed_accelerometer'][0] = math.floor(data_dictionary[segmentID]['time_elapsed_accelerometer'][0])
                data_dictionary[segmentID]['time_elapsed_accelerometer'][-1] = math.ceil(data_dictionary[segmentID]['time_elapsed_accelerometer'][-1])
                data_dictionary[segmentID]['time_elapsed_meditag'][0] = math.floor(data_dictionary[segmentID]['time_elapsed_meditag'][0])
                data_dictionary[segmentID]['time_elapsed_meditag'][-1] = math.ceil(data_dictionary[segmentID]['time_elapsed_meditag'][-1])
    return data_dictionary


def remove_incomplete_data(data_dictionary, number_of_sensors=2):
    number_of_data_before = len(data_dictionary)
    number_of_data_after = 0
    segmentID_to_remove = []
    for segmentID in data_dictionary.keys():
        if number_of_sensors == 0: #for accelerometer
            if (len(data_dictionary[segmentID].keys()) != 5):
                segmentID_to_remove.append(segmentID)
        elif number_of_sensors == 1: # for meditag
            if (len(data_dictionary[segmentID].keys()) != 5):
                segmentID_to_remove.append(segmentID)
        elif number_of_sensors == 2: #for meditag and accelerometer
            if (len(data_dictionary[segmentID].keys()) != 9):
                segmentID_to_remove.append(segmentID)
    for segmentID in segmentID_to_remove:
        del data_dictionary[segmentID]
    number_of_data_after = len(data_dictionary)
    return data_dictionary, number_of_data_before, number_of_data_after


def dictionary_fileds_for_segmentID(data_dictionary, segmentID='segmentID_12', number_of_sensors=2):
    if number_of_sensors == 0:  # for accelerometer
        print("  ---Fields for " + str(segmentID) + "---")
        print("     Data arrays: " + str(data_dictionary[segmentID].keys()))
        print("     Number of data arrays: " + str(len(data_dictionary[segmentID])))
        print("     Number of data points for the Accelerometer data: " + str(len(data_dictionary[segmentID]['x_accelerometer'])))
        print("     Data points for the Accelerometer X data: " + str(data_dictionary[segmentID]['x_accelerometer']))
        print("     Data points for the Accelerometer time duration data: " + str(data_dictionary[segmentID]['time_elapsed_accelerometer']))
    elif number_of_sensors == 1:  # for meditag
        print("  ---Fields for " + str(segmentID) + "---")
        print("     Data arrays: " + str(data_dictionary[segmentID].keys()))
        print("     Number of data arrays: " + str(len(data_dictionary[segmentID])))
        print("     Number of data points for the Meditag data: " + str(len(data_dictionary[segmentID]['pressure_meditag'])))
        print("     Data points for the Meditag Pressure data: " + str(data_dictionary[segmentID]['pressure_meditag']))
        print("     Data points for the Meditag time duration data: " + str(data_dictionary[segmentID]['time_elapsed_meditag']))
    if number_of_sensors == 2: # for meditag and for accelerometer
        print("  ---Fields for " + str(segmentID) + "---")
        print("     Data arrays: " + str(data_dictionary[segmentID].keys()))
        print("     Number of data arrays: " + str(len(data_dictionary[segmentID])))
        print("     Number of data points for the Accelerometer data: " + str(len(data_dictionary[segmentID]['x_accelerometer'])))
        print("     Data points for the Accelerometer X data: " + str(data_dictionary[segmentID]['x_accelerometer']))
        print("     Data points for the Accelerometer time duration data: " + str(data_dictionary[segmentID]['time_elapsed_accelerometer']))
        print("     Number of data points for the Meditag data: " + str(len(data_dictionary[segmentID]['pressure_meditag'])))
        print("     Data points for the Meditag Pressure data: " + str(data_dictionary[segmentID]['pressure_meditag']))
        print("     Data points for the Meditag time duration data: " + str(data_dictionary[segmentID]['time_elapsed_meditag']))


def dictionary_summary(data_dictionary, number_of_sensors=2, segmentID=None, dictionary_name="Dictionary Name"):
    print("*** " + dictionary_name + " Summary ***")
    print("     Dictionary keys: " + str(data_dictionary.keys()))
    print("     Number of dictionary keys: " + str(len(data_dictionary)))
    if segmentID is None:
        random_segmentID = random.choice(list(data_dictionary.keys()))
        dictionary_fileds_for_segmentID(data_dictionary, segmentID=random_segmentID, number_of_sensors=number_of_sensors)
    else:
        dictionary_fileds_for_segmentID(data_dictionary, segmentID=segmentID, number_of_sensors=number_of_sensors)
    print("*** END ***")


def dictionary_statistics(data_dictionary, number_of_sensors=2, dictionary_name="Dictionary Name"):
    # Look at the length of data arrays (I assume x, y, z have same length since sensors produce tuples of three always)
    accelerometer_x_array_size = []
    meditag_x_array_size = []
    segmentID_key = []
    class_label = []
    activity_duration_meditag = []
    activity_duration_accelerometer = []
    for segmentID in data_dictionary.keys():
        if number_of_sensors == 0: #for accelerometer
            segmentID_key.append(segmentID)
            class_label.append(data_dictionary[segmentID]['class_label'])
            accelerometer_x_array_size.append(len(data_dictionary[segmentID]['x_accelerometer']))
            activity_duration_accelerometer.append(data_dictionary[segmentID]['time_elapsed_accelerometer'][-1])
        elif number_of_sensors == 1: #for meditag
            segmentID_key.append(segmentID)
            class_label.append(data_dictionary[segmentID]['class_label'])
            meditag_x_array_size.append(len(data_dictionary[segmentID]['x_meditag']))
            activity_duration_meditag.append(data_dictionary[segmentID]['time_elapsed_meditag'][-1])
        elif number_of_sensors == 2:  # 2 is for meditag and accelerometer!
            segmentID_key.append(segmentID)
            class_label.append(data_dictionary[segmentID]['class_label'])
            accelerometer_x_array_size.append(len(data_dictionary[segmentID]['x_accelerometer']))
            activity_duration_accelerometer.append(data_dictionary[segmentID]['time_elapsed_accelerometer'][-1])
            meditag_x_array_size.append(len(data_dictionary[segmentID]['x_meditag']))
            activity_duration_meditag.append(data_dictionary[segmentID]['time_elapsed_meditag'][-1])


    print("*** " + dictionary_name + " Statistics ***")
    print("     Number of segmentIDs: " + str(len(segmentID_key)))
    if number_of_sensors == 2 or number_of_sensors == 0:
        print("     Length of accelerometer array: occurrences -> " + str(Counter(accelerometer_x_array_size)))
        print("     Duration of accelerometer activities: occurrences -> " + str(Counter(activity_duration_accelerometer)))
    if number_of_sensors == 2 or number_of_sensors == 1:
        print("     Length of meditag array: occurrences -> " + str(Counter(meditag_x_array_size)))
        print("     Duration of meditag activities: occurrences -> " + str(Counter(activity_duration_meditag)))
    print("*** END ***")
    return segmentID_key, class_label, accelerometer_x_array_size, activity_duration_accelerometer, meditag_x_array_size, activity_duration_meditag


def data_array_upsample(array=None, time_duration_seconds=60, data_points_per_second=4, upsampling_method='cubic_spline'):
    # https://stackoverflow.com/questions/4072844/add-more-sample-points-to-data
    x = np.arange(start=0, stop=len(array), step=1)
    y = array
    new_length = time_duration_seconds * data_points_per_second
    new_x = np.linspace(x.min(), x.max(), new_length)
    if upsampling_method == 'cubic_spline':
        new_y = scipy.interpolate.interp1d(x, y, kind='cubic')(new_x)
        array = new_y
    return array


def data_array_downsample(array=None, time_duration_seconds=60, data_points_per_second=4, downsampling_method='mean', amount_of_numbers_to_group=2, seed=11):
    random.seed(seed)
    downsampled_array = []
    null_number = -99999999

    if downsampling_method == 'mean':
        desired_number_of_points = time_duration_seconds * data_points_per_second
        while len(array) != desired_number_of_points:
            excess_points = len(array) - desired_number_of_points
            used_indices = []
            if excess_points * amount_of_numbers_to_group <= len(array):  # if few excess points, pick random indices to average
                while len(used_indices) < excess_points:
                    index = random.randint(0, len(array)-amount_of_numbers_to_group)
                    if index not in used_indices:
                        used_indices.append(index)
                used_indices.sort()
                for i in range(len(used_indices)):
                    average = 0
                    for j in range(amount_of_numbers_to_group):
                        average = average + array[used_indices[i]+j]
                        array[used_indices[i]+j] = null_number
                    average = average / amount_of_numbers_to_group
                    array[used_indices[i]] = average
                for i in range(len(array)):
                    if array[i] != null_number:
                        downsampled_array.append(array[i])
                array = downsampled_array
                downsampled_array = []
            elif excess_points * amount_of_numbers_to_group > len(array):  # if excess points are many, aveerage all points
                remainder = len(array) % amount_of_numbers_to_group
                a = (len(array)-remainder)/amount_of_numbers_to_group
                start = 0
                stop = len(array)-remainder-amount_of_numbers_to_group
                step = amount_of_numbers_to_group
                for i in range(start, stop, step):
                    average = 0
                    for j in range(amount_of_numbers_to_group):
                        average = average + array[i + j]
                    average = average / amount_of_numbers_to_group
                    downsampled_array.append(average)
                average = 0
                for j in range(stop, len(array), 1):
                    average = average + array[j]
                average = average / (len(array)-stop)
                downsampled_array.append(average)
                array = downsampled_array
                downsampled_array = []
    return array

#TODO: Try plotting all samples and seeing how to interpolate and group depending on the distribution (lo que el mauricio me recomendo)
def resample_data(data_dictionary, number_of_sensors=2, data_points_per_second=4, downsampling_method='mean', amount_of_numbers_to_group=2, upsampling_method='cubic_spline', seed=11):
    #Here I assume that all sensor readings have the same duration (which is the accelerometer duration since meditag does not have correct durations)
    accelerometer_array_prefix = ['x', 'y', 'z']
    meditag_array_prefix = ['x', 'y', 'pressure']
    main_array_name = ''
    for segment_ID in data_dictionary.keys():
        if number_of_sensors == 0 or number_of_sensors == 2: #for accelerometer
            desired_data_points = data_points_per_second * data_dictionary[segment_ID]['time_elapsed_accelerometer'][-1] #disred is the duration in seconds of the sample times the number of data points data_points_per_second
            main_array_name = 'x_accelerometer'
        elif number_of_sensors == 1: # for meditag only
            desired_data_points = data_points_per_second * data_dictionary[segment_ID]['time_elapsed_meditag'][-1] #disred is the duration in seconds of the sample times the number of data points data_points_per_second
            main_array_name = 'x_meditag'
        #print("DEBUG duration: " + str(data_dictionary[segment_ID]['time_elapsed_accelerometer'][-1]))
        if len(data_dictionary[segment_ID][main_array_name]) < desired_data_points: #interpolation is needed!!
            if number_of_sensors == 0 or number_of_sensors == 2:
                for prefix in accelerometer_array_prefix:
                    #print("DEBUG accelerometer before: " + str(len(data_dictionary[segment_ID][prefix + '_accelerometer'])))
                    data_dictionary[segment_ID][prefix + '_accelerometer'] = data_array_upsample(array=data_dictionary[segment_ID][prefix + '_accelerometer'], time_duration_seconds=data_dictionary[segment_ID]['time_elapsed_accelerometer'][-1], data_points_per_second=data_points_per_second, upsampling_method=upsampling_method)
                    #print("DEBUG accelerometer after: " + str(len(data_dictionary[segment_ID][prefix + '_accelerometer'])))
            if number_of_sensors == 1 or number_of_sensors == 2:
                for prefix in meditag_array_prefix:
                    #print("DEBUG meditag before: " + str(len(data_dictionary[segment_ID][prefix + '_meditag'])))
                    data_dictionary[segment_ID][prefix + '_meditag'] = data_array_upsample(array=data_dictionary[segment_ID][prefix + '_meditag'], time_duration_seconds=data_dictionary[segment_ID]['time_elapsed_meditag'][-1], data_points_per_second=data_points_per_second, upsampling_method=upsampling_method)
                    #print("DEBUG meditag after: " + str(len(data_dictionary[segment_ID][prefix + '_meditag'])))
        elif len(data_dictionary[segment_ID][main_array_name]) > desired_data_points: #downsampling is needed
            if number_of_sensors == 0 or number_of_sensors == 2:
                for prefix in accelerometer_array_prefix:
                    #print("DEBUG accelerometer before: " + str(len(data_dictionary[segment_ID][prefix + '_accelerometer'])))
                    data_dictionary[segment_ID][prefix + '_accelerometer'] = data_array_downsample(array=data_dictionary[segment_ID][prefix + '_accelerometer'], time_duration_seconds=data_dictionary[segment_ID]['time_elapsed_accelerometer'][-1], data_points_per_second=data_points_per_second, downsampling_method=downsampling_method, amount_of_numbers_to_group=amount_of_numbers_to_group, seed=seed)
                    #print("DEBUG accelerometer after: " + str(len(data_dictionary[segment_ID][prefix + '_accelerometer'])))
            if number_of_sensors == 1 or number_of_sensors == 2:
                for prefix in meditag_array_prefix:
                    #print("DEBUG meditag before: " + str(len(data_dictionary[segment_ID][prefix + '_meditag'])))
                    data_dictionary[segment_ID][prefix + '_meditag'] = data_array_downsample(array=data_dictionary[segment_ID][prefix + '_meditag'], time_duration_seconds=data_dictionary[segment_ID]['time_elapsed_meditag'][-1], data_points_per_second=data_points_per_second, downsampling_method=downsampling_method, amount_of_numbers_to_group=amount_of_numbers_to_group, seed=seed)
                    #print("DEBUG meditag after: " + str(len(data_dictionary[segment_ID][prefix + '_meditag'])))
        elif len(data_dictionary[segment_ID][main_array_name]) == desired_data_points: #do nothing
            pass
    return data_dictionary


class CSV:
    @staticmethod
    def write_CSV(data_dictionary=None, fileName='CSV.csv', number_of_sensors=2):
        if number_of_sensors == 0: #5 arrays for accelerometer
            header = ['segmnetID', 'class_label', 'x_accelerometer', 'y_accelerometer', 'z_accelerometer']
            with open(fileName, 'wt') as file:
                csv_writer = csv.writer(file)
                csv_writer.writerow(header)
                row = []
                for segmentID in data_dictionary:  # iterate over segmentID
                    for i in range(len(data_dictionary[segmentID]['x_accelerometer'])):  # iterate over the length of the data arrays
                        row.insert(0, segmentID)
                        row.insert(1, data_dictionary[segmentID]['class_label'])
                        row.insert(2, data_dictionary[segmentID]['x_accelerometer'][i])
                        row.insert(3, data_dictionary[segmentID]['y_accelerometer'][i])
                        row.insert(4, data_dictionary[segmentID]['z_accelerometer'][i])
                        csv_writer.writerow(row)
                        row.clear()
        elif number_of_sensors == 1: #5 arrays for meditag
            header = ['segmnetID', 'class_label', 'x_meditag', 'y_meditag', 'pressure_meditag']
            with open(fileName, 'wt') as file:
                csv_writer = csv.writer(file)
                csv_writer.writerow(header)
                row = []
                for segmentID in data_dictionary:  # iterate over segmentID
                    for i in range(len(data_dictionary[segmentID]['x_meditag'])):  # iterate over the length of the data arrays
                        row.insert(0, segmentID)
                        row.insert(1, data_dictionary[segmentID]['class_label'])
                        row.insert(2, data_dictionary[segmentID]['x_meditag'][i])
                        row.insert(3, data_dictionary[segmentID]['y_meditag'][i])
                        row.insert(4, data_dictionary[segmentID]['pressure_meditag'][i])
                        csv_writer.writerow(row)
                        row.clear()
        if number_of_sensors == 2: #9 arrays for meditag and accelerometer
            header = ['segmnetID', 'class_label', 'x_accelerometer', 'y_accelerometer', 'z_accelerometer', 'x_meditag', 'y_meditag', 'pressure_meditag']
            with open(fileName, 'wt') as file:
                csv_writer = csv.writer(file)
                csv_writer.writerow(header)
                row = []
                for segmentID in data_dictionary: #iterate over segmentID
                    for i in range(len(data_dictionary[segmentID]['x_accelerometer'])): #iterate over the length of the data arrays
                        row.insert(0, segmentID)
                        row.insert(1, data_dictionary[segmentID]['class_label'])
                        row.insert(2, data_dictionary[segmentID]['x_accelerometer'][i])
                        row.insert(3, data_dictionary[segmentID]['y_accelerometer'][i])
                        row.insert(4, data_dictionary[segmentID]['z_accelerometer'][i])
                        row.insert(5, data_dictionary[segmentID]['x_meditag'][i])
                        row.insert(6, data_dictionary[segmentID]['y_meditag'][i])
                        row.insert(7, data_dictionary[segmentID]['pressure_meditag'][i])
                        csv_writer.writerow(row)
                        row.clear()

    #Easier way to normalize only training data: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    @staticmethod
    def Z_normalize_CSV_Training(fileName='CSV.csv', number_of_sensors=2):
        dataframe = pd.read_csv(fileName)
        std_arr = []
        mean_arr = []
        col_name_arr = []
        if number_of_sensors == 2 or number_of_sensors == 1 or number_of_sensors == 0:
            column_names_to_normalize = ['x_accelerometer', 'y_accelerometer', 'z_accelerometer', 'x_meditag', 'y_meditag', 'pressure_meditag']
            for column in dataframe.columns:
                if column in column_names_to_normalize:
                    col_name_arr.append(column)
                    new_column_name = column + '1'
                    mean = dataframe[column].mean()
                    mean_arr.append(mean)
                    std = dataframe[column].std()
                    std_arr.append(std)
                    dataframe[new_column_name] = (dataframe[column] - mean) / std
                    del dataframe[column]
                    dataframe.rename(columns={new_column_name: column}, inplace=True)
        dataframe.to_csv(fileName, index=False)
        return fileName, mean_arr, std_arr, col_name_arr

    # Easier: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    @staticmethod
    def Z_normalize_CSV_Test(fileName='CSV.csv', mean_arr=None, std_arr=None, col_name_arr=None, number_of_sensors=2):
        dataframe = pd.read_csv(fileName)
        if number_of_sensors == 2 or number_of_sensors == 1 or number_of_sensors == 0:
            column_names_to_normalize = col_name_arr
            for column in dataframe.columns:
                if column in column_names_to_normalize:
                    index = column_names_to_normalize.index(column)  # get index of column in column_names_to_normalize
                    new_column_name = column + '1'
                    dataframe[new_column_name] = (dataframe[column] - mean_arr[index]) / std_arr[index]
                    del dataframe[column]
                    dataframe.rename(columns={new_column_name: column}, inplace=True)
        dataframe.to_csv(fileName, index=False)
        return fileName

    @staticmethod
    # All instances need to have same length here
    def window_data(fileName='CSV.csv', number_of_sensors=2, data_points_per_second=4, desired_window_size_in_seconds=2, overlap_percent=0.5):
        csv_string = ''
        window_length = data_points_per_second * desired_window_size_in_seconds
        if (window_length * overlap_percent) % 1 == 0: #Make sure overlap is an integer
            dataframe = pd.read_csv(fileName)
            windowed_dataset = []  # extend with each window
            window = []  # array of rows with all columns. Number of rows is window_length
            row = []
            current_segmentID = None
            previous_segmentID = None

            for i in range(len(dataframe)):
                #Set ID for current and previous (also add overlap if necessary)
                current_segmentID = dataframe.iloc[i, 0]
                if len(window) > 0:
                    previous_segmentID = window[-1][0]
                elif len(window) == 0 and len(windowed_dataset) > 0:
                    previous_segmentID = windowed_dataset[-1][0]
                    # get the overlap in window IF current_segmentID == previous_segmentID, otherwise no overlap
                    if current_segmentID == previous_segmentID:
                        #Copy overlap to window
                        num_rows_to_get_with_overlap = int(window_length * overlap_percent)
                        start = len(windowed_dataset) - num_rows_to_get_with_overlap
                        stop = len(windowed_dataset)
                        step = 1
                        for y in range(start, stop, step):  # goes from  to [len(windowed_dataset)-num_rows_to_get_with_overlap] to [len(windowed_dataset)] or last index
                            window.append(windowed_dataset[y].copy())
                    elif current_segmentID != previous_segmentID:
                        #It is the start of a new ID so no overlap is possible
                        pass
                elif len(window) == 0 and len(windowed_dataset) == 0: #if it is the very first row of the dataframe
                    previous_segmentID = current_segmentID

                #Get new row
                row = []
                for j in range(len(dataframe.columns)):  # just fill an array with row content
                    row.insert(j, dataframe.iloc[i, j])

                #Add row to window
                if len(window) < window_length and current_segmentID == previous_segmentID:
                    #add row to window
                    window.append(row)
                elif len(window) < window_length and current_segmentID != previous_segmentID:
                    #discard window, add row to new window
                    window = []
                    window.append(row)

                if len(window) == window_length:  # if window is full with size window_length, extend windowed_dataset with another window
                    windowed_dataset.extend(window)
                    window = []

            # Add new windowID (same ID for each group of size window_length)
            window_ID = -1
            for i in range(len(windowed_dataset)):
                if (i/window_length) % 1 == 0:
                    window_ID = window_ID + 1
                windowed_dataset[i].insert(1, 'windowID_' + str(window_ID))

            csv_string = 'modifiedWindowLength' + str(window_length) + "_overlap" + str(overlap_percent) + "__" + fileName

            if number_of_sensors == 0: #for accelerometer
                header = ['segmnetID', 'windowID', 'class_label', 'x_accelerometer', 'y_accelerometer', 'z_accelerometer']
            elif number_of_sensors == 1: #for meditag
                header = ['segmnetID', 'windowID', 'class_label', 'x_meditag', 'y_meditag', 'pressure_meditag']
            elif number_of_sensors == 2:
                header = ['segmnetID', 'windowID', 'class_label', 'x_accelerometer', 'y_accelerometer', 'z_accelerometer', 'x_meditag', 'y_meditag', 'pressure_meditag']

            with open(csv_string, 'wt') as file:
                csv_writer = csv.writer(file)
                csv_writer.writerow(header)
                for row in windowed_dataset:
                    csv_writer.writerow(row)
                    row.clear()

        else:
            raise Exception('Window length (data_points_per_second * desired_window_size_in_seconds = {}) * overlap_percent ({}) should be an integer!: '.format(window_length, overlap_percent))
        return csv_string, window_length

