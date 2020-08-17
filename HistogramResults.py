import numpy as np
import matplotlib.pyplot as plt


def plotGraph(x, y, y_label, x_label, title, color='g', output_path=None):
    plt.bar(y, x, color=color)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    #plt.title(title)
    if output_path is not None:
        fig1 = plt.gcf()
        plt.draw()
        title_split = title.split("\n")
        fig1.savefig(output_path + "/" + title_split[0] + "-" + title_split[1] + ".pdf", dpi=1000)
    plt.show()

output_path = 'Histogram Results'
color = '#4d0099' #['#5900b3', '#400080', '#8533ff', '#4d0099']
class_array = [0, 1, 2, 3, 4, 5]
a = "Accuracy"
b = "Activity ID"


#Train/Test
#Accuracy
predictedClasses = np.array([4, 4, 4, 0, 0, 25])
totalClasses = np.array([13, 13, 10, 10, 9, 25])
accuracyPerClass = predictedClasses / totalClasses
# Accuracy: 0.4625       |      Train on 189 samples and test on 80 samples, 300 epochs,
# lr-0.065, 240 data points per instance (4 data points per second and 60 seconds per instance)
plotGraph(accuracyPerClass, class_array, a, b, 'Two-headed CNN with Accelerometer and Meditag Sensors \n Train|Test', color=color, output_path=output_path)

#https://subscription.packtpub.com/book/big_data_and_business_intelligence/9781849513265/1/ch01lvl1sec16/plotting-multiple-bar-charts
# Number of activities in Train/test split
train_class_count = [32, 44, 25, 25, 20, 43]
test_class_count = [13, 13, 10, 10, 9, 25]
X = np.arange(6)
plt.bar(X - 0.125, train_class_count, color='#8533ff', width=0.25, label='Train')
plt.bar(X + 0.125, test_class_count, color='#400080', width=0.25, label='Test')
plt.ylabel("Number of samples")
plt.xlabel("Activity ID")
#plt.title("Samples per activity for train|test")
plt.legend()
fig1 = plt.gcf()
plt.draw()
fig1.savefig(output_path + "/" + "Samples per activity for train|test" + ".pdf", dpi=1000)
plt.show()

'''
#Cross Validation 5-Fold
predictedClasses = np.array([0, 6, 0, 0, 0, 14])
totalClasses = np.array([11, 14, 6, 5, 4, 14])
accuracyPerClass = predictedClasses / totalClasses
#Accuracy: 0.37526208 +- 0.060194865
plotGraph(accuracyPerClass, class_array, a, b, 'Sequential CNN with Accelerometer Sensor \n 5-fold CV', color=color)



predictedClasses = np.array([7, 15, 0, 3, 5, 34])
totalClasses = np.array([41, 37, 35, 21, 27, 50])
accuracyPerClass = predictedClasses / totalClasses
#Accuracy: 0.2872038 +- 0.02255048
plotGraph(accuracyPerClass, class_array, a, b, 'Sequential CNN with Meditag Sensor \n 5-fold CV', color=color)



predictedClasses = np.array([3, 2, 1, 0, 0, 11])
totalClasses = np.array([7, 16, 6, 9, 4, 11])
accuracyPerClass = predictedClasses / totalClasses
#Accuracy: 0.38266945 +- 0.0413862
plotGraph(accuracyPerClass, class_array, a, b, 'Sequential CNN with Accelerometer and Meditag Sensors \n 5-fold CV', color=color)




predictedClasses = np.array([1, 1, 2, 2, 0, 15])
totalClasses = np.array([10, 10, 6, 7, 8, 13])
accuracyPerClass = predictedClasses / totalClasses
#Accuracy: 0.39769393 +- 0.033392243
plotGraph(accuracyPerClass, class_array, a, b, 'Multi-head_6 CNN with Accelerometer and Meditag Sensors \n 5-fold CV', color=color)




predictedClasses = np.array([2, 1, 3, 0, 0, 16])
totalClasses = np.array([10, 10, 6, 7, 8, 13])
accuracyPerClass = predictedClasses / totalClasses
#Accuracy: 0.37526208 +- 0.060194865
plotGraph(accuracyPerClass, class_array, a, b, 'Multi-head_2 CNN with Accelerometer and Meditag Sensors \n 5-fold CV', color=color)
'''

#TRAIN/TEST
# *Accelerometer and Meditag
#Number of correctly predicted per classCounter({5: 25, 2: 4, 0: 4, 1: 4})
#Number of instances per class in test set: Counter({5: 25, 0: 13, 1: 13, 2: 10, 3: 10, 4: 9})





#TODO:  las predicciones estan mal, hacer denuevo y corregir para cross validation!
#CROSS VALIDATION
# *Only accelerometer:
#Number of correctly predicted per classCounter({5: 14, 1: 6})
#Number of instances per class in test set: Counter({1: 14, 5: 14, 0: 11, 2: 6, 3: 5, 4: 4})


# *Only meditag:
#Number of correctly predicted per classCounter({5: 34, 1: 15, 0: 7, 4: 5, 3: 3})
#Number of instances per class in test set: Counter({5: 50, 0: 41, 1: 37, 2: 35, 4: 27, 3: 21})


# *Meditag and accelerometer:

#Simple Sequential CNN:
#Number of correctly predicted per classCounter({5: 11, 0: 3, 1: 2, 2: 1})
#Number of instances per class in test set: Counter({1: 16, 5: 11, 3: 9, 0: 7, 2: 6, 4: 4})

#Multihead 6:
#Number of correctly predicted per classCounter({5: 15, 2: 2, 3: 2, 1: 1, 0: 1})
#Number of instances per class in test set: Counter({5: 13, 2: 10, 0: 10, 4: 8, 3: 7, 1: 6})

#Multihead 2:
#Number of correctly predicted per classCounter({5: 16, 2: 3, 0: 2, 1: 1})
#Number of instances per class in test set: Counter({5: 13, 2: 10, 0: 10, 4: 8, 3: 7, 1: 6})



#KNN with meditag and accelerometer ACC: 0.36065688329839274 +- 0.016073284101174087
