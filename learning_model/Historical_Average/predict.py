import numpy as np
import os, fnmatch
import matplotlib.pyplot as plt
from sklearn import metrics as M

#######################
## Configure dataset ##
#######################
dataset_path = '/media/tnnguyen/7E3B52AF2CE273C0/Thesis/Final-Thesis-Output/raster_imgs/CRSA/dataset/medium_term'
WD = {
    'input': {
        'test' : {
          'factors'    : dataset_path + '/test_2015/in_seq/',
          'predicted'  : dataset_path + '/test_2015/out_seq/'
        }
    },    
    'loss': './evaluation/medium.csv'
}

FACTOR = {
    # factor channel index
    'Input_congestion'        : 0,
    'Input_rainfall'          : 1,
    'Input_accident'          : 2,
    'default'                 : 0
}

MAX_FACTOR = {
    'Input_congestion'        : 5000,
    'Input_rainfall'          : 150,
    'Input_accident'          : 1,
    'default'                 : 5000,
}

print('Loading testing data...')
testDataFiles = fnmatch.filter(os.listdir(WD['input']['test']['factors']), '*30.npz')
testDataFiles.sort()
numSamples = len(testDataFiles)
print('Nunber of testing data = {0}'.format(numSamples))

###########################################
## Load data for training and evaluating ##
###########################################
def loadDataFile(path):
    try:
        data = np.load(path)
        data = data['arr_0']
    except Exception:
        data = None

    return data

def appendFactorData(factorName, factorData, X):
    # Load data
    data = factorData[:, :, :, FACTOR[factorName]]
    data = np.expand_dims(data, axis=3)
    data = np.expand_dims(data, axis=0)
    
    if factorName == 'Input_accident':
        data[data > 0] = 1
    
    # Standardize data
    data = data.astype(float)
    data /= MAX_FACTOR[factorName]

    if X[factorName] is None:
        X[factorName] = data
    else:
        X[factorName] = np.vstack((X[factorName], data))

    return X

def loadTestData(dataFiles, fileId):
    # Initalize data
    X = {}
    for key in FACTOR.keys():
        X[key] = None
    
    y = {}
    y['default'] = None    

    seqName = dataFiles[fileId]
    
    factorData = loadDataFile(WD['input']['test']['factors'] + seqName)
    predictedData = loadDataFile(WD['input']['test']['predicted'] + seqName)         

    # Load factors and predicted data
    for key in FACTOR.keys():
        X = appendFactorData(key, factorData, X)
    
    y = appendFactorData('default', predictedData, y)

    del X['default']
    return X, y

def logging(mode, contentLine):
    f = open(WD['loss'], mode)
    f.write(contentLine)
    f.close()

#############################
## Build statistical model ##
#############################
def calculateHA(data, predictStep):
    data = data[0]
    predicted = np.average(data, axis=0)
    predicted = np.expand_dims(predicted, axis=0)
    for step in range(predictStep - 2):
      predicted = np.concatenate((predicted, predicted, predicted))
    predicted = np.expand_dims(predicted, axis=0)

    return predicted

################
## Evaluation ##
################
header = 'datetime,data_congestion,data_rainfall,data_accident,ground_truth,predicted,err_MSE,err_MAE'
logging('w', header  + '\n')

start = 0 
numSamples = numSamples
for fileId in range(start, numSamples):
    Xtest, ytest = loadTestData(testDataFiles, fileId)
    ypredicted = calculateHA(Xtest['Input_congestion'], 3)

    datetime = testDataFiles[fileId].split('.')[0]

    data_congestion     = np.sum(Xtest['Input_congestion'] * MAX_FACTOR['Input_congestion'])
    data_rainfall       = np.sum(Xtest['Input_rainfall']   * MAX_FACTOR['Input_rainfall'])
    data_accident       = np.sum(Xtest['Input_accident']   * MAX_FACTOR['Input_accident'])    
    
    gt_congestion       = ytest['default']      * MAX_FACTOR['Input_congestion']
    pd_congestion       = ypredicted            * MAX_FACTOR['Input_congestion']
    gt_congestion       = np.reshape(gt_congestion, (1, -1))
    pd_congestion       = np.reshape(pd_congestion, (1, -1))

    error_MSE           = M.mean_squared_error(gt_congestion, pd_congestion)
    error_MAE           = M.mean_absolute_error(gt_congestion, pd_congestion)
    
    results = '{0},{1},{2},{3},\
               {4},{5},\
               {6},{7}'.format(
                   datetime, data_congestion, data_rainfall, data_accident,
                   np.sum(gt_congestion), np.sum(pd_congestion),
                   error_MSE, error_MAE
               )

    print(results)
    logging('a', results + '\n')
