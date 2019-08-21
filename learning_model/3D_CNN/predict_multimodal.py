import numpy as np
import os, fnmatch
import matplotlib.pyplot as plt
from keras import *

# General configurations
dataset_path = '/media/tnnguyen/7E3B52AF2CE273C0/Thesis/Final-Thesis-Output/raster_imgs/CRSA/dataset/short_term/'
WD = {
    'input': {
        'train' : {
          'factors'    : dataset_path + '/train_2014/in_seq/',
          'predicted'  : dataset_path + '/train_2014/out_seq/'    
        },
        'test' : {
          'factors'    : dataset_path + '/test_2015/in_seq/',
          'predicted'  : dataset_path + '/test_2015/out_seq/'
        }
        
    },    
    'output': {
        'model_weights' : './training_output/model/weight.h5',
        'plots'         : './training_output/monitor/'
    }
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

# Load training data
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

def createBatch(batchSize, dataFiles, trainRatio=.8, mode='train'):
    # Initalize data
    X = {}
    for key in FACTOR.keys():
        X[key] = None
    
    y = {}
    y['default'] = None    
    
    numDataFiles = len(dataFiles)
    i = 0
    while i < batchSize:
        fileId = np.random.randint(low=0, high=int(numDataFiles), size=1)
        fileId = fileId[0]

        try:
            seqName = dataFiles[fileId]
            
            factorData = loadDataFile(WD['input'][mode]['factors'] + seqName)
            predictedData = loadDataFile(WD['input'][mode]['predicted'] + seqName)            
                
            if not (factorData is not None and predictedData is not None):
                continue

            # Load factors and predicted data
            for key in FACTOR.keys():
                X = appendFactorData(key, factorData, X)
            
            y = appendFactorData('default', predictedData, y)

        except Exception:
            continue
        
        i += 1

    del X['default']
    return X, y

# longging
def logging(mode, contentLine):
    f = open(WD['output']['plots'] + 'loss_progress.csv', mode)
    f.write(contentLine)
    f.close()
    
print('Loading testing data...')
testDataFiles = fnmatch.filter(os.listdir(WD['input']['test']['factors']), '*30.npz')
testDataFiles.sort()
numSamples = len(testDataFiles)
print('Nunber of testing data = {0}'.format(numSamples))

###############################
## Define model architecture ##
###############################
# Lower levels models
def buildCNN(cnnInputs, imgShape, filters, kernelSize, factorName, isFusion=False, cnnOutputs=None):
    if isFusion == True:
        cnnInput = layers.add(cnnOutputs, name='Fusion_{0}'.format(factorName))
    else:
        cnnInput = layers.Input(shape=imgShape, name='Input_{0}'.format(factorName))

    for i in range(len(filters)):
        counter = i+1
        if i == 0:
            cnnOutput = cnnInput

        cnnOutput = layers.Conv3D(filters=filters[i], kernel_size=kernelSize, strides=1, padding='same', activation='tanh',
                                  name='Conv3D_{0}{1}'.format(factorName, counter))(cnnOutput)
        cnnOutput = layers.BatchNormalization(name='BN_{0}{1}'.format(factorName, counter))(cnnOutput)
    
    if cnnInputs is not None:
        cnnModel = Model(inputs=cnnInputs, outputs=cnnOutput)
    else:
        cnnModel = Model(inputs=cnnInput, outputs=cnnOutput)
    return cnnModel
    
def buildPrediction(orgInputs, filters, kernelSize, lastOutputs=None):
    predictionOutput = None
    for i in range(len(filters)):
        counter = i + 1
        if i == 0:
            if lastOutputs is not None:
                predictionOutput = lastOutputs
            else:
                predictionOutput = orgInputs
                    
        predictionOutput = layers.Conv3D(filters=filters[i], kernel_size=kernelSize, strides=1, padding='same', activation='sigmoid', 
                                         name='Conv3D_prediction{0}1'.format(counter))(predictionOutput)        
        predictionOutput = layers.Conv3D(filters=filters[i], kernel_size=kernelSize, strides=1, padding='same', activation='relu', 
                                         name='Conv3D_prediction{0}2'.format(counter))(predictionOutput)
        
    predictionOutput = layers.MaxPooling3D(pool_size=(2,1,1), name='output')(predictionOutput)

    predictionOutput = Model(inputs=orgInputs, outputs=predictionOutput)
    return predictionOutput

def buildCompleteModel(imgShape, filtersDict, kernelSizeDict):
    ########################################
    ## Define a CNN model for each factor ##
    ########################################
    filters = filtersDict['factors']
    kernelSize= kernelSizeDict['factors']

    filtersCongestion = list()
    for filter in range(len(filters)-1):
        filtersCongestion.append(int(filters[filter]*2.0))
    filtersCongestion.append(filters[-1])
    
    print(filtersCongestion)
    congestionCNNModel   = buildCNN(cnnInputs=None, imgShape=imgShape, filters=filtersCongestion, kernelSize=kernelSize, factorName='congestion')
    rainfallCNNModel     = buildCNN(cnnInputs=None, imgShape=imgShape, filters=filters, kernelSize=kernelSize, factorName='rainfall')
    #accidentCNNModel     = buildCNN(cnnInputs=None, imgShape=imgShape, filters=filters, kernelSize=kernelSize, factorName='accident')

    ##############################################
    ## Define a fused CNN model for all factors ##
    ##############################################
    filters = filtersDict['factors_fusion']
    kernelSize= kernelSizeDict['factors_fusion']

    fusedCNNModel       = buildCNN(cnnInputs=[congestionCNNModel.input, rainfallCNNModel.input],# accidentCNNModel.input],
                                   cnnOutputs=[congestionCNNModel.output, rainfallCNNModel.output],# accidentCNNModel.output],
                                   imgShape=imgShape,
                                   filters=filters, kernelSize=kernelSize,
                                   factorName='factors', isFusion=True
                                  )

    #################################
    ## Define the prediction model ##
    #################################
    filters = filtersDict['prediction']
    kernelSize= kernelSizeDict['prediction']
    predictionModel     = buildPrediction(orgInputs=[congestionCNNModel.input, rainfallCNNModel.input],# accidentCNNModel.input],
                                          filters=filters,
                                          kernelSize=kernelSize,
                                          lastOutputs=fusedCNNModel.output
                                         )            

    return predictionModel
    
imgShape = (6,20,250,1)
filtersDict = {}; filtersDict['factors'] = [128, 128, 128]; filtersDict['factors_fusion'] = [256, 256, 256, 128]; filtersDict['prediction'] = [64, 1]
kernelSizeDict = {}; kernelSizeDict['factors'] = (3,3,3); kernelSizeDict['factors_fusion'] = (3,3,3); kernelSizeDict['prediction'] = (3,3,3)

predictionModel = buildCompleteModel(imgShape, filtersDict, kernelSizeDict)
predictionModel.load_weights(WD['output']['model_weights'])

header = 'datetime,data_congestion,data_rainfall,data_accident,ground_truth,predicted,err_MSE,err_MAE'
logging('w', header  + '\n')

start = 0 
numSamples = numSamples
for fileId in range(start, numSamples):
    Xtest, ytest = loadTestData(testDataFiles, fileId)

    ypredicted = predictionModel.predict(Xtest['Input_congestion'], 3)

    datetime = testDataFiles[fileId].split('.')[0]

    data_congestion     = np.sum(Xtest['Input_congestion'] * MAX_FACTOR['Input_congestion'])
    data_rainfall       = np.sum(Xtest['Input_rainfall']   * MAX_FACTOR['Input_rainfall'])
    data_accident       = np.sum(Xtest['Input_accident']   * MAX_FACTOR['Input_accident'])    
    
    gt_congestion       = ytest['default']      * MAX_FACTOR['Input_congestion']
    pd_congestion       = ypredicted            * MAX_FACTOR['Input_congestion']
    gt_congestion       = np.reshape(gt_congestion, (1, -1))
    pd_congestion       = np.reshape(pd_congestion, (1, -1))

    error_MSE           = metrics.mean_squared_error(gt_congestion, pd_congestion)
    error_MAE           = metrics.mean_absolute_error(gt_congestion, pd_congestion)
    
    results = '{0},{1},{2},{3},\
               {4},{5},\
               {6},{7}'.format(
                   datetime, data_congestion, data_rainfall, data_accident,
                   np.sum(gt_congestion), np.sum(pd_congestion),
                   error_MSE, error_MAE
               )

    print(results)
    logging('a', results + '\n')