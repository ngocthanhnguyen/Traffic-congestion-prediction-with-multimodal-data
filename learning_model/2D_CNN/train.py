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
        'model_weights' : './training_output/model/',
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

# logging training progress
def logTrainingProgress(mode, contentLine):
    f = open(WD['output']['plots'] + 'loss_progress.csv', mode)
    f.write(contentLine)
    f.close()

# MSE
def mean_squared_error_eval(y_true, y_pred):
    return backend.eval(backend.mean(backend.square(y_pred - y_true)))
    
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

        cnnOutput = layers.Conv2D(filters=filters[i], kernel_size=kernelSize, strides=1, padding='same', activation='tanh',
                                  name='Conv2D_{0}{1}'.format(factorName, counter))(cnnOutput)
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
                    
        predictionOutput = layers.Conv2D(filters=filters[i], kernel_size=kernelSize, strides=1, padding='same', activation='sigmoid', 
                                         name='Conv2D_prediction{0}1'.format(counter))(predictionOutput)        
        predictionOutput = layers.Conv2D(filters=filters[i], kernel_size=kernelSize, strides=1, padding='same', activation='relu', 
                                         name='Conv2D_prediction{0}2'.format(counter))(predictionOutput)
        
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
        filtersCongestion.append(int(filters[filter]*1.0))
    filtersCongestion.append(filters[-1])
    
    print(filtersCongestion)
    congestionCNNModel   = buildCNN(cnnInputs=None, imgShape=imgShape, filters=filtersCongestion, kernelSize=kernelSize, factorName='congestion')

    #################################
    ## Define the prediction model ##
    #################################
    filters = filtersDict['prediction']
    kernelSize= kernelSizeDict['prediction']
    predictionModel     = buildPrediction(orgInputs=[congestionCNNModel.input],
                                          filters=filters,
                                          kernelSize=kernelSize,
                                          lastOutputs=congestionCNNModel.output
                                         )            

    return predictionModel

# Get the list of factors data files
print('Loading training data...')
trainDataFiles = fnmatch.filter(os.listdir(WD['input']['train']['factors']), '*30.npz')
trainDataFiles.sort()
numSamples = len(trainDataFiles)
print('Nunber of training data = {0}'.format(numSamples))

print('Loading testing data...')
testDataFiles = fnmatch.filter(os.listdir(WD['input']['test']['factors']), '*30.npz')
testDataFiles.sort()
numSamples = len(testDataFiles)
print('Nunber of testing data = {0}'.format(numSamples))

batchSize = 1
numIterations = 45000

###############################
## Define model architecture ##
###############################
imgShape = (20,250,6)
targetImgShape=(20,250,3)
filtersDict = {}; filtersDict['factors'] = [256, 256, 128, 256, 256, 256, 128]; filtersDict['prediction'] = [64, targetImgShape[2]]
kernelSizeDict = {}; kernelSizeDict['factors'] = (3,3); kernelSizeDict['prediction'] = (3,3)

predictionModel = buildCompleteModel(imgShape, filtersDict, kernelSizeDict)
predictionModel.summary()
utils.plot_model(predictionModel,to_file='architecture.png',show_shapes=True)

from tensorflow.contrib.opt import LazyAdamOptimizer

tfopt = LazyAdamOptimizer()
lr = 5e-5
predictionModel.compile(optimizer=optimizers.Adam(lr=lr, decay=1e-5),
                        loss='mse',
                        metrics=['mse']
                       )
print('After configuring model...')

##############
## Training ##
##############
from logger import Logger
train_logger = Logger('./training_output/tensorboard/train')
test_logger = Logger('./training_output/tensorboard/test')

trainLosses = list()
testLosses = list()
start = 1

for iteration in range(start, numIterations):
    # ============ Training progress ============#
    X, y = createBatch(batchSize, trainDataFiles)
    trainLoss = predictionModel.train_on_batch(X, y['default'])

    # test per epoch
    Xtest, ytest = createBatch(1, testDataFiles, mode='test')      
    ypredicted = predictionModel.predict(Xtest)
    
    testLoss = mean_squared_error_eval(ytest['default'], ypredicted)

    # ============ TensorBoard logging ============#
    # Log the scalar values
    train_info = {
        'loss': trainLoss[0],
    }
    test_info = {
        'loss': testLoss,
    }

    for tag, value in train_info.items():
        train_logger.scalar_summary(tag, value, step=epoch)
    for tag, value in test_info.items():
        test_logger.scalar_summary(tag, value, step=epoch)
    
    trainLosses.append(trainLoss[0])
    testLosses.append(testLoss)    
    
    # save model checkpoint
    if iteration % 100 == 0:   
        # save model weight
        predictionModel.save_weights(WD['output']['model_weights'] \
                                     + 'epoch_' + str(iteration) + '.h5')
