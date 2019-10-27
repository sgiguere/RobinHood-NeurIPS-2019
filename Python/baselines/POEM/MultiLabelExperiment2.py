from baselines.POEM import DatasetReader
from baselines.POEM import Skylines
from baselines.POEM import Logger
from baselines.POEM import PRM
import numpy
import sys
from baselines.POEM import PDTTest

if __name__ == '__main__':
    exptNum = 1
    pool = None 
    if len(sys.argv) > 1:
        exptNum = int(sys.argv[1])

    if len(sys.argv) > 2:
         import pathos.multiprocessing as mp
         pool = mp.ProcessingPool(7)
    name = 'scene'

    dataset = DatasetReader.DatasetReader(copy_dataset = None, verbose = True)
    if name == 'rcv1_topics':
        dataset.loadDataset(corpusName = name, labelSubset = [33, 59, 70, 102])
    else:
        dataset.loadDataset(corpusName = name)

    svm_scores = []
    crf_scores = []
    crf_expected_scores = []
    logger_scores = []
    logger_map_scores = []
    prm_scores = []
    prm_map_scores = []
    erm_scores = []
    erm_map_scores = []
    poem_scores = []
    poem_map_scores = []
    ermstoch_scores = []
    ermstoch_map_scores = []

    svm_time = []
    crf_time = []
    prm_time = []
    erm_time = []
    poem_time = []
    ermstoch_time = []
    

    streamer = Logger.DataStream(dataset = dataset, verbose = True)
    features, labels = streamer.generateStream(subsampleFrac = 0.05, replayCount = 1)

    subsampled_dataset = DatasetReader.DatasetReader(copy_dataset = dataset, verbose = True)
    subsampled_dataset.trainFeatures = features
    subsampled_dataset.trainLabels = labels
    # subsampled_dataset.createTrainValidateSplit(validateFrac = 0.25)
    logger = Logger.Logger(subsampled_dataset, loggerC = -1, stochasticMultiplier = 1, verbose = True)
    logger_map_scores.append(logger.crf.test())
    logger_scores.append(logger.crf.expectedTestLoss())

    replayed_dataset = DatasetReader.DatasetReader(copy_dataset = dataset, verbose = True)

    features, labels = streamer.generateStream(subsampleFrac = 1.0, replayCount = 4)
    replayed_dataset.trainFeatures = features
    replayed_dataset.trainLabels = labels

    sampledLabels, sampledLogPropensity, sampledLoss = logger.generateLog(replayed_dataset)

    bandit_dataset = DatasetReader.BanditDataset(dataset = replayed_dataset, verbose = True)

    replayed_dataset.freeAuxiliaryMatrices()  
    del replayed_dataset

    bandit_dataset.registerSampledData(sampledLabels, sampledLogPropensity, sampledLoss)
    bandit_dataset.createTrainValidateSplit(validateFrac = 0.25)
    

    import numpy as np
    C = DatasetReader.BanditDataset(dataset = None, verbose = False)
    C.trainFeatures = np.random.random((100,10))
    C.testFeatures = np.random.random((20,10))
    C.trainLabels = np.random.choice(5, size=(100,1))
    C.testLabels = np.random.choice(5, size=(20,1))
    C.registerSampledData(C.trainLabels, np.log(np.random.random(100)), np.random.random(100))
    C.createTrainValidateSplit(0.1)
    bandit_dataset = C
   
    prm = Skylines.PRMWrapper(bandit_dataset, n_iter = 1000, tol = 1e-6, minC = 0, maxC = -1, minV = 0, maxV = 1, 
                                minClip = 0, maxClip = 0, estimator_type = 'Vanilla', verbose = False, 
                                parallel = pool, smartStart = None)
    prm.calibrateHyperParams()
    prm_time.append(prm.validate())
    prm_map_scores.append(prm.test())
    prm_scores.append(prm.expectedTestLoss())
    



    # prm.freeAuxiliaryMatrices()  
    # del prm

    # erm = Skylines.PRMWrapper(bandit_dataset, n_iter = 1000, tol = 1e-6, minC = 0, maxC = -1, minV = 0, maxV = -1, 
    #                             minClip = 0, maxClip = 0, estimator_type = 'Vanilla', verbose = True, 
    #                             parallel = None, smartStart = coef)
    # erm.calibrateHyperParams()
    # erm_time.append(erm.validate())
    # erm_map_scores.append(erm.test())
    # erm_scores.append(erm.expectedTestLoss())
   
    # erm.freeAuxiliaryMatrices()  
    # del erm

    # maj = Skylines.PRMWrapper(bandit_dataset, n_iter = 1000, tol = 1e-6, minC = 0, maxC = -1, minV = -6, maxV = 0,
    #                             minClip = 0, maxClip = 0, estimator_type = 'Stochastic', verbose = True,
    #                             parallel = pool, smartStart = coef)
    # maj.calibrateHyperParams()
    # poem_time.append(maj.validate())
    # poem_map_scores.append(maj.test())
    # poem_scores.append(maj.expectedTestLoss())

    # maj.freeAuxiliaryMatrices()  
    # del maj

    # majerm = Skylines.PRMWrapper(bandit_dataset, n_iter = 1000, tol = 1e-6, minC = 0, maxC = -1, minV = 0, maxV = -1,
    #                             minClip = 0, maxClip = 0, estimator_type = 'Stochastic', verbose = True,
    #                             parallel = None, smartStart = coef)
    # majerm.calibrateHyperParams()
    # ermstoch_time.append(majerm.validate())
    # ermstoch_map_scores.append(majerm.test())
    # ermstoch_scores.append(majerm.expectedTestLoss())

    # majerm.freeAuxiliaryMatrices()  
    # del majerm

    # bandit_dataset.freeAuxiliaryMatrices()  
    # del bandit_dataset
