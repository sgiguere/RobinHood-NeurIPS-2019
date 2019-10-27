import numpy
import os.path
import scipy.sparse
import sklearn.datasets
import sklearn.decomposition
import sklearn.preprocessing
import sys


class DatasetReader:
    def __init__(self, copy_dataset, verbose):
        self.verbose = verbose

        if copy_dataset is None:
            self.trainFeatures = None
            self.trainLabels = None
            self.testFeatures = None
            self.testLabels = None
        else:
            if copy_dataset.trainFeatures is not None:
                self.trainFeatures = copy_dataset.trainFeatures.copy()
            else:
                self.trainFeatures = None
            if copy_dataset.trainLabels is not None:
                self.trainLabels = copy_dataset.trainLabels.copy()
            else:
                self.trainLabels = None
            if copy_dataset.testFeatures is not None:
                self.testFeatures = copy_dataset.testFeatures
            else:
                self.testFeatures = None
            if copy_dataset.testLabels is not None:
                self.testLabels = copy_dataset.testLabels
            else:
                self.testLabels = None

    def freeAuxiliaryMatrices(self):
        if self.trainFeatures is not None:
            del self.trainFeatures
            del self.trainLabels
            del self.testFeatures
            del self.testLabels
            if self.verbose:
                print("DatasetReader: [Message] Freed matrices")
                sys.stdout.flush()

    def reduceDimensionality(self, numDims):
        if (self.trainFeatures is None) and self.verbose:
            print("DatasetReader: [Error] No training data loaded.")
            sys.stdout.flush()
            return
        LSAdecomp = sklearn.decomposition.TruncatedSVD(n_components = numDims, algorithm = 'arpack')

        LSAdecomp.fit(self.trainFeatures)
        self.trainFeatures = LSAdecomp.transform(self.trainFeatures)
        self.testFeatures = LSAdecomp.transform(self.testFeatures)

        if self.verbose:
            print("DatasetReader: [Message] Features now have shape: Train:",\
                    numpy.shape(self.trainFeatures), "Test:", numpy.shape(self.testFeatures))
            sys.stdout.flush()

    def sanitizeLabels(self, labelList):
        returnList = []
        for tup in labelList:
            if -1 in tup:
                returnList.append(())
            else:
                returnList.append(tup)
        return returnList

    def loadDataset(self, corpusName, labelSubset = None):
        trainFilename = '../DATA/%s_train.svm' % corpusName
        testFilename = '../DATA/%s_test.svm' % corpusName
        if (not os.path.isfile(trainFilename)) or (not os.path.isfile(testFilename)):
            print("DatasetReader: [Error] Invalid corpus name ", trainFilename, testFilename)
            sys.stdout.flush()
            return
        labelTransform = sklearn.preprocessing.MultiLabelBinarizer(sparse_output = False)

        train_features, train_labels = sklearn.datasets.load_svmlight_file(trainFilename, 
            dtype = numpy.longdouble, multilabel = True)
        sanitized_train_labels = self.sanitizeLabels(train_labels)
        
        numSamples, numFeatures = numpy.shape(train_features)

        biasFeatures = scipy.sparse.csr_matrix(numpy.ones((numSamples, 1),
            dtype = numpy.longdouble), dtype = numpy.longdouble)

        self.trainFeatures = scipy.sparse.hstack([train_features, biasFeatures], dtype = numpy.longdouble)
        self.trainFeatures = self.trainFeatures.tocsr()

        test_features, test_labels = sklearn.datasets.load_svmlight_file(testFilename,
            n_features = numFeatures, dtype = numpy.longdouble, multilabel = True)        
        sanitized_test_labels = self.sanitizeLabels(test_labels)

        numSamples, numFeatures = numpy.shape(test_features)

        biasFeatures = scipy.sparse.csr_matrix(numpy.ones((numSamples, 1),
            dtype = numpy.longdouble), dtype = numpy.longdouble)
        self.testFeatures = scipy.sparse.hstack([test_features, biasFeatures], dtype = numpy.longdouble)
        self.testFeatures = self.testFeatures.tocsr()

        self.testLabels = labelTransform.fit_transform(sanitized_test_labels)
        if labelSubset is not None:
            self.testLabels = self.testLabels[:, labelSubset]

        self.trainLabels = labelTransform.transform(sanitized_train_labels)
        if labelSubset is not None:
            self.trainLabels = self.trainLabels[:, labelSubset]

        if self.verbose:
            print("DatasetReader: [Message] Loaded ", corpusName, " [p, q, n_train, n_test]: ",\
                numpy.shape(self.trainFeatures)[1], numpy.shape(self.trainLabels)[1],\
                numpy.shape(self.trainLabels)[0], numpy.shape(self.testFeatures)[0])
            sys.stdout.flush()


class SupervisedDataset(DatasetReader):
    def __init__(self, dataset, verbose):
        DatasetReader.__init__(self, copy_dataset = dataset, verbose = verbose)
        self.validateFeatures = None
        self.validateLabels = None

    def freeAuxiliaryMatrices(self):
        DatasetReader.freeAuxiliaryMatrices(self)
        if self.validateFeatures is not None:
            del self.validateFeatures
            del self.validateLabels

        if self.verbose:
            print("SupervisedDataset: [Message] Freed matrices")
            sys.stdout.flush()

    def createTrainValidateSplit(self, validateFrac):
        self.trainFeatures, self.validateFeatures, self.trainLabels, self.validateLabels = \
            sklearn.model_selection.train_test_split(self.trainFeatures, self.trainLabels,
                test_size = validateFrac, dtype = numpy.longdouble)

        if self.verbose:
            print("SupervisedDataset: [Message] Created supervised split [n_train, n_validate]: ",\
            numpy.shape(self.trainFeatures)[0], numpy.shape(self.validateFeatures)[0])
            sys.stdout.flush()


class BanditDataset(DatasetReader):
    def __init__(self, dataset, verbose):
        DatasetReader.__init__(self, copy_dataset = dataset, verbose = verbose)
        self.sampledLabels = None
        self.sampledLogPropensity = None
        self.sampledLoss = None

        self.validateFeatures = None
        self.validateLabels = None
        self.trainSampledLabels = None
        self.validateSampledLabels = None
        self.trainSampledLogPropensity = None
        self.validateSampledLogPropensity = None
        self.trainSampledLoss = None
        self.validateSampledLoss = None

    def freeAuxiliaryMatrices(self):
        DatasetReader.freeAuxiliaryMatrices(self)
        if self.sampledLabels is not None:
            del self.sampledLabels
            del self.sampledLogPropensity
            del self.sampledLoss

        if self.validateFeatures is not None:
            del self.validateFeatures
            del self.validateLabels
            del self.trainSampledLabels
            del self.validateSampledLabels
            del self.trainSampledLogPropensity
            del self.validateSampledLogPropensity
            del self.trainSampledLoss
            del self.validateSampledLoss

        if self.verbose:
            print("BanditDataset: [Message] Freed matrices")
            sys.stdout.flush()

    def registerSampledData(self, sampledLabels, sampledLogPropensity, sampledLoss):
        self.sampledLabels = sampledLabels
        self.sampledLogPropensity = sampledLogPropensity
        self.sampledLoss = sampledLoss

        if self.verbose:
            print("BanditDataset: [Message] Registered bandit samples [n_samples]: ", numpy.shape(sampledLogPropensity)[0])
            sys.stdout.flush()

    def createTrainValidateSplit(self, validateFrac):
        self.trainFeatures, self.validateFeatures, self.trainLabels, self.validateLabels, \
        self.trainSampledLabels, self.validateSampledLabels, self.trainSampledLogPropensity, self.validateSampledLogPropensity, \
        self.trainSampledLoss, self.validateSampledLoss = sklearn.model_selection.train_test_split(self.trainFeatures,
                                                            self.trainLabels, self.sampledLabels, self.sampledLogPropensity,
                                                            self.sampledLoss, test_size = validateFrac)

        if self.verbose:
            print("BanditDataset: [Message] Created bandit split [n_train, n_validate]:",\
                numpy.shape(self.trainFeatures)[0], numpy.shape(self.validateFeatures)[0])
            sys.stdout.flush()

