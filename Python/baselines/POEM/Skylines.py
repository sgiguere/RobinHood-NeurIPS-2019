from baselines.POEM import MajorizePRM
import numpy
from baselines.POEM import PRM
import scipy.linalg
import sklearn.model_selection
import sklearn.linear_model
import sklearn.metrics
import sklearn.multiclass
import sklearn.svm
import sys
import time
import math


class Skylines:
    def __init__(self, dataset, tol, minC, maxC, verbose, parallel):
        paramArray = numpy.logspace(minC, maxC, num = (maxC - minC + 1), base = 10.0)
        self.params = paramArray.tolist()
        self.tol = tol
        self.verbose = verbose
        self.dataset = dataset
        self.labeler = None
        self.pool = parallel

    def freeAuxiliaryMatrices(self):
        if self.labeler is not None:
            del self.labeler
        del self.dataset
        if self.verbose:
            print( "Skylines: [Message] Freed matrices")
            sys.stdout.flush()

    def parallelValidate(self, param, reportValidationResult):
        predictionError, classifier = self.generateModel(param, reportValidationResult)
        return {'param': param, 'classifier': classifier, 'perf': predictionError}

    def validate(self):
        numSamples, numLabels = numpy.shape(self.dataset.trainLabels)
        numFeatures = numpy.shape(self.dataset.trainFeatures)[1]

        results = None
        start_time = time.time()
        if (self.pool is not None) and (len(self.params) > 1):
            dummy = [True]*len(self.params)
            results = self.pool.map(self.parallelValidate, self.params, dummy)
            results = list(results)
        else:
            results = []
            for param in self.params:
                results.append(self.parallelValidate(param, len(self.params) > 1))
        end_time = time.time()
        avg_time = (end_time - start_time)

        bestPerformance = None
        bestClassifier = None
        bestParam = None

        for result in results:
            param = result['param']
            classifier = result['classifier']
            predictionError = result['perf']
            if len(self.params) > 1:
                predictionError = predictionError * numLabels
                if self.verbose:
                    print(self.Name, " Validation. Parameter = ", param, " Performance: ", predictionError)
                    sys.stdout.flush()
           
            if (bestPerformance is None) or (bestPerformance > predictionError):
                bestPerformance = predictionError
                bestClassifier = classifier
                bestParam = param

        if self.verbose:
            print(self.Name, " Best. Parameter = ", bestParam, "Time: ", avg_time, " Performance: ", bestPerformance)
            sys.stdout.flush()

        self.labeler = bestClassifier
        return avg_time

    def test(self):
        predictedLabels = self.generatePredictions(self.labeler)
        numLabels = numpy.shape(self.dataset.testLabels)[1]
        predictionError = sklearn.metrics.hamming_loss(self.dataset.testLabels,
            predictedLabels) * numLabels

        if self.verbose:
            print(self.Name," Test. Performance: ", predictionError)
            sys.stdout.flush()
        return predictionError 

    def expectedTestLoss(self):
        predictionError = None
        numLabels = numpy.shape(self.dataset.testLabels)[1]
        if self.Name == "SVM":
            predictionError = self.test()
        elif 'CRF' in self.Name:
            numFeatures = numpy.shape(self.dataset.testFeatures)[1]

            predictor = PRM.VanillaISEstimator(n_iter = 0, tol = 0, l2reg = 0,
                varpenalty = 0, clip = 1, verbose = False)
            predictor.coef_ = numpy.zeros((numFeatures,numLabels), dtype = numpy.longdouble)
            for i in range(numLabels):
                if self.labeler[i] is not None:
                    predictor.coef_[:,i] = self.labeler[i].coef_

            if self.verbose:
                print("wNorm", scipy.linalg.norm(predictor.coef_))
                sys.stdout.flush()

            predictionError = predictor.computeExpectedLoss(self.dataset.testFeatures,
                self.dataset.testLabels) * numLabels
        else:
            if self.verbose:
                print("wNorm", scipy.linalg.norm(self.labeler.coef_))
                sys.stdout.flush()

            predictionError = self.labeler.computeExpectedLoss(self.dataset.testFeatures,
                self.dataset.testLabels) * numLabels

        if self.verbose:
            print(self.Name,"Test. Expected Loss: ", predictionError)
            sys.stdout.flush()
        return predictionError


class SVM(Skylines):
    def __init__(self, dataset, tol, minC, maxC, verbose, parallel):
        Skylines.__init__(self, dataset, tol, minC, maxC, verbose, parallel)
        self.Name = "SVM"

    def generateModel(self, param, reportValidationResult):
        svmClassifier = sklearn.svm.LinearSVC(loss = 'l1', C = param, penalty = 'l2',
                dual = True, tol = self.tol, verbose = False, fit_intercept = False)
        multilabelClassifier = sklearn.multiclass.OneVsRestClassifier(svmClassifier, n_jobs=-1)
        multilabelClassifier.fit(self.dataset.trainFeatures, self.dataset.trainLabels)

        predictionError = None
        if reportValidationResult:
            predictedLabels = multilabelClassifier.predict(self.dataset.validateFeatures)

            predictionError = sklearn.metrics.hamming_loss(self.dataset.validateLabels,
                predictedLabels)

        return predictionError, multilabelClassifier

    def generatePredictions(self, classifier):
        return classifier.predict(self.dataset.testFeatures)


class CRF(Skylines):
    def __init__(self, dataset, tol, minC, maxC, verbose, parallel):
        Skylines.__init__(self, dataset, tol, minC, maxC, verbose, parallel)
        self.Name = "CRF"

    def generateModel(self, param, reportValidationResult):
        regressors = []

        predictedLabels = None
        if reportValidationResult:
            predictedLabels = numpy.zeros(numpy.shape(self.dataset.validateLabels), dtype = numpy.int)

        numLabels = numpy.shape(self.dataset.trainLabels)[1]
        for i in range(numLabels):
            currLabels = self.dataset.trainLabels[:, i]
            if currLabels.sum() > 0:        #Avoid training labels with no positive instances
                logitRegressor = sklearn.linear_model.LogisticRegression(C = param,
                    penalty = 'l2', tol = self.tol, dual = True, fit_intercept = False)
                logitRegressor.fit(self.dataset.trainFeatures, currLabels)
                regressors.append(logitRegressor)
                if reportValidationResult:
                    predictedLabels[:,i] = logitRegressor.predict(self.dataset.validateFeatures)
            else:
                regressors.append(None)

        predictionError = None
        if reportValidationResult:
            predictionError = sklearn.metrics.hamming_loss(self.dataset.validateLabels,
                predictedLabels)

        return predictionError, regressors

    def generatePredictions(self, classifiers):
        predictedLabels = numpy.zeros(numpy.shape(self.dataset.testLabels), dtype = numpy.int)
        numLabels = numpy.shape(predictedLabels)[1]
        for i in range(numLabels):
            if classifiers[i] is not None:
                predictedLabels[:,i] = classifiers[i].predict(self.dataset.testFeatures)

        return predictedLabels


class PRMWrapper(Skylines):
    def __init__(self, dataset, n_iter, tol, minC, maxC, minV, maxV, 
                    minClip, maxClip, estimator_type, verbose, parallel, smartStart):
        Skylines.__init__(self, dataset, tol, 0, 0, verbose, parallel)
        self.Name = "PRM("+estimator_type
        if maxV < minV:
            self.Name = self.Name + "-ERM)"
        else:
            self.Name = self.Name + "-SVP)"

        self.params = {}
        numSamples = numpy.shape(self.dataset.trainFeatures)[0]

        if minC <= maxC:
            l2Array = numpy.logspace(minC, maxC, num = (maxC - minC + 1), base = 10.0)
            l2List = l2Array.tolist()
            self.params['l2reg'] = l2List
        else:
            self.params['l2reg'] = [0]

        if minV <= maxV:
            varArray = numpy.logspace(minV, maxV, num = (maxV - minV + 1), base = 10.0)
            varList = varArray.tolist()
            self.params['varpenalty'] = varList
        else:
            self.params['varpenalty'] = [0]

        if minClip <= maxClip:
            clipArray = numpy.logspace(minClip, maxClip, num = (maxClip - minClip + 1), base = 10.0)
            clipArray = numpy.log(clipArray)
            clipList = clipArray.tolist()
            self.params['clip'] = clipList
        else:
            self.params['clip'] = [0]


        self.estimator_type = estimator_type
        self.n_iter = n_iter
        self.smart_start = smartStart

    def calibrateHyperParams(self):
        l2Array = numpy.array(self.params['l2reg'])
        l2penalty = l2Array
        self.params['l2reg'] = l2penalty.tolist()

        numSamples, numLabels = numpy.shape(self.dataset.trainSampledLabels)
        
        percentileMinPropensity = numpy.percentile(self.dataset.trainSampledLogPropensity, 10, interpolation = 'higher')
        percentileMaxPropensity = numpy.percentile(self.dataset.trainSampledLogPropensity, 90, interpolation = 'lower')
        percentileClip = percentileMaxPropensity - percentileMinPropensity

        if percentileClip < 1:
            percentileClip = 1
        if self.verbose:
            print("Calibrating clip to ", percentileClip)
            sys.stdout.flush()

        clipArray = numpy.array(self.params['clip'])
        clip = clipArray + percentileClip
        self.params['clip'] = clip.tolist()

        meanLoss = self.dataset.trainSampledLoss.mean(dtype = numpy.longdouble)
        lossDelta = self.dataset.trainSampledLoss - meanLoss
        sqrtLossVar = scipy.linalg.norm(lossDelta) / numpy.sqrt(numSamples*(numSamples - 1))
        max_val = - meanLoss / sqrtLossVar

        if self.verbose:
            print("Calibrating variance regularizer to ", max_val)
            sys.stdout.flush()

        varArray = numpy.array(self.params['varpenalty'])
        varpenalty = varArray * max_val

        self.params['varpenalty'] = varpenalty.tolist()

        self.params = list(sklearn.model_selection.ParameterGrid(self.params))

    def generateModel(self, param, reportValidationResult):
        predictor = None
        if self.estimator_type == 'Majorize':
            predictor = MajorizePRM.MajorizeISEstimator(n_iter = self.n_iter, tol = self.tol, l2reg = param['l2reg'],
                            varpenalty = param['varpenalty'], clip = param['clip'], verbose = self.verbose)
        elif (self.estimator_type == 'Stochastic'):
            numSamples = numpy.shape(self.dataset.trainSampledLabels)[0]
            epoch_batches = int(numSamples / 100)
            predictor = MajorizePRM.MajorizeStochasticEstimator(n_iter = max(2 * epoch_batches, self.n_iter),
                            min_iter = epoch_batches, tol = self.tol, l2reg = param['l2reg'],
                            varpenalty = param['varpenalty'], clip = param['clip'], verbose = False)
        elif (self.estimator_type == 'SelfNormal'):
            predictor = PRM.SelfNormalEstimator(n_iter = self.n_iter, tol = self.tol, l2reg = param['l2reg'],
                            varpenalty = param['varpenalty'], clip = param['clip'], verbose = self.verbose)
        else:
            predictor = PRM.VanillaISEstimator(n_iter = self.n_iter, tol = self.tol, l2reg = param['l2reg'],
                            varpenalty = param['varpenalty'], clip = param['clip'], verbose = False)

        predictor.setAuxiliaryMatrices(self.dataset.trainFeatures, self.dataset.trainSampledLabels,
            self.dataset.trainSampledLogPropensity, self.dataset.trainSampledLoss)

        predictor.fit(self.smart_start)

        predictionError = None
        if reportValidationResult:
            predictionError = predictor.computeCfactHammingLoss(self.dataset.validateFeatures, self.dataset.validateSampledLabels,
                                self.dataset.validateSampledLoss, self.dataset.validateSampledLogPropensity)

        predictor.freeAuxiliaryMatrices()
        return predictionError, predictor

    def generatePredictions(self, classifier):
        return classifier.predict(self.dataset.testFeatures)

