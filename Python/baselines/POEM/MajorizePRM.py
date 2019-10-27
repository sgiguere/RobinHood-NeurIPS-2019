import numpy
import baselines.POEM.PRM as PRM
import scipy.linalg
import scipy.optimize
import scipy.special
import sys


class MajorizeISEstimator(PRM.VanillaISEstimator): 
    def computeConstant(self, w):
        currW = w.astype(numpy.longdouble)
        numSamples = numpy.shape(self.logpropensity)[0]
        numLabels = numpy.shape(self.y)[1]
        numFeatures = numpy.shape(self.X)[1]
        currW = numpy.reshape(currW, (numFeatures, numLabels))
       
        WX = self.X.dot(currW)
        YWX = numpy.multiply(WX, self.ySign)
        ProbPerInstanceLabel = scipy.special.expit(YWX)

        zeroMask = ProbPerInstanceLabel <= 0
        ProbPerInstanceLabel[zeroMask] = 1.0
        zeroEntries = zeroMask.sum(axis = 1, dtype = numpy.int)
        zeroMask = zeroEntries > 0

        logProbPerInstanceLabel = numpy.log(ProbPerInstanceLabel)
        logProbPerInstance = numpy.sum(logProbPerInstanceLabel, axis = 1, dtype = numpy.longdouble)

        logImportanceSampleWeights = logProbPerInstance - self.logpropensity
        mask = numpy.logical_and(logImportanceSampleWeights >= self.clip, numpy.logical_not(zeroMask))
        logImportanceSampleWeights[mask] = self.clip

        if self.verbose:
            print("MajorizeISEstimator: [Debug] C=", mask.sum())
            sys.stdout.flush()

        ImportanceSampleWeights = numpy.exp(logImportanceSampleWeights)
        ImportanceSampleWeights[zeroMask] = 0.0

        WeightedLossPerInstance = numpy.multiply(self.sampledLoss, ImportanceSampleWeights)
        
        #This is Mean(u_w)
        meanWeightedLoss = WeightedLossPerInstance.mean(dtype = numpy.longdouble)

        if self.varpenalty <= 0:
            return 1.0, 0.0, 0.0, meanWeightedLoss

        diffWtLossPerInstance = WeightedLossPerInstance - meanWeightedLoss
        sqrtVar = scipy.linalg.norm(diffWtLossPerInstance)
        N_1 = numSamples - 1
        sqrtN_1 = numpy.sqrt(N_1)
        sqrtN = numpy.sqrt(numSamples)
        sqrtVar = sqrtVar / sqrtN_1

        if sqrtVar < self.tol:
            sqrtVar = self.tol

        meanConstant = 1.0 - (self.varpenalty * sqrtN * meanWeightedLoss) / (N_1 * sqrtVar)     #Factor sqrt_n
        sqConstant = self.varpenalty / (2 * N_1 * sqrtVar * sqrtN)      #Factor sqrt_n
        CConstant = (sqrtVar * self.varpenalty) / (2 * sqrtN) + (self.varpenalty * sqrtN * meanWeightedLoss * meanWeightedLoss) / (2 * N_1 * sqrtVar)   #Factor sqrt_n
        Obj = meanWeightedLoss + self.varpenalty * sqrtVar / sqrtN #Factor sqrt_n

        return meanConstant, sqConstant, CConstant, Obj

    """
    def Objective(self, w):
        w = w.astype(numpy.longdouble)
        numSamples = numpy.shape(self.logpropensity)[0]
        numLabels = numpy.shape(self.y)[1]
        numFeatures = numpy.shape(self.X)[1]
        currW = numpy.reshape(w, (numFeatures, numLabels))

        WX = self.X.dot(currW)
        YWX = numpy.multiply(WX, self.ySign)
        ProbPerInstanceLabel = scipy.special.expit(YWX)

        zeroMask = ProbPerInstanceLabel <= 0
        ProbPerInstanceLabel[zeroMask] = 1.0
        zeroEntries = zeroMask.sum(axis = 1, dtype = numpy.int)
        zeroMask = zeroEntries > 0

        logProbPerInstanceLabel = numpy.log(ProbPerInstanceLabel)
        logProbPerInstance = numpy.sum(logProbPerInstanceLabel, axis = 1, dtype = numpy.longdouble)

        logImportanceSampleWeights = logProbPerInstance - self.logpropensity
        mask = numpy.logical_and(logImportanceSampleWeights >= self.clip, numpy.logical_not(zeroMask))
        logImportanceSampleWeights[mask] = self.clip

        if self.verbose:
            print("MAJ C=", mask.sum())

        ImportanceSampleWeights = numpy.exp(logImportanceSampleWeights)
        ImportanceSampleWeights[zeroMask] = 0.0

        WeightedLossPerInstance = numpy.multiply(self.sampledLoss, ImportanceSampleWeights)

        sqWeightedLossPerInstance = numpy.square(WeightedLossPerInstance)
        
        #This is \hat{Err}(w)
        meanWeightedLoss = WeightedLossPerInstance.mean(dtype = numpy.longdouble)
        sumSqWeightedLoss = sqWeightedLossPerInstance.sum(dtype = numpy.longdouble)

        #Computing Grad \hat{Err}(w)
        PartitionPerInstanceLabel = scipy.special.expit(WX)
        LabelPartitionPerInstanceLabel = self.y - PartitionPerInstanceLabel
        LabelPartitionPerInstanceLabel[mask, :] = 0

        LossLabelPartitionPerInstanceLabel = numpy.multiply(LabelPartitionPerInstanceLabel,
            WeightedLossPerInstance[:,None])

        #This is Grad \hat{Err}(w)
        g = self.Xtranspose.dot(LossLabelPartitionPerInstanceLabel) / numSamples

        WeightedGradient = numpy.multiply(2*LossLabelPartitionPerInstanceLabel,
           WeightedLossPerInstance[:,None])

        gSq = self.Xtranspose.dot(WeightedGradient)
        
        L2Obj = 0
        L2Grad = numpy.zeros(numpy.shape(w), dtype = numpy.longdouble)
        if self.l2reg > 0:
            wNorm = numpy.dot(w,w)
            L2Obj = wNorm
        
            L2Grad = 2 * w

        Obj = self.meanConstant * meanWeightedLoss + self.sqConstant * sumSqWeightedLoss + self.CConstant + self.l2reg * L2Obj
        Grad = self.meanConstant * numpy.reshape(g,numFeatures*numLabels)  + self.l2reg * L2Grad + self.sqConstant * numpy.reshape(gSq, numFeatures*numLabels)

        if self.verbose:
            print(".",)
            sys.stdout.flush()
       
        return (Obj, Grad.astype(numpy.float_))
    
    def fit(self, start_point):
        numLabels = numpy.shape(self.y)[1]
        numFeatures = numpy.shape(self.X)[1]

        ops = {'maxiter': self.n_iter, 'disp': self.verbose, 'gtol': self.tol,\
            'ftol': self.tol, 'maxcor': 50}
        self.coef_ = numpy.zeros((numFeatures, numLabels), dtype = numpy.longdouble)
        if start_point is not None:
            self.coef_ = start_point

        self.coef_ = numpy.reshape(self.coef_, numFeatures*numLabels).astype(numpy.float_)
        self.meanConstant, self.sqConstant, self.CConstant, self.prevObj = self.computeConstant(self.coef_)
        epochNum = 0
        while True:
            epochNum += 1
            Result = scipy.optimize.minimize(fun = self.Objective, x0 = self.coef_,
                method = 'L-BFGS-B', jac = True, tol = self.tol, options = ops)

            if self.verbose:
                print("Finished optimization", Result['message'])
                sys.stdout.flush()

            self.coef_ = Result['x']
            self.meanConstant, self.sqConstant, self.CConstant, currObj = self.computeConstant(self.coef_)
            if self.verbose:
                print('f(w_t-1)', self.prevObj, 'g(w_t, w_t-1)', Result['fun'], 'f(w_t)', currObj, "Norm", scipy.linalg.norm(self.coef_))
                sys.stdout.flush()
        
            if numpy.isclose(currObj, self.prevObj, atol = self.tol) or epochNum > self.n_iter:
                self.prevObj = currObj
                break
            self.prevObj = currObj

        self.coef_ = numpy.reshape(self.coef_, (numFeatures, numLabels))

    def checkGradient(self):
        numLabels = numpy.shape(self.y)[1]
        numFeatures = numpy.shape(self.X)[1]
        print("CHECKG", self.clip)
        sys.stdout.flush()
        predictor = PRM.VanillaISEstimator(n_iter = self.n_iter, tol = self.tol, l2reg = self.l2reg,
                        varpenalty = self.varpenalty, clip = self.clip, verbose = self.verbose)
        predictor.setAuxiliaryMatrices(self.X, self.y, self.logpropensity, self.sampledLoss)

        for i in xrange(10):
            startW = 10*numpy.random.randn(numFeatures*numLabels)
            doubleCheckObj = predictor.ObjectiveOnly(startW)

            self.meanConstant, self.sqConstant, self.CConstant, prevObj = self.computeConstant(startW)

            print("Majorize Objective", prevObj, "Vanilla Objective", doubleCheckObj)
            print(scipy.optimize.check_grad(self.ObjectiveOnly, self.GradientOnly, startW))
            sys.stdout.flush()
    """


class MajorizeStochasticEstimator(MajorizeISEstimator):
    def __init__(self, n_iter, min_iter, tol, l2reg, varpenalty, clip, verbose):
        MajorizeISEstimator.__init__(self, n_iter, tol, l2reg, varpenalty, clip, verbose)
        self.min_iter = min_iter

    def computeUWGrad(self, w, x, y, ySign, p, delta):
        numBatchSamples = numpy.shape(x)[0]
        wx = x.dot(w)
        ywx = numpy.multiply(wx, ySign)
        ProbPerInstanceLabel = scipy.special.expit(ywx)
        zeroMask = ProbPerInstanceLabel <= 0
        ProbPerInstanceLabel[zeroMask] = 1.0
        zeroEntries = zeroMask.sum(axis = 1, dtype = numpy.int)
        zeroMask = zeroEntries > 0

        logProbPerInstanceLabel = numpy.log(ProbPerInstanceLabel)
        logProbPerInstance = numpy.sum(logProbPerInstanceLabel, axis = 1, dtype = numpy.longdouble)

        logImportanceSampleWeights = logProbPerInstance - p
        mask = numpy.logical_and(logImportanceSampleWeights >= self.clip, numpy.logical_not(zeroMask))
        logImportanceSampleWeights[mask] = self.clip

        if self.verbose:
            print("MajorizeStochasticEstimator: [Debug] C=", mask.sum())
            sys.stdout.flush()

        ImportanceSampleWeights = numpy.exp(logImportanceSampleWeights)
        ImportanceSampleWeights[zeroMask] = 0.0

        WeightedLossPerInstance = numpy.multiply(delta, ImportanceSampleWeights)

        PartitionPerInstanceLabel = scipy.special.expit(wx)
        LabelPartitionPerInstanceLabel = y - PartitionPerInstanceLabel
        LabelPartitionPerInstanceLabel[mask, :] = 0

        LossLabelPartitionPerInstanceLabel = numpy.multiply(LabelPartitionPerInstanceLabel,
            WeightedLossPerInstance[:,None])

        xTranspose = x.transpose()
        g = xTranspose.dot(LossLabelPartitionPerInstanceLabel) / numBatchSamples

        WeightedGradient = numpy.multiply(LossLabelPartitionPerInstanceLabel,
                    WeightedLossPerInstance[:,None])
        VarGrad = xTranspose.dot(WeightedGradient) / numBatchSamples

        return g, VarGrad, WeightedLossPerInstance.mean(dtype = numpy.longdouble)

    def sgd(self, start_point):
        numLabels = numpy.shape(self.y)[1]
        numSamples, numFeatures = numpy.shape(self.X)
        batch_size = 100

        coef = numpy.zeros((numFeatures, numLabels), dtype = numpy.longdouble)
        if start_point is not None:
            coef = start_point

        adagrad_history = numpy.ones(numpy.shape(coef))
        #adaVargrad_history = numpy.ones(numpy.shape(coef))
        batchNum = 0

        detectedConvergence = False
        prevMean = None
        currMean = 0.0
        prevObj = None
        batchInterval = 50
        snapshot_coef = None
        patience = 0
        while True:
            permute = numpy.random.permutation(numSamples)
            self.X = self.X[permute, :]
            self.y = self.y[permute, :]
            self.ySign = self.ySign[permute, :]
            self.logpropensity = self.logpropensity[permute]
            self.sampledLoss = self.sampledLoss[permute]

            meanConstant, sqConstant, CConstant, currObj = self.computeConstant(coef)
            currInd = 0
            while currInd < numSamples:
                currEnd = currInd + batch_size
                if currEnd > numSamples:
                    currEnd = numSamples
                batchNum += 1
                if (batchNum >= self.n_iter):
                    if self.verbose:
                        print("MajorizeStochasticEstimator: [Message] Reached max iterations.")
                        sys.stdout.flush()
                    detectedConvergence = True
                    break

                g, VarGrad, u_w = self.computeUWGrad(coef, self.X[currInd:currEnd,:], self.y[currInd:currEnd,:],
                                self.ySign[currInd:currEnd,:], self.logpropensity[currInd:currEnd], self.sampledLoss[currInd:currEnd])

                adagrad_history = adagrad_history + numpy.multiply(g, g)
                rectified_adagrad = numpy.sqrt(adagrad_history)
                rectified_g = numpy.divide(g, numpy.sqrt(adagrad_history))

                #adaVargrad_history = adaVargrad_history + numpy.multiply(VarGrad, VarGrad)
                #rectified_adaVargrad = numpy.sqrt(adaVargrad_history)
                #rectified_VarGrad = numpy.divide(VarGrad, numpy.sqrt(adaVargrad_history))
                #rectified_VarGrad = numSamples * VarGrad
                
                #Grad = meanConstant * rectified_g  + self.l2reg * 2 * coef + sqConstant * 2 * rectified_VarGrad
                Grad = meanConstant * rectified_g  + self.l2reg * 2 * coef + sqConstant * 2 * u_w * rectified_g
                
                if (numpy.isclose(scipy.linalg.norm(Grad), self.tol, atol = self.tol) and
                        (batchNum >= self.min_iter)):
                    detectedConvergence = True
                    if self.verbose:
                        print("MajorizeStochasticEstimator: [Message] Gradient close to 0.")
                        sys.stdout.flush()
                    break

                if self.verbose:
                    print("MajorizeStochasticEstimator: [Debug] [coef_norm, grad_norm]: ", scipy.linalg.norm(coef), scipy.linalg.norm(Grad))
                    sys.stdout.flush()

                currMean += u_w
                if batchNum % batchInterval == 0:
                    currMean /= batchInterval
                    if prevMean is None:
                        prevMean = currMean
                        snapshot_coef = coef.copy()
                        patience = 0
                    elif (currMean >= (prevMean + self.tol)) and (batchNum >= self.min_iter):
                        if self.verbose:
                            print("MajorizeStochasticEstimator: [Debug] Progressive validation [prev, curr]: ", prevMean, currMean)
                            sys.stdout.flush()
                        patience += 1
                        if patience >= 5:
                            if self.verbose:
                                print("MajorizeStochasticEstimator: [Message] Ran out of patience during progressive validation. Reverting.")
                                sys.stdout.flush()
                            detectedConvergence = True
                            coef = snapshot_coef.copy()
                            break
                    else:
                        prevMean = currMean
                        patience = 0
                        snapshot_coef = coef.copy()
                    #meanConstant, sqConstant, CConstant, currObj = self.computeConstant(coef)
                    #batchInterval += 5

                currInd = currEnd
                coef = coef - Grad

            if detectedConvergence:
                break

        return coef, self.ObjectiveOnly(numpy.reshape(coef, numLabels * numFeatures))

    def fit(self, start_point):
        numSamples = numpy.shape(self.X)[0]

        best_coef, best_val = self.sgd(start_point)
        self.coef_ = best_coef
