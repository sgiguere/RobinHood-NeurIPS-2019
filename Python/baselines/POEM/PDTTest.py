import sys
import numpy

class ExperimentResult:
    def __init__(self, resultList, verbose):
        self.verbose = verbose
        self.numSamples = len(resultList)
        if self.numSamples <= 0 and self.verbose:
            print("No observations provided. Aborting.")
            sys.stdout.flush()
            return

        self.results = numpy.array(resultList, dtype = numpy.longdouble)

    def reportMean(self):
        avg = self.results.mean(dtype = numpy.longdouble)
        return avg

    def testDifference(self, otherResult):
        numSamples = self.numSamples
        if self.numSamples > otherResult.numSamples:
            numSamples = otherResult.numSamples
            if self.verbose:
                print("Restricting numSamples to", numSamples)
                sys.stdout.flush()

        sqrtN = numpy.sqrt(numSamples)

        Obs1 = self.results[0:numSamples]
        Obs2 = otherResult.results[0:numSamples]

        Diff = Obs2 - Obs1
        MeanDiff = Diff.mean(dtype = numpy.longdouble)
        StdDiff = numpy.std(Diff, dtype = numpy.longdouble, ddof = 1)
        T = MeanDiff * sqrtN / StdDiff

        if self.verbose:
            print("T value", T)
            sys.stdout.flush()

        if T > 1.833:
            return True
        else:
            return False

