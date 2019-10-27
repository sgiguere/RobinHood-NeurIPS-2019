from math import sqrt
import numpy as np
from numpy import log, transpose
from numpy.linalg import inv, pinv
from scipy.stats import norm


def eta(T):
    """
    Generates the cutoff probabilities for exploration rounds in interval
    chaining.

    :param T: the total number of iterations
    """
    return np.array([pow(t, -1/3) for t in range(1, T+1)])


def beta(k, d, c):
    """
    Generates the scaled down feature weights for a true model from the
    distribution β ∼ U[0, c]^d.

    :param k: the number of arms
    :param d: the number of features
    :param c: the scale of the feature weights
    """
    return np.random.uniform(0, c+1, size=(k, d))


def print_progress(s, should_print):
    """
    Helper function to print the progress of an algorithm as it's running.

    :param s: the string to print
    :should_print: whether or not the string should be printed
    """
    if should_print:
        print(s)


def top_interval(X, Y, k, d, _delta, T, n_train, _print_progress=True):
    """
    Simulates T rounds of TopInterval for k.

    :param X: a 3-axis (T, k, d) ndarray of d-dimensional context vectors for
              each time-step and arm
    :param Y: a T x k ndarray of reward function output for each context vector
    :param k: the number of arms
    :param d: the number of features
    :param _delta: confidence parameter
    :param T: the number of iterations
    :param _print_progress: True if progress should be printed; False otherwise
    :returns: cum_regret (the total regret across all T runs of the algorithm),
              avg_regret (the regret averaged across all T runs of the algorithm),
              final_regret (the regret in the last round of the algorithm)
    """
    pp = _print_progress
    _eta = eta(T)  # exploration cutoff probabilities
    picks = []
    probs = []
    for t in range(T):
        print_progress('Iteration [{0} / {1}]'.format(t, T), pp)
        if np.random.rand() <= _eta[t]:
            # Play uniformly at random from [1, k].
            picks.append(np.random.randint(0, k))
            print_progress('Exploration round.', pp)
            prob = np.ones(k) / k
        else:
            intervals = []
            for i in range(k):
                # Compute beta hat.
                I = (np.array(picks) == i)
                if sum(I) < 3:
                    intervals.append([-1, 1])
                else:
                    Xti = X[:t, i][I]
                    Yti = Y[:t, i][I]
                    xi  = X[t, i]
                    XTX = pinv(Xti.T.dot(Xti))
                    Bh_t_i = XTX.dot(Xti.T).dot(Yti)  # Compute OLS estimators.
                    yh_t_i = Bh_t_i.dot(xi)
                    # Compute the confidence interval width using the inverse CDF.
                    if np.unique(Yti).shape[0] == 1 and np.unique(Yti)[0] == 1:
                        intervals.append([1 - 2*3/len(Yti), 1])
                    elif np.unique(Yti).shape[0] == 1 and np.unique(Yti)[0] == -1:
                        intervals.append([-1, -1 + 2*3/len(Yti)])
                    else:
                        w_t_i = norm.ppf(1 - _delta/(2*T*k), loc=0,
                                         scale=np.sqrt(np.var(Yti) * xi.dot(XTX).dot(transpose(xi))))
                        intervals.append([max(yh_t_i - w_t_i, -1), min(yh_t_i + w_t_i, 1)])
            # Pick the agent with the largest upper bound.
            picks.append(np.argmax(np.array(intervals)[:, 1]) if intervals else np.random.randint(0, k))
            prob = np.zeros(k)
            prob[np.argmax(np.array(intervals)[:, 1])] = 1.0
        if t >= n_train:
            probs.append(prob)
    return np.array(probs)


def compute_chain(i_st, intervals, k, _print_progress=True):
    # Sort intervals by decreasing order.
    pp = _print_progress
    chain = [i_st]
    print_progress(intervals[:, 1], pp)
    ordering = np.argsort(intervals[:, 1])[::-1]
    intervals = intervals[ordering, :]

    lowest_in_chain = intervals[0][0]
    for i in range(1, k):
        if intervals[i][1] >= lowest_in_chain:
            chain.append(i)
            lowest_in_chain = min(lowest_in_chain, intervals[i][0])
        else:
            return chain
    return np.unique(chain)


def interval_chaining(X, Y, k, d, _delta, T, n_train, _print_progress=True):
    """
    Simulates T rounds of TopInterval for k.

    :param X: a 3-axis (T, k, d) ndarray of d-dimensional context vectors for
              each time-step and arm
    :param Y: a T x k ndarray of reward function output for each context vector
    :param k: the number of arms
    :param d: the number of features
    :param _delta: confidence parameter
    :param T: the number of iterations
    :param _print_progress: True if progress should be printed; False otherwise
    :returns: cum_regret (the total regret across all T runs of the algorithm),
              avg_regret (the regret averaged across all T runs of the algorithm),
              final_regret (the regret in the last round of the algorithm)
    """
    pp = _print_progress
    _eta = eta(T)  # exploration cutoff probabilities
    picks = []
    probs = []
    for t in range(T):
        print_progress('Iteration [{0} / {1}]'.format(t, T), pp)
        if np.random.rand() <= _eta[t]:
            # Play uniformly at random from [1, k].
            picks.append(np.random.randint(0, k))
            print_progress('Exploration round.', pp)
            prob = np.ones(k) / k
        else:
            intervals = []
            for i in range(k):
                I = (np.array(picks) == i)
                if sum(I) < 3:
                    intervals.append([-1, 1])
                else:
                    Xti = X[:t, i][I]
                    Yti = Y[:t, i][I]
                    xi  = X[t, i]
                    XTX = pinv(Xti.T.dot(Xti))
                    Bh_t_i = XTX.dot(Xti.T).dot(Yti)  # Compute OLS estimators.
                    yh_t_i = Bh_t_i.dot(xi)
                    # Compute the confidence interval width using the inverse CDF.
                    if np.unique(Yti).shape[0] == 1 and np.unique(Yti)[0] == 1:
                        intervals.append([1 - 2*3/len(Yti), 1])
                    elif np.unique(Yti).shape[0] == 1 and np.unique(Yti)[0] == -1:
                        intervals.append([-1, -1 + 2*3/len(Yti)])
                    else:
                        w_t_i = norm.ppf(1 - _delta/(2*T*k), loc=0,
                                         scale=np.sqrt(np.var(Yti) * xi.dot(XTX).dot(transpose(xi))))
                        intervals.append([max(yh_t_i - w_t_i, -1), min(yh_t_i + w_t_i, 1)])
            # Chaining
            i_st  = np.argmax(np.array(intervals)[:, 1])
            chain = compute_chain(i_st, np.array(intervals), k, pp)
            picks.append(np.random.choice(chain))
            prob = np.array([ 1/len(chain) if (i in chain) else 0.0 for i in range(k) ])
        if t >= n_train:
            probs.append(prob)
    return np.array(probs)


def ridge_fair(X, Y, k, d, _delta, T, _lambda, n_train, _print_progress=True):
    """
    Simulates T rounds of ridge_fair.

    :param X: a 3-axis (T, k, d) ndarray of d-dimensional context vectors for
              each time-step and arm
    :param Y: a T x k ndarray of reward function output for each context vector
    :param k: the number of arms
    :param d: the number of features
    :param _delta: confidence parameter
    :param T: the number of iterations
    :param _lambda: regularization paramameter
    """
    pp = _print_progress
    picks = []
    probs = []
    rets  = []
    for t in range(T):
        intervals = []
        for i in range(k):
            R = 1
            I = (np.array(picks) == i)
            if sum(I) < 3:
                intervals.append([-1, 1])
            else:
                Xti = X[:t, i][I]
                Yti = Y[:t, i][I]
                xi  = X[t, i]
                if np.unique(Yti).shape[0] == 1 and np.unique(Yti)[0] == 1:
                    intervals.append([1 - 2*3/len(Yti), 1])
                elif np.unique(Yti).shape[0] == 1 and np.unique(Yti)[0] == -1:
                    intervals.append([-1, -1 + 2*3/len(Yti)])
                else:
                    C = pinv(Xti.T.dot(Xti) + (_lambda * np.identity(d)))
                    B_it = C.dot(Xti.T).dot(Yti)
                    y_ti = transpose(xi).dot(B_it)
                    _wti1 = sqrt(transpose(xi).dot(C).dot(xi))
                    _wti2 = R * sqrt(d * log((1 + (t / _lambda)) / _delta)) + sqrt(_lambda)
                    w_ti = _wti1 * _wti2
                    intervals.append([max(y_ti-w_ti,-1), min(y_ti+w_ti,1)])
        # play uniformly random from chain
        i_st = np.argmax(np.array(intervals)[:, 1])
        chain = compute_chain(i_st, np.array(intervals), k, pp)
        picks.append(np.random.choice(chain))
        prob = np.array([ 1/len(chain) if (i in chain) else 0.0 for i in range(k) ])
        if t >= n_train:
            probs.append(prob)
    return np.array(probs)