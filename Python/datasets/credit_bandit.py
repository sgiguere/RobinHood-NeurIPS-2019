"""Assumed to be run from this directory for now--may change in the future."""
import numpy as np
import os

from datasets.dataset import BanditDataset


credit_f = os.path.join('datasets', 'credit', 'german.data-numeric')


class CreditBanditFactory(object):
    def __init__(self):
        self.raw = []
        with open(credit_f, 'r') as f:
            for line in f:
                splits = line.split()
                self.raw.append(np.array([int(x) for x in splits]))
        self.raw = np.array(self.raw)
        self.feedback = self.raw[:, -1]    # last column (1 or 2).
        self.contexts = self.raw[:, :-1]  # all but last column.
        self.sex_marital_idx = 6          # sex/marital status.
        self.history_idx = 2              # credit history.
        self.history_all_paid = 0
        self.history_all_paid_at_bank = 1
        self.history_all_paid_till_now = 2
        self.history_delay = 3
        self.history_critical = 4

    def historical_policy(self, examples):
        """Return 1 if all credits until now have been paid."""
        history = examples[:, self.history_idx]
        paid = (history == self.history_all_paid).astype(int)
        paid_at_bank = (history == self.history_all_paid_at_bank).astype(int)
        paid_till_now = (history == self.history_all_paid_till_now).astype(int)
        positives = paid + paid_at_bank + paid_till_now
        # Return positive examples as 2 and negatives as 1 to match dataset.
        return (positives > 0).astype(int) + 1

    def generate(self, r_train=0.4, r_candidate=0.2, include_intercept=True,
                 include_T=False, seed=None, use_pct=1.0):
        """Generate an BanditDataset from the German Credit dataset.

        Args:
            r_train - fraction of examples to use during training.
            r_candidate - fraction of example to use as candidates.
            include_intercept - bool; append ones to each example if True.
            include_intercept - whether to include the protected attribute.
            seed - int, random seed.

        Returns:
            BanditDataset.
        """
        random = np.random.RandomState(seed)

        if include_T:
            S = self.contexts[:, :]
        else:
            all_but_protected = [i for i in range(len(self.contexts[0]))
                                 if i != self.sex_marital_idx]
            S = self.contexts[:, all_but_protected]

        if include_intercept:
            S = np.hstack((S, np.ones((len(S), 1))))

        # protected attributes
        T = self.raw[:, self.sex_marital_idx]
        T = (T>=4)*1 # Set type to 1 for females and zero for males
        A = self.historical_policy(S)
        A = A - A.min()
        n_actions = len(np.unique(A))

        # TODO(AK): should be a switch on policy as new policies implemented.
        # Rewards are either 1 or -1 and correspond to matching the feedback.
        R = (A == self.feedback).astype(int) * 2 - 1
        P = np.ones_like(A).astype(np.float)  # The historical policy is deterministic, so the reference probabilities are all 1


        # Use the specified percent of the data
        n_keep = int(np.ceil(len(S) * use_pct))
        I = np.arange(len(S))
        random.shuffle(I)
        I = I[:n_keep]
        S = S[I]
        A = A[I]
        R = R[I]
        T = T[I]    
        P = P[I]

        # Compute split sizes
        n_samples = len(S)
        n_train = int(r_train * n_samples)
        n_test = n_samples - n_train
        n_candidate = int(r_candidate * n_train)
        n_safety = n_train - n_candidate
        max_reward = max(R)
        min_reward = min(R)
        dataset = BanditDataset(S, A, R, n_actions, n_candidate, n_safety,
                            n_test, min_reward, max_reward, seed=seed, P=P, T=T)
        dataset.feature_maximums = S.max(axis=0)
        return dataset


def load(*args, **kwargs):
    return CreditBanditFactory().generate(*args, **kwargs)

if __name__ == '__main__':
    cbf = CreditBanditFactory()
    rld = cbf.generate()
    print(rld.max_reward)
