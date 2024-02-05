#!/usr/bin/env python

import collections,logging,math,pprint,random
import numpy as np
import scipy

class TargetEncoder:
    
    def __init__(self, excluded_cols = [], pseudocount=0.5):
        """ Initialize """
        self.pseudocount = pseudocount
        
        self.n_neg_training_examples = -1
        self.n_pos_training_examples = -1
        self.colidxto01counter = None

    def _abbrevshow(alist, anum=5):
        if len(alist) <= anum*2: return [alist]
        else: return [alist[0:anum], alist[(len(alist)-anum):len(alist)]]
         
    def _assert_input(self, X, y):
        for label in y: assert label in [0, 1], F'Label {label} is not binary'
        X0 = self.X0 = X[y==0,:]
        X1 = self.X1 = X[y==1,:]
        assert X.shape[1] == X0.shape[1] and X0.shape[1] == X1.shape[1], 'InternalError'
        assert X0.shape[0] > 1, 'At least two negative examples should be provided'
        assert X1.shape[0] > 1, 'At least two positive examples should be provided'
        assert X1.shape[0] < X0.shape[0], 'The number of positive examples should be less than the number of negative examples'
    
    ''' Note: this method (which uses resampling) is supposed to outperform sklearn.preprocessing.TargetEncoder.fit_transform (which uses cross-fitting)'''
    def fit_transform(self, X1, y1, random_state=0):
        """
            X1: categorical features
            y1: binary response
            return: numerically encoded features using the WOE (weight of evidence) encoding (https://letsdatascience.com/target-encoding/) with resampling
        """
        self.fit(X1, y1)
        return self.transform(X1, random_state=random_state)

    def fit(self, X1, y1):
        X = np.array(X1)
        y = np.array(y1)
        self._assert_input(X, y)
        X0 = self.X0 = X[y==0,:]
        X1 = self.X1 = X[y==1,:]
        baseratio = (float(len(X1)) / float(len(X0)))
        logbase = np.log(baseratio)
        prior_w = self.pseudocount       
        self.n_neg_training_examples = X0.shape[0]
        self.n_pos_training_examples = X1.shape[0]
        self.colidxto01counter = []
        for colidx in range(X.shape[1]):
            elements = sorted(np.unique(X[:,colidx]))
            x0counter = collections.Counter(sorted(X0[:,colidx]))
            x1counter = collections.Counter(sorted(X1[:,colidx]))
            self.colidxto01counter.append((x0counter, x1counter))
        return self

    def transform(self, X1, random_state=0):
        X = np.array(X1)
        assert X.shape[1] == len(self.colidxto01counter)
        running_rand = np.random.default_rng(random_state)
        npc = self.pseudocount * self.n_neg_training_examples / self.n_pos_training_examples
        ppc = self.pseudocount
        n_training_examples = (self.n_neg_training_examples+self.n_pos_training_examples)
        ret = []
        for colidx in range(X.shape[1]):
            x0counter, x1counter = self.colidxto01counter[colidx]
            binom_rvs = [
                scipy.stats.binom.rvs(
                    n=n_training_examples,
                    p=float(x1counter[k]+ppc)/(x0counter[k]+x1counter[k]+ppc+npc),
                    random_state=running_rand
                ) for k in X[:,colidx]]
            targets = [(rv/float(n_training_examples)) for rv in binom_rvs]
            ret.append(targets)
        return np.array(ret).transpose()
    
def test_1():
    pp = pprint.PrettyPrinter(indent=4)
    logging.basicConfig(format='TargetEncoder %(asctime)s - %(message)s', level=logging.DEBUG)
    X = np.array([
        [ 1, 'A'],
        [ 1, 'B'],
        [ 1, 'B'],
        [ 1, 'B'],
        [ 1, 'B'],
        [ 2, 'C'],
        [ 2, 'C'],
        [ 2, 'C'],
        [ 2, 'C'],
        [ 2, 'D']
    ])
    y = np.array([1,1,0,1,1,0,0,0,0,0])
    encoder = TargetEncoder()
    X2 = encoder.fit_transform(X, y, random_state=17)
    logging.info(F'Encoded={X2}')
    X3 = encoder.fit_transform(X, y, random_state=997)
    logging.info(F'Encoded={X3}')

if __name__ == '__main__':
    test_1()

