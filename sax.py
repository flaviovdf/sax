#-*- coding: utf8
from __future__ import division, print_function

from collections import Counter
from itertools import combinations
from numba import jit
from sklearn.preprocessing import normalize
from scipy import stats as ss

import numpy as np

def znorm(X):
    
    R = ((X.T - X.mean(axis=1)) / X.std(axis=1)).T
    R[np.isnan(R)] = 0
    
    return R

def paa(X, paa_segments=8):

    win_size = X.shape[1] // paa_segments

    #This line ignores the last window if it does not have w elements
    A = X[:, :win_size * paa_segments]

    #Reshapes the matrix then computes mean of each window of size w
    return np.reshape(A, (A.shape[0], win_size, paa_segments)).mean(axis=2)

def get_cut_points(alphabet_size=8):

    o = np.ones(alphabet_size)
    probs = (o / o.shape[0])[:-1]

    return ss.norm.ppf(probs.cumsum())

def sax(X, paa_segments=8, alphabet_size=8):

    #normalize and do the PAA representation
    Z = znorm(X)
    P = paa(Z, paa_segments)
    
    #Computes the sax representation
    reverse_cut_points = get_cut_points(alphabet_size)[::-1]
    prev_cut_point = np.inf
    S = np.zeros_like(P, dtype='i')
    
    for word in xrange(alphabet_size):
        if word < reverse_cut_points.shape[0]:
            cut_point = reverse_cut_points[word]
        else:
            cut_point = -np.inf

        idx = (P >= cut_point) & (P < prev_cut_point)
        S[idx] = word
        
        if word < reverse_cut_points.shape[0]:
            prev_cut_point = reverse_cut_points[word]

    #Computes the lookup distance matrix
    L = np.zeros((alphabet_size, alphabet_size))
    for word_i, word_j in combinations(xrange(alphabet_size), 2):
        dist = reverse_cut_points[word_i] - reverse_cut_points[word_j - 1]
        
        L[word_i, word_j] = dist
        L[word_j, word_i] = dist
    
    return S, L

def bag_of_words(X, window_size, paa_segments=8, alphabet_size=8):
    
    window_size = min(window_size, X.shape[1])
    assert paa_segments < window_size
    
    documents = []
    for _ in xrange(X.shape[0]):
        documents.append([])
    
    token_to_id = {}
    
    for i in xrange(X.shape[1] - window_size + 1):
        W = X[:, i:i + window_size]
        S, _ = sax(W, paa_segments, alphabet_size)
        
        for row in xrange(S.shape[0]):
            str_repr = ''.join(str(x) for x in S[row])
            
            if not documents[row] or str_repr != documents[row][-1]:
                documents[row].append(str_repr)
                if str_repr not in token_to_id:
                    token_to_id[str_repr] = len(token_to_id)
    
    R = np.zeros((X.shape[0], len(token_to_id)))
    for i, doc in enumerate(documents):
        counts = Counter(doc)
        for token in counts:
            R[i, token_to_id[token]] = counts[token]
    
    return normalize(R)

@jit('void(i4[:, :], f8[:, :], i4, i4, f8[:, :])', nopython=True)
def _jit_dist_all(S, L, n, paa_segments, D):
    nrm_factor = n / paa_segments
    for row_i in xrange(S.shape[0]):
        for row_j in xrange(row_i + 1, S.shape[0]):
            dist = 0
            for col_i in xrange(S.shape[1]):
                dist += L[S[row_i, col_i], S[row_j, col_i]] ** 2
        
            D[row_i, row_j] = dist * nrm_factor
            D[row_j, row_i] = dist * nrm_factor

@jit('void(i4[:, :], f8[:, :], i4, i4, f8[:, :], f8)', nopython=True)
def _jit_dist_all_roll(S, L, n, paa_segments, D, inf):
    nrm_factor = n / paa_segments
    for row_i in xrange(S.shape[0]):
        for row_j in xrange(row_i + 1, S.shape[0]):
            dist = inf
            
            for roll in xrange(S.shape[1]):
                curr_dist = 0
                
                for rolled_i in xrange(roll, S.shape[1] + roll):
                    col_i = rolled_i - roll
                    col_j = rolled_i % S.shape[1]
                    curr_dist += L[S[row_i, col_i], S[row_j, col_j]] ** 2

                if curr_dist < dist:
                    dist = curr_dist

            D[row_i, row_j] = dist * nrm_factor
            D[row_j, row_i] = dist * nrm_factor

def dist_all(S, L, n, paa_segments=8):
    D = np.zeros((S.shape[0], S.shape[0]))
    _jit_dist_all(S, L, n, paa_segments, D)
    return D

def dist_all_roll(S, L, n, paa_segments=8):
    D = np.zeros((S.shape[0], S.shape[0]))
    _jit_dist_all_roll(S, L, n, paa_segments, D, np.inf)
    return D

if __name__ == '__main__':
    import numpy as np
    X = np.genfromtxt('data/top.dat')[:, 1:]
    S, L = sax(X)
    print(S)
    print(L)
    
    B = bag_of_words(X, 32, 4, 4)
    for row in xrange(B.shape[0]):
        assert B[row].any()
    
    for col in xrange(B.shape[1]):
        assert B[:, col].any()

    print()
    S, L = sax(X, 4, 4)
    D_norm = dist_all(S, L, X.shape[1], 4)
    D_roll = dist_all_roll(S, L, X.shape[1], 4)
    print(S)
    print(L)
    print(D_norm)
    print(D_roll)
