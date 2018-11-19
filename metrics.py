import numpy as np
from scipy.spatial import distance
from itertools import combinations

def _unsimilarity_(classes, distance_function, aggregation_functional, internal):
    if internal:
        results = []
        for A in classes:
            tr = np.triu(distance.cdist(A, A, distance_function), k = 1).ravel()
            tr = tr[tr != 0]
            results.append(aggregation_functional(tr) if tr.any() else 0)
    else:
        pairs = combinations(classes, 2)
        results = [aggregation_functional(distance.cdist(pair[0], pair[1], distance_function)) for pair in pairs]
    
    return np.mean(results)
    
    
def mean_cos_unsimilarity(classes, internal = False):
    return _unsimilarity_(classes, distance_function = 'cosine', aggregation_functional = np.mean, internal = internal)

def test_metrics():
    A = [[1, 1]]
    B = [[1, 2], [3, 4]]
    C = [[5, 6], [-1, 1]]
    D = [[1, 1], [2, 1], [3, 4]]
    
    assert np.allclose(mean_cos_unsimilarity([A], internal = True), 0.0)
    assert np.allclose(mean_cos_unsimilarity([D], internal = True), 0.05564667242946792)
    assert np.allclose(mean_cos_unsimilarity([A, B]), 0.0306836041441)
    assert np.allclose(mean_cos_unsimilarity([A, B, C]), 0.3084328374647)