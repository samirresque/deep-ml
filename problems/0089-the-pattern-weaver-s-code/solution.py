import numpy as np

def softmax(values):
    m = np.max(values)
    exps = [np.exp(x-m) for x in values]
    total = np.sum(exps)
    return [e/total for e in exps]


def pattern_weaver(n, crystal_values, dimension):
    result = []
    for i in range(n):
        scores = [crystal_values[i]*crystal_values[j]/np.sqrt(dimension) for j in range(n)]
        weights = softmax(scores)
        weighted_sum = np.dot(weights, crystal_values)
        result.append(np.round(weighted_sum,4))

    return result


