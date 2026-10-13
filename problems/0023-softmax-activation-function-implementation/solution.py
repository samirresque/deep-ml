import math

def softmax(scores: list[float]) -> list[float]:
    max_score = max(scores)
    stable_scores = [s-max_score for s in scores]
    exps = [math.exp(s) for s in stable_scores]
    sum_exps = sum(exps)
    output = [e/sum_exps for e in exps]

    return output