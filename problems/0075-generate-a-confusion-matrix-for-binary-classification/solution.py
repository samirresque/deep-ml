
from collections import Counter

def confusion_matrix(data):
	counts = Counter(tuple(pair) for pair in data)
	tp = counts[(1,1)]
	fn = counts[(1,0)]
	fp = counts[(0,1)]
	tn = counts[(0,0)]
	tp_fn = [tp, fn]
	fp_tn = [fp, tn]
	confusion_matr = [tp_fn, fp_tn]
	return confusion_matr
