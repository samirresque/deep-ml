import math

def normal_pdf(x, mean, std_dev):
	"""
	Calculate the probability density function (PDF) of the normal distribution.
	:param x: The value at which the PDF is evaluated.
	:param mean: The mean (μ) of the distribution.
	:param std_dev: The standard deviation (σ) of the distribution.
	"""
	pi = math.pi
	z = (x-mean)/std_dev
	val = math.exp(-math.pow(z,2)/2)/(std_dev*math.sqrt(2*pi))
	return round(val,5)