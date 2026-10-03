import numpy as np

def descriptive_statistics(data: list | np.ndarray) -> dict:
    """
    Calculate various descriptive statistics metrics for a given dataset.
    
    Args:
        data: List or numpy array of numerical values
    
    Returns:
        Dictionary containing mean, median, mode, variance, standard deviation,
        percentiles (25th, 50th, 75th), and interquartile range (IQR)
    """
    
    descriptive_stas = {}
    x_sum = np.sum(data)
    mean=np.mean(data)
    values, counts = np.unique(data, return_counts=True)
    mode= values[counts.argmax()]
    variance = np.var(data)
    std_dev = variance**0.5
    q1, median, q3 = np.percentile(data, [25, 50, 75])
    iqr = q3-q1

    return {
        'mean': mean,
        'median': median,
        'mode': mode,
        'variance': variance,
        'standard_deviation': std_dev,
        '25th_percentile': q1,
        '50th_percentile': median,
        '75th_percentile': q3,
        'interquartile_range': iqr
    }
