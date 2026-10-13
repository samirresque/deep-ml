import numpy as np
def k_means_clustering(points: list[tuple[float, ...]], k: int, initial_centroids: list[tuple[float, ...]], max_iterations: int) -> list[tuple[float, ...]]:

	for _ in range(max_iterations):
		points_around_nearest_centroid = [[] for _ in range(k)]
		for point in points:
			min_distance = float('inf')
			nearest_idx = None
			for i, centroid in enumerate(initial_centroids):
				distance = dist(point, centroid)
				if distance < min_distance:	
					min_distance = distance
					nearest_centroid = centroid
					nearest_idx = i
			
			# assign the point to the cluster of the nearest centroid
			points_around_nearest_centroid[nearest_idx].append(point)

		final_centroids = initial_centroids
		for i in range(k): # for each cluster
			# check for potential empty cluster 
			mean = np.mean(points_around_nearest_centroid[i], axis=0)
			final_centroids[i] = tuple(mean)

	return np.round(final_centroids,4)

#helper function to calculate the Euclidean distance

def dist(A: tuple,B: tuple):
	distance = np.sqrt((A[0]-B[0])**2 + (A[1]-B[1])**2)
	return distance