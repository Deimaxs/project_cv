#Script Adrian Rosebrock: https://pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/

class TrackableObject:
	def __init__(self, objectID, centroid):
		self.objectID = objectID
		self.centroids = [centroid]

		self.counted = False
		self.attributes = []