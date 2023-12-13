from lsh import LSH

from kdtree import KDTree
from scipy import sparse
from collections import defaultdict
import numpy as np

class KNN:
    '''
    Base class for KNN.
    '''
    def __init__(self, input_dim, similarity_metric="unweighted", seed=None):
        self.input_dim = input_dim
        self.adj_matrix = None
        self.sp_adj_matrix = None
        self.similarity_metric = similarity_metric

        # map point to index
        self.points_to_index = {}

    def _compute_jaccard(self, adjacency_matrix):
        num_points = adjacency_matrix.shape[0]
        jac_adj_mat = np.zeros((num_points, num_points))
        for i in range(num_points):
            for j in range(i+1, num_points):
                if adjacency_matrix[i, j]==1:
                    # Get the intersection of the sets of neighbors of i and j
                    intersection = np.sum(np.logical_and(adjacency_matrix[i], adjacency_matrix[j]))
                    union = np.sum(np.logical_or(adjacency_matrix[i], adjacency_matrix[j]))
                    jaccard_similarity = intersection / union
                    jac_adj_mat[i, j] = jaccard_similarity
                    jac_adj_mat[j, i] = jaccard_similarity

        return jac_adj_mat

class KNN_LSH(KNN):
    '''
    Approximate KNN using LSH.

    Attributes:
    - input_dim: the dimension of input data
    - seed: random seed (used to replicate results)
    - k: number of LSH functions in each g hash
    - l: number of g hashes
    - bucket_size: max bucket size B? in second layer hash which doesn't exist right now
    '''
    def __init__(self, input_dim, k, l, bucket_size, similarity_metric="unweighted", seed=None):
        super(KNN_LSH, self).__init__(input_dim, similarity_metric, seed)

        self.bucket_size = bucket_size
        self.k = k
        self.l = l
        self.g_list = []

        # map point to index
        self.points_to_index = {}

        for _ in range(l):
            # list of hashes
            f_list = [LSH(input_dim, bucket_size, seed) for _ in range(k)]
            self.g_list.append(f_list)

        self.hash_table = [defaultdict(list) for i in range(l)]

    def _g(self, index, point):
        '''
        Project a point onto k random lines. Return k-dim hash.
        '''
        g = self.g_list[index]
        g_hash = [g[i].hash(point) for i in range(self.k)]
        g_hash = tuple(g_hash)

        return g_hash
    
    def _get_candidates(self, point):
        candidates = set()

        for i in range(self.l):

            g_hash = self._g(i, point)
            for p in self.hash_table[i].get(g_hash):
                candidates.add(p)
                # only check max 4l points
                if len(candidates) >= 4 * self.l:
                    return np.array(list(candidates))
        
        return np.array(list(candidates))

    def insert_points(self, points):
        '''
        Preprocess a set of points into the hash table.
        '''
        for i, point in enumerate(points):
            if i%50 == 0:
                print("iteration ", i)
            self.points_to_index[tuple(point)] = i
            for i in range(self.l):
                g_hash = self._g(i, point)
                self.hash_table[i][g_hash].append(tuple(point))
    
    def query(self, point, num_neighbors):
        '''
        Collect set of points with same hash value as query point for any choice of g. Return K = num_neighbors closest points.
        '''
        candidates = self._get_candidates(point)

        args = np.argsort([np.linalg.norm(c - point) for c in candidates])[:num_neighbors]

        return candidates[args]

    def construct_knng(self, points, num_neighbors):
        '''
        Construct an adjacency matrix from the hash table.
        '''
        print("constructing data structure")
        self.insert_points(points)

        num_points = len(points)

        adjacency_matrix = np.zeros((num_points, num_points))

        print("querying points...")
        for i, point in enumerate(points):
            if i%2000 == 0:
                print("iteration ", i)
            neighbors = self.query(point, num_neighbors + 1)
            neighbor_indices = [self.points_to_index[tuple(n)] for n in neighbors]
            adjacency_matrix[i, neighbor_indices] = 1
            adjacency_matrix[neighbor_indices, i] = 1
        
        if self.similarity_metric == "jaccard":
            print("computing jaccard similarity...")
            adjacency_matrix = self._compute_jaccard(adjacency_matrix)

        self.adj_matrix = adjacency_matrix
        self.sp_adj_matrix = sparse.csr_matrix(adjacency_matrix)

class KNN_KDT(KNN):
    '''
    Exact KNN using KDTree.

    Attributes:
    - input_dim: the dimension of input data
    - similarity_metric: the similarity metric to use (default: "unweighted")
    - seed: random seed (used to replicate results)
    '''
    def __init__(self, input_dim, similarity_metric="unweighted", seed=None):
        super(KNN_KDT, self).__init__(input_dim, similarity_metric, seed)
        self.kdtree = None
    
    def insert_points(self, points):
        '''
        Preprocess a set of points into the hash table.
        '''
        for i, point in enumerate(points):
            self.points_to_index[tuple(point)] = i

    def construct_knng(self, points, num_neighbors):
        '''
        Construct the exact KNN graph using KDTree.

        Args:
        - points: input data points
        - num_neighbors: number of neighbors to consider
        '''
        self.insert_points(points)

        num_points = len(points)

        self.kdtree = KDTree(list(points), points.shape[1])

        adjacency_matrix = np.zeros((num_points, num_points))

        for i, point in enumerate(points):
            neighbors = self.kdtree.get_knn(point, num_neighbors+1)
            neighbor_indices = [self.points_to_index[tuple(n)] for n in neighbors]
            adjacency_matrix[i, neighbor_indices] = 1
            adjacency_matrix[neighbor_indices, i] = 1
        
        if self.similarity_metric == "jaccard":
            adjacency_matrix = self._compute_jaccard(adjacency_matrix)

        self.adj_matrix = adjacency_matrix
        self.sp_adj_matrix = sparse.csr_matrix(adjacency_matrix)

class KNN_Brute(KNN):
    '''
    Exact KNN with brute force.

    Attributes:
    - input_dim: the dimension of input data
    - similarity_metric: the similarity metric to use (default: "unweighted")
    - seed: random seed (used to replicate results)
    '''
    def __init__(self, input_dim, similarity_metric="unweighted", seed=None):
        super(KNN_Brute, self).__init__(input_dim, similarity_metric, seed)
    
    def insert_points(self, points):
        '''
        Preprocess a set of points into the hash table.
        '''
        for i, point in enumerate(points):
            self.points_to_index[tuple(point)] = i

    def construct_knng(self, points, num_neighbors):
        '''
        Construct an adjacency matrix of the kNN graph from the hash table.
        '''
        self.insert_points(points)

        num_points = len(points)

        adjacency_matrix = np.zeros((num_points, num_points))

        for i, point in enumerate(points):
            distances = []
            for point_2 in points:
                distances.append(np.linalg.norm(point - point_2))
            
            argsorted = np.argsort(np.array(distances))[:num_neighbors]
            neighbors = points[argsorted, :]
            neighbor_indices = [self.points_to_index[tuple(n)] for n in neighbors]
            adjacency_matrix[i, neighbor_indices] = 1
            adjacency_matrix[neighbor_indices, i] = 1

        if self.similarity_metric == "jaccard":
            adjacency_matrix = self._compute_jaccard(adjacency_matrix)

        self.adj_matrix = adjacency_matrix
        self.sp_adj_matrix = sparse.csr_matrix(adjacency_matrix)
