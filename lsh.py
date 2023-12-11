import numpy as np
from collections import defaultdict

class LSH:
    '''
    random projection based locality sensitive hashing.

    Attributes:
    - input_dim: the dimension of input data
    - seed: random seed (used to replicate results)
    - projection_line: random projection line; input_dim x 1 numpy array
    - rng: a random number generator
    '''
    def __init__(self, input_dim, bucket_size,  seed=None):
        self.input_dim = input_dim
        self.seed = seed
        self.bucket_size = bucket_size 
        self.hash_table = defaultdict(list)

        if self.seed is not None:
            self.rng = np.random.Generator(seed=self.seed)
        else:
            self.rng = np.random.default_rng()

        self.projection_line = self._generate_line()

    def _generate_line(self):
        '''
        Generate a random line for projection.
        '''
        return self.rng.standard_normal(self.input_dim)
    
    def _hash(self, point):
        projection = np.dot(self.projection_line, point)

        return projection // self.bucket_size
    
    def index(self, point):
        '''
        Index a single point. Return the hash index.
        '''
        hash_index = self._hash(point)

        self.hash_table[hash_index].append(tuple(point))

        return hash_index

class KNN:
    '''
    Approximate KNN using LSH.

    Attributes:
    - input_dim: the dimension of input data
    - seed: random seed (used to replicate results)
    - k: number of LSH functions in each g hash
    - l: number of g hashes
    - bucket_size: max bucket size B? in second layer hash which doesn't exist right now
    '''
    def __init__(self, input_dim, k, l, bucket_size, seed=None):
        self.input_dim = input_dim
        self.bucket_size = bucket_size
        self.k = k
        self.l = l
        self.g_list = []

        for i in range(l):
            f_list = [LSH(input_dim, bucket_size, seed) for j in range(k)]
            self.g_list.append(f_list)

        self.hash_table = [defaultdict(list) for i in range(l)]

    def _g(self, index, point):
        '''
        Project a point onto k random lines. Return k-dim hash.
        '''
        g = self.g_list[index]
        g_hash = [g[i].index(point) for i in range(self.k)]
        g_hash = tuple(g_hash)

        self.hash_table[index][g_hash].append(tuple(point))

        return g_hash

    def insert_points(self, points):
        '''
        Preprocess a set of points into the hash table.
        '''
        for point in points:
            for i in range(self.l):
                self._g(i, point)
    
    def query(self, point, num_neighbors):
        '''
        Collect set of points with same hash value as query point for any choice of g. Return K = num_neighbors closest points.
        '''
        candidates = []

        for i in range(self.l):
            g_hash = self._g(i, point)
            candidates.append(self.hash_table[i].get(g_hash))
        
        args = np.argsort([np.linalg.norm(c - point) for c in candidates])[:num_neighbors]

        return candidates[args]
