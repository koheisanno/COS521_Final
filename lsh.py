import numpy as np
from collections import defaultdict

class LSH:
    '''
    Random projection based locality sensitive hashing.

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
    
    def hash(self, point):
        '''
        Return the hash index of a point.
        '''
        projection = np.dot(self.projection_line, point)

        return 1 if projection >= 0 else 0
    
    def index(self, point):
        '''
        Index a single point. Return the hash index.
        '''
        hash_index = self.hash(point)

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

        for _ in range(l):
            # list of hashes
            f_list = [LSH(input_dim, bucket_size, seed) for j in range(k)]
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
        for point in points:
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
