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
    def __init__(self, input_dim, bucket_size, distance='hamming', C=0, seed=None):
        self.input_dim = input_dim
        self.seed = seed
        self.bucket_size = bucket_size 
        self.hash_table = defaultdict(list)

        self.distance = distance
        self.C = C

        if self.seed is not None:
            self.rng = np.random.Generator(seed=self.seed)
        else:
            self.rng = np.random.default_rng()

        self.projection_line = self._generate_projection()

    def _generate_projection(self):
        '''
        Generate a random line for projection.
        '''
        return self.rng.standard_normal(self.input_dim)
    
    def hash(self, point):
        '''
        Return the hash index of a point.
        '''
        projection = np.dot(self.projection_line, point)

        return projection // self.bucket_size
        #self.projections.append(projection)

        #return 1 if projection >= 0 else -1
    
    def index(self, point):
        '''
        Index a single point. Return the hash index.
        '''
        hash_index = self.hash(point)

        self.hash_table[hash_index].append(tuple(point))

        return hash_index
    