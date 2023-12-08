import numpy as np

class LSH:
    '''
    random projection based locality sensitive hashing.

    Attributes:
    - hash_size: the length of the hash output
    - input_dim: the dimension of input data
    - seed: random seed (used to replicate results)
    - projection_plane: random projection plane; hash_size x input_dim numpy array
    - rng: a random number generator
    '''
    def __init__(self, hash_size, input_dim, bucket_size, seed=None):
        self.hash_size = hash_size
        self.input_dim = input_dim
        self.seed = seed
        self.bucket_size = bucket_size
        self.hash_table = {}

        if self.seed is not None:
            self.rng = np.random.Generator(seed=self.seed)
        else:
            self.rng = np.random.default_rng()

        self.projection_plane = self._generate_plane()

    def _generate_plane(self):
        '''
        Generate a random plane for projection.
        '''
        return self.rng.standard_normal(size=(self.hash_size, self.input_dim))
    
    def _hash(self, point):
        projection = np.dot(self.projection_plane, point)

        return projection // self.bucket_size
    
    def index(self, point):
        '''
        Index a single point. Return the hash index.
        '''
        hash_index = self._hash(point)

        self.hash_table[hash_index] = tuple(point)

        return hash_index
