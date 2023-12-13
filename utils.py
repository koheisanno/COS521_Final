from sklearn.metrics.cluster import adjusted_rand_score
from itertools import combinations

def adj_clusters(adata, obs_name, to_combine):
    adata.obs[f'{obs_name}_adj'] = adata.obs[obs_name]
    for c in to_combine:
        print(c)
        print(f'{obs_name}_adj')
        for d in c[1:]:
            adata.obs[f'{obs_name}_adj'][adata.obs[obs_name] == d] = c[0]

    return adata