"""Fixed consensus-unit conversion; zero stays zero, no learned parameters."""
import numpy as np

def geometric_consensus(q, components=4):
    q=np.asarray(q)
    if type(components) is not int or components!=4:raise ValueError('registered four-component core required')
    if q.ndim!=2 or q.shape[0]!=q.shape[1] or not np.isfinite(q).all() or (q<0).any() or (q>1).any():
        raise ValueError('finite square affinity in [0,1] required')
    return np.power(q.astype(np.float64),1./components)
