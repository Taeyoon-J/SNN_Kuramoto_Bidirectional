import json
import sys

import h5py

with h5py.File(sys.argv[1], "r") as data:
    print(json.dumps({key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                      for key, value in data.items()}, indent=2))

