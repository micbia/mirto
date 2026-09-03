# see https://gitlab.com/ska-telescope/ost/ska-ost-array-config/-/blob/master/docs/examples/examples_low_station.ipynb?ref_type=heads

import numpy as np
from ska_ost_array_config.array_config import LowSubArray

typ_layout = ['AA0.5', 'AA1', 'AA2', 'AA*', 'AA4']

for ll in typ_layout:
    layout = LowSubArray(subarray_type=ll)
    data = layout.array_config.xyz.to_numpy()
    if(ll == 'AA*'):
        ll = 'AAstar'
    np.savetxt('skalow_%s_layout.txt' %ll, data)

