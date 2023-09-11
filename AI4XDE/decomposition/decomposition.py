import numpy as np
import deepxde as dde
from deepxde.geometry import *


def time_marching_decomposition(PDECase, num_split):
    geometry = PDECase.geomtime

    assert isinstance(geometry, GeometryXTime)
    timedomain = geometry.timedomain
    time_split_list = np.linspace(timedomain.t0, timedomain.t1, num_split)
    time_split_list = np.vstack((time_split_list[:-1], time_split_list[1:])).T

    split_time_PDECase_list = []
    for time_split in time_split_list:
        timedomain = TimeDomain(time_split[0], time_split[1])

        class TimeSplitPDECase(PDECase.__class__):
            def gen_geomtime(self):
                geom = geometry.geometry
                return dde.geometry.GeometryXTime(geom, timedomain)

        split_time_PDECase_list.append(TimeSplitPDECase())

    return split_time_PDECase_list
