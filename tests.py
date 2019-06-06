from mpi4py import MPI
import numpy as np
from npy2mpi import fromnumpy, tonumpy
import unittest
import itertools
typecodes = "?cbhilqpBHILQfdgFDG"
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


npArray=np.arange(0,10,1,dtype=np.int16)

mt1 = fromnumpy(np.dtype(np.int16))
dt2 = np.dtype(tonumpy(mt1))
#assert(mt1==dt2)
print("mt1 = ",mt1)
print("dt2 = ",dt2)


#
# senddata = None
# if rank == 0:
#    senddata = np.arange(size*4,dtype=np.float64)
# recvdata = np.empty(4, dtype=np.float64)
# counts = None
# dspls = None
# if rank == 0:
#     senddata = np.arange(100, dtype=np.float64)
#     counts = (1, 2, 3)
#     dspls = (4, 3, 10)
# comm.Scatterv([senddata, counts, dspls, MPI.DOUBLE], recvdata, root=0)
# print('on task', rank, 'after Scatterv:    data = ', recvdata)

# senddata = None
# if self.rank == 0:
#     senddata = np.arange(self.size * 4, dtype=np.float64)
# recvdata = np.empty(4, dtype=np.float64)
# counts = None
# dspls = None
# if self.rank == 0:
#     senddata = np.arange(100, dtype=np.float64)
#     counts = (1, 2, 3)
#     dspls = (4, 3, 10)
# self.comm.Scatterv([senddata, counts, dspls, MPI.DOUBLE], recvdata, root=0)
# print('on task', self.rank, 'after Scatterv:    data = ', recvdata)
