from mpi4py import MPI
import numpy as np
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


CHUNK = 20
sizeOfChunk = CHUNK//size
sizeOfLastChunk = CHUNK-sizeOfChunk*(size-1)



senddata = None

if rank != size-1:
    recvdata = np.zeros(sizeOfChunk, dtype=np.int)
   # print(len(recvdata))
else:
    recvdata = np.zeros(sizeOfLastChunk, dtype=np.int)


counts = ()
dspls = ()
countsMinusOne = (sizeOfChunk,) * (size - 1)
counts = countsMinusOne + (sizeOfLastChunk,)
for i in range(0, size):
    dspls = dspls + (i * sizeOfChunk,)

if rank == 0:
    senddata = np.arange(CHUNK,dtype=np.int)

comm.Scatterv([senddata,counts,dspls,MPI.INT],recvdata,root=0)
print ('on task',rank,'after Scatterv:    data = ',recvdata)


recvD2 = None
if rank ==0:
    recvD2=np.zeros(CHUNK,dtype=np.int)

comm.Gatherv(recvdata,[recvD2,counts,dspls, MPI.INT])
if rank == 0:
    print("On task ", rank, "after Gatherv:    data = ",recvD2)
