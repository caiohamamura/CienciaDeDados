from mpi4py import MPI
import numpy as np
import socket
hostname = socket.gethostname()


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

numDataPerRank = 1
sendbuf = np.array([hostname], dtype=np.unicode)
print('Rank: ',rank, ', sendbuf: ',sendbuf)

recvbuf = None
if rank == 0:
    recvbuf = np.empty(numDataPerRank*size, dtype=np.unicode)

comm.Gather(sendbuf, recvbuf, root=0)

if rank == 0:
    print('Rank: ',rank, ', recvbuf received: ',recvbuf)
