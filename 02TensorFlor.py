from mpi4py import MPI
import numpy as np
import socket
hostname = list(socket.gethostname().encode('utf8'))


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

numDataPerRank = len(hostname)
sendbuf = np.array(hostname, dtype=np.uint8)
print('Rank: ',rank, ', sendbuf: ',sendbuf)

recvbuf = None
if rank == 0:
    recvbuf = np.empty(numDataPerRank*size, dtype=np.uint8)

comm.Gather(sendbuf, recvbuf, root=0)

if rank == 0:
    print('Rank: ',rank, ', recvbuf received: ',recvbuf)
