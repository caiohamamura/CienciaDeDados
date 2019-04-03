from mpi4py import MPI
import numpy as np
import socket
hostname = list(socket.gethostname().encode('utf8'))


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

numDataPerRank = len(hostname)
sendbuf = np.array(hostname, dtype=np.uint8)
sendbuf2 = np.array([rank], dtype=np.uint8)
print('Rank: ',rank, ', sendbuf: ',sendbuf)


recvbuf = np.empty(numDataPerRank*size, dtype=np.uint8)
recvbuf2 = np.empty(size, dtype=np.uint8)

comm.Allgather(sendbuf, recvbuf, root=0)
comm.Allgather(sendbuf2, recvbuf2, root=0)


# if rank == 0:
result = np.split(recvbuf, size)
result = [bytes(list(i)).decode('utf8') for i in result]
print('This rank ', rank,' Ranks: ',recvbuf2, ', recvbuf received: ', result)

# hosts = result
# ranks = recvbuf2