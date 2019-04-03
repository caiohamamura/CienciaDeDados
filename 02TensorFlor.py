from mpi4py import MPI
import numpy as np
import socket
import tensorflow as tf

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

comm.Allgather(sendbuf, recvbuf)
comm.Allgather(sendbuf2, recvbuf2)


# if rank == 0:
result = np.split(recvbuf, size)
result = [bytes(list(i)).decode('utf8') for i in result]
print('This rank ', rank,' Ranks: ',recvbuf2, ', recvbuf received: ', result)

hosts = result
ranks = recvbuf2

jobs = {'nodes': ['%s:222%s' % (i, j) for (i,j) in zip(hosts, ranks)]}
cluster = tf.train.ClusterSpec(jobs)

server = tf.train.Server(cluster, job_name="nodes", task_index=rank)
tf.reset_default_graph()
var = tf.Variable(initial_value=0.0, name='var')
sess = tf.Session(server.target)
sess.run(tf.global_variables_initializer())

if rank == 0:
    sess.run(var.assign_add(1.0))

with tf.train.MonitoredTrainingSession(
                master=server.target,
                is_chief=(rank == 0),
                checkpoint_dir="/tmp/train_logs",
                hooks=hooks) as mon_sess:

    print("Value of var in session %d:" % (rank), mon_sess.run(var))


with tf.device("/job:nodes/task:1"):
    print("This is rank %d talking" % (rank))


# var = tf.Variable(initial_value=0.0, name='var')
# sess = tf.Session(server.target)
# sess.run(tf.global_variables_initializer())
# print("Initial value of var in session %d:" % (rank), sess.run(var))

# if rank == 0:
#     sess.run(var.assign_add(1.0))

#     print("Value of var in session %d:" % (rank), sess.run(var))

# print("Value of var in session %d:" % (rank), sess.run(var))