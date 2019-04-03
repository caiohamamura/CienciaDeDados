import tensorflow as tf
import json
f = open("saida.out", "r")
jobs = json.loads(f.read())

cluster = tf.train.ClusterSpec(jobs)

x = tf.constant(2)


with tf.device("/job:nodes/task:1"):
    y2 = x - 66

with tf.device("/job:nodes/task:0"):
    y1 = x + 300
    y = y1 + y2

with tf.Session("grpc://s001-n103:2220") as sess:
    result = sess.run(y)
    print(result)
