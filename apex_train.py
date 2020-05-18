import os
os.environ["OMP_NUM_THREADS"] = "1"
import torch
from worker import GlobalBuffer, Learner, Actor
import time
import ray
import threading

import config

if __name__ == '__main__':
    ray.init()

    buffer = GlobalBuffer.remote(config.global_buffer_size)
    learner = Learner.remote(buffer)
    num_actors = 8
    actors = [Actor.remote(i, 0.4**(1+(i/(num_actors-1))*7), learner, buffer) for i in range(num_actors)]

    actor_id = [ actor.run.remote() for actor in actors ]
    # time.sleep(10)
    # learner = Learner.remote(buffer)
    # for _ in range(100):
    #     ret = ray.get(learner.run.remote())
    #     if ret is None:
    #         time.sleep(2)
    while not ray.get(buffer.ready.remote()):
        time.sleep(5)
        ray.get(buffer.stats.remote(5))

    print('start training')
    learner.train.remote()
    
    done = False
    while not done:
        time.sleep(5)
        done = ray.get(learner.stats.remote(5))
        ray.get(buffer.stats.remote(5))

    ray.terminate()