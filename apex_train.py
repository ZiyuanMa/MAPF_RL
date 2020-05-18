import os
os.environ["OMP_NUM_THREADS"] = "1"
import torch
from worker import GlobalBuffer, Learner, Actor
import time
import ray
import threading



if __name__ == '__main__':
    ray.init()

    buffer = GlobalBuffer.remote(1024)
    learner = Learner.remote(buffer)
    actors = [Actor.remote(i, 0.5, learner, buffer) for i in range(10)]

    actor_id = [ actor.run.remote() for actor in actors ]
    # time.sleep(10)
    # learner = Learner.remote(buffer)
    # for _ in range(100):
    #     ret = ray.get(learner.run.remote())
    #     if ret is None:
    #         time.sleep(2)
    while not ray.get(buffer.ready.remote()):
        time.sleep(5)

    learner.run.remote()
    
    while True:
        time.sleep(5)
        ray.get(buffer.stats.remote())