import ray
import time
from model_dqn import Network
from environment import Environment

@ray.remote(num_cpus=1)
class Actor:
    def __init__(self, worker_id, epsilon=0.5):
        self.id = worker_id
        self.model = Network()
        self.env = Environment()
        self.epsilon = epsilon

    def run(self):
        # while True:
        #     print('worker {} start'.format(self.id))
        #     time.sleep(2)
        return self.id
        
    
    