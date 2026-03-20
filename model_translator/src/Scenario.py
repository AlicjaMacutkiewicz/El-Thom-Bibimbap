import numpy as np
import rocketpy as rk

class Scenario():
    def __init__(self):
        super().__init__()
        self.environment = rk.Environment()
   

    def finish_scenario(self):
        self.destroy()