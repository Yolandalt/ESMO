import numpy as np

class User_class:
    def __init__(self,
                 n_edge = 6
                  ):
        self.n_edge = n_edge
        # The number of frames per user
        self.n_frames = np.random.randint(1, 10)

        self.frame_id = np.random.randint(0, 100, self.n_frames)




