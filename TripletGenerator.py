import random
import numpy as np
import constants as c

class TripletGenerator:
    def __init__(self, reader):
        self.reader = reader
    
    def ids(self):
        return self.reader.ids
        
    def get_reshaped_ids_random_feature(self, Id, batch_size):
        ffts = np.asarray([self.reader.get_ids_random_feature(Id).reshape(*c.input_shape) for _ in range(batch_size)])
        return ffts

    def create_triplets(self, batch_size):
        anchor_id = random.choice(self.reader.ids)
        foreign_id = random.choice(self.reader.ids)

        return [self.get_reshaped_ids_random_feature(anchor_id, batch_size),
                 self.get_reshaped_ids_random_feature(anchor_id, batch_size),
                 self.get_reshaped_ids_random_feature(foreign_id, batch_size)]