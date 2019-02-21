import numpy as np
import constants as c
import random
import os

from wav_reader import get_fft_spectrum

def build_buckets(max_sec, step_sec, frame_step):
    buckets = {}
    frames_per_sec = int(1/frame_step)
    end_frame = int(max_sec*frames_per_sec)
    step_frame = int(step_sec*frames_per_sec)
    for i in range(0, end_frame+1, step_frame):
        s = i
        s = np.floor((s-7+2)/2) + 1  # conv1
        s = np.floor((s-3)/2) + 1  # mpool1
        s = np.floor((s-5+2)/2) + 1  # conv2
        s = np.floor((s-3)/2) + 1  # mpool2
        s = np.floor((s-3+2)/1) + 1  # conv3
        s = np.floor((s-3+2)/1) + 1  # conv4
        s = np.floor((s-3+2)/1) + 1  # conv5
        s = np.floor((s-3)/2) + 1  # mpool5
        s = np.floor((s-1)/1) + 1  # fc6
        if s > 0:
            buckets[i] = int(s)
    return buckets

class Feature_reader:
    
    def __init__(self, max_seconds, directory):
        self.buckets = build_buckets(max_seconds, c.BUCKET_STEP, c.FRAME_STEP)
        
        self.directory = directory
        self.ids = [Id for Id in os.listdir(self.directory) if Id.startswith("id")]
    
    def get_random_feature(self):
        random_id = random.choice(self.ids)
        fft = self.get_ids_random_feature(random_id)
        return random_id, fft    
    
    def get_ids_random_feature(self, Id):
        id_directory = os.path.join(self.directory, Id)
        id_random_speech = os.path.join(id_directory, random.choice(os.listdir(id_directory)))
        id_random_utterance_path = os.path.join(id_random_speech, random.choice(os.listdir(id_random_speech)))
        fft = get_fft_spectrum(id_random_utterance_path, self.buckets)
        return fft
    
    def get_ids_features(self, Id, count = np.iinfo(np.int32).max):
        yielded = 0
        features = []
        
        for path in self.__file_iterator__(Id):
            if yielded == count:
                break
            feature = get_fft_spectrum(path, self.buckets)
            features.append(feature)
            yielded += 1
            
        return features
    
    def __file_iterator__(self, Id):
        id_directory = os.path.join(self.directory, Id)
        for root, dirs, files in os.walk(id_directory, topdown=False):
            for name in files:
                path = os.path.join(root, name)
                yield path
