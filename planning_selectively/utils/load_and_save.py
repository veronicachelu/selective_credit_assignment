import pickle
from imports import *
def save_obj(obj, name ):
    filename = name + '.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    # files.download(filename)

def load_obj(name):
    filename = name + '.pkl'
    with open(filename, 'rb') as f:
        return pickle.load(f)
