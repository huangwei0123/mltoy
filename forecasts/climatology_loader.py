import pickle


def load_climatology(path):
    with open(path, "rb") as f:
        return pickle.load(f)
