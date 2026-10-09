import pickle
with open("checkpoints/climatology.pkl", "rb") as f:
    clim = pickle.load(f)
print(type(clim))
k = list(clim.keys())[0]
print("first key:", k)
print("shape:", clim[k].shape)
