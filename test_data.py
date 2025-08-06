import pickle

# Replace 'file.pkl' with your actual file path
with open('/Odyssey/public/glorys/obs_masks/global_obs_6sats_masks_2022.pickle', 'rb') as f:
        data = pickle.load(f)

        print(data[0].shape)
        print(len(data))

