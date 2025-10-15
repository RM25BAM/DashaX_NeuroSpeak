import numpy as np
from preprocess_chan14 import preprocess_data
path_arr = ["Data/pre-processed/windows_2.0s_0.5s_14ch.npz","Data/pre-processed/windows_2.0s_0.5s_8ch.npz","Data/pre-processed/windows_2.0s_0.5s_4ch.npz"] 

def main():
    user_input = int(input("Input Channel 1)14 2)8 3)4:"))
    if user_input == 1:
        npz_path = "Data/pre-processed/windows_2.0s_0.5s_14ch.npz"
        data = np.load(npz_path)
        X, y = data["X"], data["y"]
        print(f"Loaded dataset chan-14: X={X.shape}, y={y.shape}")
        X_train, X_val, X_test, Y_train, Y_val, Y_test = preprocess_data(X, y)
    elif user_input == 2:
        npz_path = "Data/pre-processed/windows_2.0s_0.5s_8ch.npz"
        data = np.load(npz_path)
        X, y = data["X"], data["y"]
        print(f"Loaded dataset chan-8: X={X.shape}, y={y.shape}")

    if user_input == 3:
        npz_path = "Data/pre-processed/windows_2.0s_0.5s_4ch.npz"
        data = np.load(npz_path)
        X, y = data["X"], data["y"]
        print(f"Loaded dataset chan-4: X={X.shape}, y={y.shape}")

    else:
        print("Input value from 1 - 3")
        main()
    #chan_14 = np.load(path_14)
main()