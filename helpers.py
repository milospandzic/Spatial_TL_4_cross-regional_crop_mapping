from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd

data_path = Path('data')

seed = 42
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)

def load_data(filename, train_size, test_size):

    data = pd.read_pickle(data_path / filename)

    X = data.iloc[:, 1:-1]
    Y = data.iloc[:, -1]

    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=1-train_size, random_state=seed)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=test_size/(1-train_size), random_state=seed)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test