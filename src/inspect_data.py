import pandas as pd

from src.config import DATA_PATH


def main():
    data = pd.read_csv(DATA_PATH)

    print("\nFirst five messages:")
    print(data.head())

    print("\nDataset dimensions:")
    print(data.shape)

    print("\nClass distribution:")
    print(data["label"].value_counts())

    print("\nMissing values:")
    print(data.isnull().sum())


if __name__ == "__main__":
    main()
