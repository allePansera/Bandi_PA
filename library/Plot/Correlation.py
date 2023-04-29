import matplotlib.pyplot as plt
import pandas as pd


def plot(df: pd.DataFrame):
    try:
        plt.matshow(df.corr())
        plt.show()
    except Exception as e:
        raise Exception(f"Error '{e}' realizing Correlation Matrix plot")
