import math
import pandas as pd
import numpy as np
import pickle
from sklearn import preprocessing
from sklearn.preprocessing import QuantileTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, f1_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def run():
    # -- LOAD FILE -- #
    filename = "Data/JabRef_train.csv"
    df = pd.read_csv(filename, dtype=object)

    df.dropna(inplace=True)

    # # - Set target - #
    y = df["set_clicked"]

    # # --- Label Encoder --- #
    encoder = preprocessing.LabelEncoder()
    df = df.apply(encoder.fit_transform)

    # # -- Get relevant columns -- #
    cor = df.corr()
    cor_target = abs(cor["set_clicked"])
    relevant_features = cor_target[cor_target > 0.01]
    print(relevant_features.index)

    # # -- Normal Distribution, Reduce impact of outliers -- #
    transformer = QuantileTransformer(output_distribution="normal")

    X = df[relevant_features.index]
    X = X.drop(["set_clicked"], 1)

    regressor = LinearRegression()
    regr = TransformedTargetRegressor(regressor=regressor, transformer=transformer)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=0
    )
    regr.fit(X_train, y_train)

    y_pred = regr.predict(X_test)
    df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
    print(df)

    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}")
    y = y_test.to_numpy(dtype=int)
    print(f"F1 Score: {f1_score(y, y_pred)}")

    df[["Actual", "Predicted"]].to_csv("david_results.csv")


if __name__ == "__main__":
    run()
