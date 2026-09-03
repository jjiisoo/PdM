import pandas as pd
import numpy as np
from download_data import RAW

df = pd.read_csv(RAW / "ai4i" / "ai4i2020.csv")
print("shape:", df.shape)
print("Null", df.isnull().sum())

REN = {
    "Air temperature [K]": "air_temp_k",
    "Process temperature [K]": "process_temp_k",
    "Rotational speed [rpm]": "rot_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "failure",
    "Product ID": "product_id",
    "Type": "type",
}

df = df.rename(columns=REN)


df.columns = df.columns.str.strip()

df = df.rename(
    columns={
        "TWF": "twf",
        "HDF": "hdf",
        "PWF": "pwf",
        "OSF": "osf",
        "RNF": "rnf",
    }
)

MODES = ["twf", "hdf", "pwf", "osf", "rnf"]

print(df["tool_wear_min"].describe())
anyf = df[MODES].sum(axis=1)
print(
    f"고장=1 인데 세부 모드가 하나도 없음 : {((df['failure'] == 1) & (anyf == 0)).sum()}건"
)
print(f"세부 모드가 있는데 고장=0 : {((df['failure'] == 0) & (anyf > 0)).sum()}건")


mismatch = ((df["failure"] == 1) & (anyf == 0)) | ((df["failure"] == 0) & (anyf > 0))

print(df.loc[mismatch, ["failure"] + MODES].to_string())


rate = df["failure"].mean()
print(f"고장률 : {rate * 100:.2f}% ({df['failure'].sum()}건 / {len(df)}건)")

print("세부 모드가 있는 행:", df[MODES].eq(1).any(axis=1).sum())
print("전체 고장 라벨이 1인 행:", df["failure"].eq(1).sum())

# from sklearn.dummy import DummyClassifier

# dummy = DummyClassifier(strategy="most_frequent").fit(Xtr, ytr)
# dp = dummy.predict(Xte)


from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier

features = [
    "type",
    "air_temp_k",
    "process_temp_k",
    "rot_speed_rpm",
    "torque_nm",
    "tool_wear_min",
]

X = pd.get_dummies(df[features], columns=["type"], dtype=int)
y = df["failure"]

Xtr, Xte, ytr, yte = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

dummy = DummyClassifier(strategy="most_frequent").fit(Xtr, ytr)
dp = dummy.predict(Xte)
