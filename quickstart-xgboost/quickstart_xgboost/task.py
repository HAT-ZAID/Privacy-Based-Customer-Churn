"""quickstart_xgboost: A Flower / XGBoost app."""

# import os
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from pathlib import Path
from flwr_datasets.partitioner import IidPartitioner
from quickstart_xgboost.preprocess import preprocess_telco
import sys
from datasets import Dataset

FEATURE_NAMES: list[str] = [
    "Gender",
    "Age",
    "Under 30",
    "Senior Citizen",
    "Married",
    "Dependents",
    "Number of Dependents",
    "Zip Code",
    "Latitude",
    "Longitude",
    "Population",
    "Referred a Friend",
    "Number of Referrals",
    "Tenure in Months",
    "Phone Service",
    "Avg Monthly Long Distance Charges",
    "Multiple Lines",
    "Internet Service",
    "Avg Monthly GB Download",
    "Online Security",
    "Online Backup",
    "Device Protection Plan",
    "Premium Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Streaming Music",
    "Unlimited Data",
    "Paperless Billing",
    "Monthly Charge",
    "Total Charges",
    "Total Refunds",
    "Total Extra Data Charges",
    "Total Long Distance Charges",
    "Total Revenue",
    "Satisfaction Score",
    "CLTV",
    "Offer_No Offer",
    "Offer_Offer A",
    "Offer_Offer B",
    "Offer_Offer C",
    "Offer_Offer D",
    "Offer_Offer E",
    "Internet Type_Cable",
    "Internet Type_DSL",
    "Internet Type_Fiber Optic",
    "Internet Type_No Internet",
    "Contract_Month-to-Month",
    "Contract_One Year",
    "Contract_Two Year",
    "Payment Method_Bank Withdrawal",
    "Payment Method_Credit Card",
    "Payment Method_Mailed Check",
]


# LOCAL_HIGGS_PATH = os.environ.get(
#     "LOCAL_HIGGS_PATH",
#     r"D:\hf_cache\hub\datasets--jxie--higgs\snapshots"
#     r"\d53733cf80e5059f189f38c1b542d581377629a0"
#     r"\data\train-00000-of-00005-72c7f0393dc7600e.parquet",
# )

DATASET_PATH = r"D:\Privacy Based Customer Churn\data\telco_original.parquet"
encoder = joblib.load(r"D:\Privacy Based Customer Churn\models\telco_encoder.joblib")

LABEL_COL = "label"
INPUT_COLS = None  # None = all columns except label


def train_test_split(partition, test_fraction, seed):
    """Split the data into train and validation set given split rate."""
    train_test = partition.train_test_split(test_size=test_fraction, seed=seed)
    partition_train = train_test["train"]
    partition_test = train_test["test"]

    num_train = len(partition_train)
    num_test = len(partition_test)

    return partition_train, partition_test, num_train, num_test


def transform_dataset_to_dmatrix(data):
    """Transform dataset to DMatrix format for xgboost."""
    batch = data[:]
    x = np.asarray(batch["inputs"], dtype=np.float32)
    y = np.asarray(batch["label"], dtype=np.float32)
    return xgb.DMatrix(x, label=y)


fds = None  # Cache loaded dataset partitions

def load_data(partition_id, num_clients):
    "Load Telco dataset in parquet format"
    global fds
    if fds is None:
        print(f"Loading Telco dataset from local parquet: {DATASET_PATH}")
        df = pd.read_parquet(DATASET_PATH)
        preprocess_telco(df)
        one_hot_cols = ['Offer', 'Internet Type', 'Contract', 'Payment Method']
        encoded = encoder.transform(df[one_hot_cols])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(one_hot_cols), index = df.index)
        df.drop(columns= one_hot_cols, inplace=True)
        df = pd.concat([df, encoded_df], axis = 1)
        labels = np.array(df['Churn Label'].tolist(), dtype=np.float32)
        inputs = df.drop('Churn Label', axis=1)
        df = df.reindex(columns=FEATURE_NAMES, fill_value = 0)
        
        inputs = np.array(inputs, dtype=np.float32)
        dataset = Dataset.from_dict(
            {"inputs": list(inputs), "label": labels.astype(np.float32)}
        )
        dataset = dataset.shuffle(seed=42)
        partitioner = IidPartitioner(num_partitions=num_clients)
        partitioner.dataset = dataset
        fds = partitioner
    partition = fds.load_partition(partition_id)
    partition.set_format("numpy")
    
    train_data, valid_data, num_train, num_val = train_test_split(
        partition, test_fraction=0.2, seed=42
    )
    train_dmatrix = transform_dataset_to_dmatrix(train_data)
    valid_dmatrix = transform_dataset_to_dmatrix(valid_data)
    
    return train_dmatrix, valid_dmatrix, num_train, num_val


# def load_data(partition_id, num_clients):
#     """Load partition HIGGS data from local parquet shard."""
#     global fds
#     if fds is None:
#         print(f"[task] Loading HIGGS from local parquet: {LOCAL_HIGGS_PATH}")
#         df = pd.read_parquet(LOCAL_HIGGS_PATH)
#         labels = np.array(df[LABEL_COL].tolist(), dtype=np.float32)
#         inputs = np.array(df["inputs"].tolist(), dtype=np.float32)

#         from datasets import Dataset

#         dataset = Dataset.from_dict(
#             {"inputs": list(inputs), "label": labels.astype(np.float32)}
#         )

#         partitioner = IidPartitioner(num_partitions=num_clients)
#         partitioner.dataset = dataset
#         fds = partitioner

#     partition = fds.load_partition(partition_id)
#     partition.set_format("numpy")

#     train_data, valid_data, num_train, num_val = train_test_split(
#         partition, test_fraction=0.2, seed=42
#     )

#     train_dmatrix = transform_dataset_to_dmatrix(train_data)
#     valid_dmatrix = transform_dataset_to_dmatrix(valid_data)

#     return train_dmatrix, valid_dmatrix, num_train, num_val


def replace_keys(input_dict, match="-", target="_"):
    """Recursively replace match string with target string in dictionary keys."""
    new_dict = {}
    for key, value in input_dict.items():
        new_key = key.replace(match, target)
        if isinstance(value, dict):
            new_dict[new_key] = replace_keys(value, match, target)
        else:
            new_dict[new_key] = value
    return new_dict
