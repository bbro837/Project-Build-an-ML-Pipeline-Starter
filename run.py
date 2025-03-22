#!/usr/bin/env python
"""
This script tests a regression model
"""
import argparse
import logging
import os
import shutil
import matplotlib.pyplot as plt

import mlflow
import json

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

import wandb


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="test_model", settings=wandb.Settings(disable_gpu=True))
    run.config.update(args)

    # Use run.use_artifact(...).file() to get the test dataset artifact
    test_local_path = run.use_artifact(args.test_dataset).file()
   
    X_test = pd.read_csv(test_local_path)
    y_test = X_test.pop("price")  # this removes the column "price" from X_test and puts it into y_test

    logger.info(f"Minimum price: {y_test.min()}, Maximum price: {y_test.max()}")

    # Load the model
    model = mlflow.sklearn.load_model(args.mlflow_model)

    logger.info("Scoring")
    r_squared = model.score(X_test, y_test)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    logger.info(f"Score: {r_squared}")
    logger.info(f"MAE: {mae}")

    ######################################
    # Here we save variable r_squared under the "r2" key
    run.summary['r2'] = r_squared
    # Now save the variable mae under the key "mae".
    run.summary['mae'] = mae
    ######################################


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Test a regression model")

    parser.add_argument(
        "--mlflow_model",
        type=str,
        help="MLflow model to test",
        required=True
    )

    parser.add_argument(
        "--test_dataset",
        type=str,
        help="Artifact containing the test dataset",
        required=True
    )

    args = parser.parse_args()

    go(args)