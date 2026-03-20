# Import model path
import sys
sys.path.append('/oncoGAN/models')

# Import modules
import os
import time
import click
import traceback
import json
import pickle
import torch
import numpy as np
import pandas as pd

from caloforest.writer import get_writer
from caloforest.forest_diffusion import ForestModel

# CLI options
@click.command(name='trainCaloForest')
@click.option("--csv",
              type=click.Path(exists=True, file_okay=True),
              required=True,
              help="CSV file to train the model. The column with the predicted labels must be the last column")
@click.option("--config",
              type=click.Path(exists=True, file_okay=True),
              required=True,
              help="Config file to use for training the model")
@click.option('--prefix',
              type=click.STRING,
              required=True,
              help="Prefix to be used for saving files")
def TrainCaloForest(csv, config, prefix):

    """
    Command to train a Calo-Forest model on tabular data
    """

    # Load configuration file
    with open(config, "r") as f:
        cfg = json.load(f)

    # Set up writer functions
    writer = get_writer(os.path.basename(csv).rstrip('.csv'), cfg=cfg)

    # Seeds
    np.random.seed(seed=cfg["seed"])
    torch.manual_seed(cfg["seed"]) # torch is only used for calorimeter evaluation
    torch.cuda.manual_seed(cfg["seed"])

    try:
        # Open the training file
        df:pd.DataFrame = pd.read_csv(csv)

        # Map prediction labels
        pred_col = df.columns[-1]
        feature_cols = df.columns[df.columns != pred_col]
        pred_codes, pred_uniques = pd.factorize(df[pred_col])
        code_mapping:dict = dict(enumerate(pred_uniques))
        df[pred_col] = pred_codes

        # Split X and y
        X:np.ndarray = df.loc[:, feature_cols].to_numpy()
        y:np.ndarray = df[pred_col].to_numpy()

        # Package XGB hyperparameters
        hyper_names = ["max_depth", "n_estimators", "eta", "min_child_weight", "gamma", 
                    "lambda", "multi_strategy", "early_stopping_rounds", "device"]
        xgb_hypers = {k: v for k, v in cfg.items() if k in hyper_names}
        xgb_hypers["n_jobs"] = cfg["xgb_n_jobs"]

        # Train the model
        print("Starting forest_model")
        forest_model = ForestModel(
            n_t=cfg["n_t"],
            diffusion_type=cfg["diffusion_type"],
            xgb_hypers=xgb_hypers,
            duplicate_K=cfg["duplicate_K"],
            cat_indexes=[],
            bin_indexes=[],
            int_indexes=[],
            eps=cfg["eps"],
            beta_min=cfg["beta_min"],
            beta_max=cfg["beta_max"],
            solver=cfg["solver"],
            scaler=cfg["scaler"],
            n_jobs=cfg["n_jobs"],
            backend=cfg["backend"],
            n_batch=cfg["n_batch"],
            seed=cfg["seed"],
            logdir=writer.logdir,
            )
        
        prepro_X = forest_model.preprocess(
            X=X,
            label_y=y
        )
        # Save model wrapper as pickle
        writer.write_pickle('forest_model', {'model': forest_model, 'columns': df.columns.values.tolist(), 'mapping': code_mapping, 'cfg': cfg, 'label_y': y})

        t0 = time.time()
        forest_model.train(prepro_X)
        t1 = time.time()
        print(f"Done forest_model training in {t1-t0}s")
        
        # Inference
        forest_model.load_models()
        
        # Generate one set of samples using the labels from the train set
        t2 = time.time()
        Xy_fake = forest_model.generate(batch_size=X.shape[0], label_y=y)
        t3 = time.time()
        print(f"Generated data in {t3-t2}s")

        # Map back the labels to their original values
        Xy_fake:pd.DataFrame = pd.DataFrame.from_records(Xy_fake, columns=df.columns.values.tolist())
        Xy_fake[pred_col] = Xy_fake[pred_col].apply(lambda x: code_mapping[x])
        writer.write_pandas(f"{prefix}_calo_forest_simulations", Xy_fake)

    except Exception:
        traceback.print_exc()

@click.command(name='useCaloForest')
@click.option("--input",
              type=click.Path(exists=True, file_okay=True),
              required=False,
              help="A CSV file with two columns: n and label_y, used to simulate samples. If not provided, the same number of samples as in the training set will be generated")
@click.option("--load-dir", "load_dir",
              type=click.Path(exists=True, file_okay=False),
              required=True,
              help="Directory to load from")
@click.option('--prefix',
              type=click.STRING,
              required=True,
              help="Prefix to be used for saving the generated samples")
@click.option("--out_dir",
              type=click.Path(exists=True, file_okay=False),
              required=False,
              default=os.getcwd(),
              help="Directory to save the generated samples")
def UseCaloForest(input, load_dir, prefix, out_dir):
    
    """
    Command to quickly use a trained Calo-Forest model
    """

    # Load the forest model
    model = os.path.join(load_dir, 'forest_model.pkl')
    with open(model, 'rb') as file:
        model_dict = pickle.load(file)
    model_dict['model'].set_logdir(load_dir)
    model_dict['model'].set_solver_fn(model_dict['cfg']["solver"])
    reverse_mapping = {v: k for k, v in model_dict['mapping'].items()}

    # Prepare the number and type of samples to generate
    if input is not None:
        df = pd.read_csv(input)
        df = df.loc[df.index.repeat(df['n'])].reset_index(drop=True)
        df['label_y'] = df['label_y'].apply(lambda x: reverse_mapping[x])
        n = df.shape[0]
        y = df['label_y'].to_numpy()
    else:
        y = model_dict['label_y']
        n = len(y)

    # Generate one set of samples using the labels from the train set
    Xy_fake = model_dict['model'].generate(batch_size=n, label_y=y)

    # Map back the labels to their original values
    Xy_fake:pd.DataFrame = pd.DataFrame.from_records(Xy_fake, columns=model_dict['columns'])
    pred_col = Xy_fake.columns[-1]
    Xy_fake[pred_col] = Xy_fake[pred_col].apply(lambda x: model_dict['mapping'][x])
    Xy_fake.to_csv(os.path.join(out_dir,f"{prefix}_calo_forest_simulations.csv"), index=False)

