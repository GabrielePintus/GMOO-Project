import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

entity  = "pintus261-university-of-trieste"
project = "GMOOP"
path    = entity + "/" + project



ACTIVATIONS = ['ReLU', 'LeakyReLU', 'ELU', 'GELU', 'Tanh', 'Sigmoid']
class Filter:
    
    MLP_SINEWAVE = {
        "tags": {
            "$all": ["sinewave", "mlp"],
            "$nin": ["BO"]
        }
    }
    MLP_SINEWAVE_BO = {
        "tags": {
            "$all": ["sinewave", "mlp", "BO"],
        }
    }
    MLP_MNIST = {
        "tags": {
            "$all": ["mnist", "mlp", "large_batch"],
            "$nin": ["BO", "plateau"]
        }
    }


def get_metrics(df, run, generation):
    df[run.name] = run.config
    # Drop actiavtion key
    df[run.name].pop('activation')
    df[run.name]['generation'] = generation
    df[run.name]['val_mse'] = run.summary['val_mse']
    df[run.name]['val_mae'] = run.summary['val_mae']
    df[run.name]['val_R2'] = run.summary['val_R2']
    # df[run.name]['test_mse'] = run.summary['test_mse']
    # df[run.name]['test_mae'] = run.summary['test_mae']
    # df[run.name]['test_R2'] = run.summary['test_R2']
    df[run.name]['activation_function'] = ACTIVATIONS[df[run.name]['activation_function']]
    return df


def plot_metrics_evolution(df, cols):   
    mean = df.groupby('generation')[cols].mean()
    std = df.groupby('generation')[cols].std()
    best = []
    
    for generation in df['generation'].unique():
        sub_df = df[df['generation'] == generation]
        best.append(sub_df[sub_df['val_mse'] == sub_df['val_mse'].min()][cols])
        
    best = pd.concat(best)
    
    return mean, std, best