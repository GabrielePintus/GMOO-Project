import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

entity  = "pintus261-university-of-trieste"
project = "GMOOP"
path    = entity + "/" + project



ACTIVATIONS = ['ReLU', 'LeakyReLU', 'ELU', 'GELU', 'Tanh', 'Sigmoid']
class Filter:
    MLP_SINEWAVE_GA = {
        "tags": {
            "$all": ["sinewave", "GA", "plateau", "MLP"],
        }
    }
    MLP_SINEWAVE_BO = {
        "tags": {
            "$all": ["sinewave", "BO", "plateau", "MLP"],
        }
    }
    MLP_MNIST_GA = {
        "tags": {
            "$all": ["mnist", "GA", "plateau", "MLP"],
        }
    }
    MLP_MNIST_BO = {
        "tags": {
            "$all": ["mnist", "BO", "plateau", "MLP"],
        }
    }




def runs_to_df_ga(runs):
    df = dict()
    for run in runs:
        df[run.name] = run.config
        df[run.name].pop('activation')
        generation = int(run.name.split('-')[1])
        df[run.name]['generation'] = generation
        df[run.name]['val_mse'] = run.summary['val_mse']
        df[run.name]['val_mae'] = run.summary['val_mae']
        df[run.name]['val_R2'] = run.summary['val_R2']
        df[run.name]['activation_function'] = ACTIVATIONS[df[run.name]['activation_function']]
    df = pd.DataFrame.from_dict(df, orient='index')
    return df

def runs_to_df_bo(runs):
    df = dict()
    for run in runs:
        df[run.name] = run.config
        df[run.name].pop('activation')
        generation = int(run.name.split('-')[-1])
        df[run.name]['generation'] = ((generation-1)//5)+1
        df[run.name]['val_mse'] = run.summary['val_mse']
        df[run.name]['val_mae'] = run.summary['val_mae']
        df[run.name]['val_R2'] = run.summary['val_R2']
        df[run.name]['activation_function'] = ACTIVATIONS[df[run.name]['activation_function']]
    df = pd.DataFrame.from_dict(df, orient='index')
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