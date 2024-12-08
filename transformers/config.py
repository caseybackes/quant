from pathlib import Path


def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 20, 
        "lr": 1e-4,
        "lang_src": "en",
        "lang_tgt": "fr",
        "seq_len": 350,
        "d_model": 512,
        "model_folder": "weights",
        "model_filename": "tmodel_",
        "preload": None, 
        "tokenizer_file": "tokenizer_{0}.json", 
        "experiment_name": "runs/tmodel",
    }

def get_weights_file_path(config, epoch:str):
    model_folder= config['model_folder']
    model_basename = config['model_bnase_name'],
    model_filename = f"{model_basename}_{epoch}.pt"
    return str(Path('.')/ model_folder / model_filename)