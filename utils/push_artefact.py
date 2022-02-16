import os
import wandb
import argparse

def push_artifact(name):
    path_artifact = os.path.join('input', f'{name}')

    wandb.init(
        entity="altegrad-gnn-link-prediction",
        project="altegrad"
    )

    artifact = wandb.Artifact(
        name=name,
        type="dataset",
        metadata={
            "emb_dim":768
        },
        description=f"Embeddings obtained for each abstract using {name}"
    )

    artifact.add_file(path_artifact)
    wandb.log_artifact(artifact, aliases=["latest"])
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Push an artifact to wandb")
    parser.add_argument("--file_name",required=True,type = str, help = "name of the file which should be located at /input/file_name")
    args = parser.parse_args()
    push_artifact(args.file_name)