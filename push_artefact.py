import os
import wandb

path_artifact = os.path.join('input', 'embeddings.npy')

wandb.init(
    entity="altegrad-gnn-link-prediction",
    project="altegrad"
)

artifact = wandb.Artifact(
    name="Allenai-SpecterEmbedding",
    type="dataset",
    metadata={
        "emb_dim":768
    },
    description="Embeddings obtained for each abstract using Allenai-SpecterEmbedding"
)

artifact.add_file(path_artifact)
wandb.log_artifact(artifact, aliases=["latest"])