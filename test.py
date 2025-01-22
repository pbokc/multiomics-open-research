import numpy as np # my import

import haiku as hk
import jax
import jax.numpy as jnp
import pandas as pd

from multiomics_open_research.bulk_rna_bert.pretrained import get_pretrained_model
from multiomics_open_research.bulk_rna_bert.preprocess import preprocess_rna_seq_for_bulkrnabert

# Get pretrained model
parameters, forward_fn, tokenizer, config = get_pretrained_model(
    model_name="bulk_rna_bert_tcga",
    embeddings_layers_to_save=(4,),
    checkpoint_directory="checkpoints/",
)
forward_fn = hk.transform(forward_fn)

# Get bulk RNASeq data and tokenize it
rna_seq_df = pd.read_csv("data/tcga_sample.csv")
rna_seq_array = preprocess_rna_seq_for_bulkrnabert(rna_seq_df, config)
tokens_ids = tokenizer.batch_tokenize(rna_seq_array)
tokens = jnp.asarray(tokens_ids, dtype=jnp.int32)
print("type of tokens:", type(tokens))
print("shape of tokens:", tokens.shape)

# Convert tokens to numpy array and print elements
tokens_np = np.array(tokens)
print("tokens elements:", tokens_np[0][0])

# Inference
random_key = jax.random.PRNGKey(0)
outs = forward_fn.apply(parameters, random_key, tokens)

# Get mean embeddings from layer 4
mean_embedding = outs["embeddings_4"].mean(axis=1)
print("type of mean_embedding:", type(mean_embedding))
print("mean_embedding:", mean_embedding)
