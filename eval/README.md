# Evaluation

We evaluate the compared methods on tasks from HELMET and LongProc. The tasks we evaluate on are Recall, RAG, Rerank, ICL, LongQA, and Summarization from HELMET, and HTML to TSV and Travel Planning from LongProc.

## Setup

You should first install the python dependencies for HELMET by following the instructions in the [HELMET](https://github.com/princeton-nlp/HELMET) repository.
This will also install the add-ons for LongProc evaluation.
Then, you should install dependencies for running the various methods.
- **DuoAttention and PruLong:** These methods are evaluated by loading the saved masks and/or weights into the modeling classes exposed by the DuoAttention code; therefore, you should install the `duo_attn` library from the [DuoAttention](https://github.com/mit-han-lab/duo-attention) repository.
- **PyramidKV, SnapKV, and Locret:** These methods are evaluated by using the caching utilities from [MInference](https://github.com/microsoft/MInference). We had to modify the package a bit to add support for chunked patching, and to support Locret. The modified version can be found in the `Minference/` directory from the top level of this repository.

## Running the evaluation

Please follow the instructions in the [HELMET](https://github.com/princeton-nlp/HELMET) repository to run the evaluation. We provide extensive run scripts for different methods, prefilling chunk sizes, and options (e.g., patching/no patching) under `run_scripts/`. The various arguments are defined in `arguments.py` - it should be self-explanatory. If you have any questions, please feel free to contact us.

## Visualizing the results

The code for visualizing the KV footprint is provided in `viz/plot_kv_footprint.py`. You may have to modify the paths to the data to match your setup.