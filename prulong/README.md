# PruLong

This README file outlines the setup and usage of the PruLong codebase.

## Setup

Our code is based on [ProLong](https://github.com/princeton-nlp/ProLong) and uses ProLong data (which should go under `datasets/`).
Please refer to the ProLong repository for instructions on downloading the data. We use newer versions of some libraries, so please use our `requirements.txt` file instead of the one in the ProLong repository (you will still have to install Flash Attention from the instructions in the ProLong repository).

## Running the code

The main change in our code is the use of learned masks that implement local/global attention heads in `training/modeling_flash_llama.py`. Consequently, PruLong may be run in three modes:
- Train masks only - this is the default mode and is a good option when using an already instruction-tuned model.
- Train masks and weights - training weights might allow for additional improvements, but might undo already-learned post-training (as the training is done on pre-training data)
- Train weights only - this is a good option for performing SFT on an already prulonged model.

We provide example scripts for all three modes inside `run_scripts/`, with the hyperparameters set to the values we used for our experiments. At the end of training, you may save the masks to the checkpoint directory by running 

```python
python save_prulong_masks.py --checkpoint /path/to/checkpoint [--sparsity <float>]
```

If a sparsity value is provided, the masks are discretized to 0/1 to match the provided sparsity. The saved masks can then be used for evaluation in `eval/` from the top level of this repository.