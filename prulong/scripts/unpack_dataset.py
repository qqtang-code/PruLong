from functools import partial

from tqdm import tqdm

from typing import Optional

import numpy as np
from pathlib import Path

from simple_parsing import ArgumentParser, field
from dataclasses import dataclass

from datatools.load import load, LoadOptions
from datatools.process import process, ProcessOptions
from streaming.base.array import Array



def unpack_fn(data: Array,
                process_id: int):

    for i in tqdm(range(len(data)), desc=f"Process {process_id}"):
        item = data[i]
        input_ids = item["input_ids"]
        masks = item["mask"] if "mask" in item else np.ones_like(input_ids, dtype=np.uint8)
        indices = item["indices"]
        
        for a, b in indices:
            output_item = {
                "input_ids": input_ids[a:b],
                "mask": masks[a:b],
                "length": b - a
            }
            yield output_item

def main():
    parser = ArgumentParser()

    parser.add_argument("inputs", type=Path, nargs="+", help="Input dataset paths")
    parser.add_argument("output", type=Path, help="Output dataset path")

    parser.add_arguments(LoadOptions, dest="load_options")
    parser.add_arguments(ProcessOptions, dest="process_options")

    args = parser.parse_args()

    print("Arguments:", args)
    dataset = load(*args.inputs, options=args.load_options)
    N = len(dataset)
    print(f"Loaded dataset with {N} samples")

    process(dataset,
            partial(unpack_fn, options=args.tokenize_options),
            args.output,
            args.process_options)


if __name__ == "__main__":
    main()
