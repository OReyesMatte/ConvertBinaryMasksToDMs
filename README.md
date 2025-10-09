# Convert binary masks to distance matrices
Euclidean distance matrices (EDMs) are convenient representations of point sets created by collecting all squared distances between points (see [10.1109/MSP.2015.2398954](https://doi.org/10.1109/MSP.2015.2398954) for more details). 
In this repository, we illustrate how distance matrices can be extracted from binary segmentation masks and saved for downstream analysis with self-supervised shape analysis methods such as [ShapeEmbed](https://github.com/uhlmanngroup/ShapeEmbed) and [ShapeEmbedLite](https://github.com/uhlmanngroup/ShapeEmbedLite).

## Getting started

_tested with python 3.12_

Create a virtual environment and install the dependencies as follows:
```sh
python3 -m venv .venv --prompt PrepareDataset
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install --requirement requirements.txt
```

`source .venv/bin/activate` enters the python virtual environment while a simple
`deactivate` from within the virtual environment exits it.

## BBBC010 example
The Jupyter Notebook [prepare_BBBC010.ipynb](prepare_BBBC010.ipynb) demonstrates how binary segmentation masks from the [BBBC010 dataset](https://bbbc.broadinstitute.org/BBBC010) can be transformed into distance matrices.
