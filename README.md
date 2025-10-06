# Prepare a dataset for use by ShapeEmbed
[ShapeEmbed](https://github.com/uhlmanngroup/ShapeEmbed)[^1] and its [updates](https://github.com/uhlmanngroup/ShapeEmbedLite)[^2] work on Euclidean distance matrices (EDM).
This is used as a shape descriptor and is not the form that raw takes out of a microscope. Here we demonstrate a method to produce distance matrices from segmentation binary masks.
The example provided uses the BBBC010[^3] dataset. You should leverage the presented method for your own data before using it as input to ShapeEmbed. 

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

## Worked BBBC010 example

You can find the Jupyter Notebook demostrating the conversion method over [here](prepare_BBBC010.ipynb).

[^1]: Foix-Romero, A., Russell, C., Krull, A. and Uhlmann, V., 2025. ShapeEmbed: a self-supervised learning framework for 2D contour quantification. NeurIPS 2025. [Pre-print DOI](https://doi.org/10.48550/arXiv.2507.01009).
[^2]: Foix-Romero, A., Krull, A. and Uhlmann, V., 2025. A comparison of data-driven shape quantification methods for 2D microscopy images. ICCV-BIC. [DOI]()
[^3]: We used the C.elegans infection live/dead image set version 1 provided by Fred Ausubel and available from the Broad Bioimage Benchmark Collection [Ljosa et al., Nature Methods, 2012].
