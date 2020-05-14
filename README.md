# Archeoview

This project aims to improve visualisation of satellite imagery through modern visualisation techniques. It also provides a way to extract GeoTiff images in Python code.

## Installation

A particular library is needed to interpret the GeoTiff files, [rasterio](https://rasterio.readthedocs.io/en/latest/). The easiest way to install everything is to clone the `conda` environment, although it is pretty big:

```bash
conda env create -f environment.yml
conda activate archeoview
```

Otherwise, install [rasterio](https://rasterio.readthedocs.io/en/latest/)'s dependencies manually and then install all the needed libraries through `pip`:

```bash
pip install -r requirements.txt
```

## Demo

An example of applying [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis) to an image extracted from Sentinel2:

![Original image](https://raw.githubusercontent.com/tommasobonomo/archeoview/master/out/original/20180807-kortgene.png)

![PCA of many bands](https://raw.githubusercontent.com/tommasobonomo/archeoview/master/out/pca/20180807-kortgene.png)