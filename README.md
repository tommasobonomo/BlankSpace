# BlankSpace

This project aims to improve visualisation of satellite imagery through modern visualisation techniques. It also provides a way to extract GeoTiff images in Python code, and to download images to Google Drive using Google's Earth Engine API.

## Installation

A particular library is needed to interpret the GeoTiff files, [rasterio](https://rasterio.readthedocs.io/en/latest/). There are a few platform issues to install this library, as it depends on C libraries.

### Linux

The easiest way to install everything is to clone the `conda` environment, although it is pretty big:

```bash
conda env create -f environment.yml
conda activate blankspace
```

Otherwise, install [rasterio](https://rasterio.readthedocs.io/en/latest/)'s dependencies manually and then install all the needed libraries through `pip`:

```bash
pip install -r requirements.txt
```

### Windows

Can't rely on installing through conda, as it will not find some platform dependant libraries. Should checkout [rasterio](https://rasterio.readthedocs.io/en/latest/) and follow Windows instructions there, then install all needed libraries through `pip`:

```bash
pip install -r requirements.txt
```

## Demo

An example of applying [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis) to an image extracted from Sentinel2:

![Original image](https://raw.githubusercontent.com/tommasobonomo/blankspace/master/out/original/20180807-kortgene.png)

![PCA of many bands](https://raw.githubusercontent.com/tommasobonomo/blankspace/master/out/pca/20180807-kortgene.png)
