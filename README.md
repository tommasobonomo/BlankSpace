# BlankSpace

This project aims to improve visualisation of satellite imagery through modern visualisation techniques. It also provides a way to extract GeoTiff images in Python code, and to download images to Google Drive using Google's Earth Engine API. Finally, it is a showcase of Team BlankSpace's work during the Innovation Space Project course at Eindhoven Technical University, academic-year 2019/2020.

## Demos

### Train Track monitoring

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tommasobonomo/BlankSpace/blob/master/insar/Interactive_InSAR.ipynb)

The link above is a convinient way to access the prototype and play around with it.
The prototype is complete in every way, except for the automatic masking of train tracks in the elevation picture computation.
This is just a minor aspect, the main implementation is there. 

The notebook file is also available in the directory `insar/`, at the top level of this repository.

Here are some demo images from the prototype:

![InSAR complete](https://raw.githubusercontent.com/tommasobonomo/BlankSpace/master/demo_images/InSAR_complete.png)

![InSAR masked](https://raw.githubusercontent.com/tommasobonomo/BlankSpace/master/demo_images/InSAR_masked.png)

### Coastal Monitoring

The prototype can be run following the instructions below for installation and then executing:

```bash
pip install -e .
python -m blankspace.visualize_grid
```

The prototype is complete, although slow. Possibly will face an architectural refactoring. To configure, one must dive into the source code to select the correct data source folder, the desired number of cells and the desired resolution.

Here are some screenshots from the prototype:

![Coastal monitoring complete](https://raw.githubusercontent.com/tommasobonomo/BlankSpace/master/demo_images/Coastal_complete.png)

![Coastal monitoring one trend](https://raw.githubusercontent.com/tommasobonomo/BlankSpace/master/demo_images/Coastal_one_trend.png)

![Coastal monitoring two trends](https://raw.githubusercontent.com/tommasobonomo/BlankSpace/master/demo_images/Coastal_two_trends.png)

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