# K2-18b

This repository reproduces Figures 1 through 4 in Wogan et al. (2024), which is a paper on the K2-18b exoplanet. There are two steps to run the code, as outlined below. This code will only work on MacOS or a Linux OS (Windows will not work).

## Installation and setup

If you do not have Anaconda on your system, install it here or in any way you perfer: https://www.anaconda.com/download . Next, run the following code to cerate a conda environment `k218b` with `photochem` v0.4.5 installed.

```bash
conda create -n k218b -c conda-forge -c bokeh bokeh=2.4.3 python photochem=0.4.5 numpy=1.24 matplotlib pandas pip numba xarray cantera pathos threadpoolctl miepython matplotlib-label-lines
conda activate k218b
```

Next, run the code below to install and setup picaso v3.1.2.

```bash
pip install picaso==3.1.2

# Get reference
wget https://github.com/natashabatalha/picaso/archive/4d90735.zip
unzip 4d90735.zip
cp -r picaso-4d907355da9e1dcca36cd053a93ef6112ce08807/reference input/picaso/
export picaso_refdata=$(pwd)"/input/picaso/reference/"
echo $picaso_refdata
rm -rf picaso-4d907355da9e1dcca36cd053a93ef6112ce08807
rm 4d90735.zip

# Get opacities
wget https://zenodo.org/records/3759675/files/opacities.db
mv opacities.db input/picaso/reference/opacities/

# Get the star stuff
wget http://ssb.stsci.edu/trds/tarfiles/synphot3.tar.gz
tar -xvzf synphot3.tar.gz
mv grp input/picaso/
export PYSYN_CDBS=$(pwd)"/input/picaso/grp/redcat/trds"
echo $PYSYN_CDBS
rm synphot3.tar.gz

# Get more star stuff
wget https://archive.stsci.edu/hlsps/reference-atlases/hlsp_reference-atlases_hst_multi_pheonix-models_multi_v3_synphot5.tar
tar -xvzf hlsp_reference-atlases_hst_multi_pheonix-models_multi_v3_synphot5.tar
mv grp/redcat/trds/grid/phoenix input/picaso/grp/redcat/trds/grid/
rm hlsp_reference-atlases_hst_multi_pheonix-models_multi_v3_synphot5.tar

# Get climate opacity
wget https://zenodo.org/records/7542068/files/sonora_2020_feh+200_co_100.data.196.tar.gz
tar -xvzf sonora_2020_feh+200_co_100.data.196.tar.gz
mv sonora_2020_feh+200_co_100.data.196 input/picaso/climate/
rm sonora_2020_feh+200_co_100.data.196.tar.gz
```

## Run the code

To do all calculations, and reproduce Figure 1 to 4 in the paper, run all the code with

```bash
# environment setup
conda activate k218b
export picaso_refdata=$(pwd)"/input/picaso/reference/"
export PYSYN_CDBS=$(pwd)"/input/picaso/grp/redcat/trds"

# run the main script
python main.py
```

The code might take 1 or 2 hours to run. Once completed, Figures 1 to 4 are in the `figures/` directory.
