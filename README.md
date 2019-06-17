# Extract-Cosmology-with-the-BAO-peak-position

Package to extract cosmology from the BAO peak position with precomputed data from the Sloan Digital Sky Survey https://www.sdss.org/

# Installation
Simply git it:
```
git clone https://github.com/lontelis/Extract-Cosmology-with-the-BAO-peak-position.git
```

if iminuit is not install you can install it through:
```
pip install iminuit --user
```

if CLASS (software for the Theoretical Power Spectra) is not installed:
```
git clone https://github.com/lesgourg/class_public.git class
cd class/
make
```
For more details check:
https://github.com/lesgourg/class_public/wiki/Installation

# Use
```
cd Extract-Cosmology-with-the-BAO-peak-position
jupyter notebook
```
Then select the analyse_DR12_BAO.ipynb to start with.
