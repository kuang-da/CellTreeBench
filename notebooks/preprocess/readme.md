# Preprocess
This folder contains the notebooks for preprocessing the datasets. 

## Raw Data
The input for notebooks in this directory are all the Raw Data are stored in the `/workspaces/CellTreeBench/data/raw` folder.

- `/workspaces/CellTreeBench/data/raw/celegans_packer` is from paper [Packer et al., 2019](https://www.science.org/doi/10.1126/science.aax1971). It is used to generate `C.elegans Large` dataset.
- `/workspaces/CellTreeBench/data/raw/celegans_packer_curation` is also from paper [Packer et al., 2019](https://www.science.org/doi/10.1126/science.aax1971) but it contains processed data table prepared by Chris Large. Da Kuang curated a dataset based on this data table and generate `C.elegans Small` dataset.
- `/workspaces/CellTreeBench/data/raw/celegans_chris` is from paper [Large et al., 2025](https://www.science.org/doi/full/10.1126/science.adu8249?af=R). It is used to generate `C.elegans Mid` dataset.

## Preprocess
The output of notebooks in this directory are all the preprocessed data are stored in some subdirectory of `/workspaces/CellTreeBench/data` folder with the same name as the dataset.

### C. elegans
- `C.elegans Large` dataset is stored in `/workspaces/CellTreeBench/data/celegans_large` folder.
- `C.elegans Small` dataset is stored in `/workspaces/CellTreeBench/data/celegans_small` folder.
- `C.elegans Mid` dataset is stored in `/workspaces/CellTreeBench/data/celegans_mid` folder.

### C. briggsae

### DNA Methylation

### CRISPR
