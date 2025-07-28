# C.elegans Preprocessing Notebooks

This directory contains preprocessing notebooks for C.elegans single-cell RNA-sequencing datasets used in CellTreeBench.

## Overview

This directory contains preprocessing pipelines for C.elegans single-cell RNA-sequencing datasets used in CellTreeBench. Each dataset has different preprocessing requirements ranging from complex multi-notebook pipelines to direct raw data usage:

- **celegans_small**: Multi-phase preprocessing with tree curation pipeline + data processing notebook
- **celegans_mid**: Single notebook preprocessing with regex-based lineage mapping  
- **celegans_large**: No preprocessing required - direct raw data usage

## celegans_small Dataset Preprocessing

### Purpose
This section describes the complete preprocessing pipeline for the C.elegans small-scale dataset, which creates a curated dataset from high-quality lineage tree structures and corresponding single-cell expression data.

### Prerequisites
The preprocessing consists of two main phases. First, you must build the curated lineage tree using the notebooks in the `build_tree_celegans_small/` directory:

1. **`0-find_subtrees.ipynb`** - Discovers molecular lineage subtrees from raw lineage data
2. **`1-find-molecular-subtrees.ipynb`** - Builds detailed tree structures for each lineage subtree
3. **`3-merge_tree.ipynb`** - Merges and curates subtrees into final P0 tree structure

### Key Functions

1. **Curated Tree Integration**
   - Loads the pre-built P0 lineage tree from `build_tree_celegans_small/` pipeline
   - Validates tree structure and cell count annotations
   - Uses expert-curated tree with 103 leaf nodes covering major C.elegans lineages

2. **Expression Data Loading**
   - Intelligently detects available expression data formats (Matrix Market, CSV)
   - Loads cell barcodes and gene annotations
   - Handles both normalized and raw expression matrices

3. **Data Alignment and Filtering**
   - Maps cells between expression data and lineage tree annotations
   - Filters to retain only cells with valid lineage assignments
   - Ensures consistency between metadata, expression data, and tree structure

4. **Standardized Output Generation**
   - Saves data in `CElegansDatasetBase`-compatible formats
   - Generates sparse Matrix Market files for efficient storage
   - Updates tree with actual cell counts from filtered data

### Output Files

```
/workspaces/CellTreeBench/data/celegans_small/
├── P0/
│   └── tree_df-P0.csv                # Curated P0 tree with cell counts
└── raw/
    ├── metadata.csv                  # Filtered cell metadata
    ├── exprs.mm                      # Expression matrix (sparse format)
    ├── cell_barcodes.csv             # Cell identifiers
    └── genes.csv                     # Gene names
```

### Dataset Statistics
- **Tree nodes**: 103 leaf nodes representing major lineages
- **Expert curation**: Hand-curated tree structure with quality filtering
- **Expression data**: Variable based on input data (typically ~20K genes)
- **Cells**: Filtered to cells with valid lineage annotations

### Complete Workflow
1. **Phase 1**: Run the tree building pipeline in `build_tree_celegans_small/` (3 notebooks)
2. **Phase 2**: Run `preprocess_celegans_small.ipynb` to generate final dataset
3. **Usage**: Dataset ready for `CElegansDatasetBase(dataset_name="celegans_small")`

## celegans_large Dataset (No Preprocessing Notebook Required)

### Overview
The `celegans_large` dataset is designed to work **directly with raw data files** and does not require a preprocessing notebook. This represents the most "plug-and-play" option among the C.elegans datasets.

### Why No Preprocessing Notebook?
- **Direct Raw Data Usage**: The `CElegansDatasetBase` class can load GSE126954 files directly
- **Pre-built Components**: Essential preprocessing outputs (tree and cell mappings) are already available
- **Automatic Processing**: All data loading and filtering happens automatically in the dataset class

### Data Source
- **Origin**: GSE126954 - Large-scale C.elegans single-cell atlas
- **Scale**: Full organism-level single-cell dataset (389K cells)
- **Format**: Standard Matrix Market format with separate annotation files

### Required Files Structure

The dataset expects these files to be present (no notebook needed to generate them):

```
/workspaces/CellTreeBench/data/celegans_large/
├── P0/
│   ├── p0-topology_tree.nwk          # Pre-built lineage tree
│   └── P0-cell_to_leaf_df.csv        # Pre-computed cell-to-lineage mapping
└── raw/
    ├── GSE126954_gene_by_cell_count_matrix.txt    # Original expression matrix
    ├── GSE126954_gene_annotation.csv              # Gene annotations  
    └── GSE126954_cell_annotation.csv              # Cell annotations
```

### Usage
Simply instantiate the dataset class - **no preprocessing steps required**:
```python
from celltreebench.datasets.celegans_dataset_base import CElegansDatasetBase
dataset = CElegansDatasetBase(dataset_name="celegans_large")
```

The dataset class automatically handles:
- Loading and parsing raw expression matrix
- Applying pre-computed cell-to-lineage mappings  
- Filtering to cells with valid lineage assignments
- Creating properly indexed DataFrames

### Dataset Statistics
- **Original data**: ~389K cells from full C.elegans organism
- **Filtered data**: ~34K cells with lineage annotations
- **Tree structure**: Comprehensive pre-built lineage tree
- **Genes**: ~20K features

## celegans_mid Dataset Preprocessing

### Purpose
This section describes preprocessing the C.elegans medium-scale dataset from Packer et al., creating a standardized dataset suitable for lineage tree reconstruction benchmarking through the `preprocess_celegans_mid.ipynb` notebook.

### Key Functions

1. **Lineage Tree Processing**
   - Loads the reference lineage tree from Packer et al. (668 leaf nodes)
   - Maps single-cell lineage annotations to tree leaves using regex pattern matching
   - Prunes the tree to retain only leaves with corresponding single-cell data (final: 183 leaves)
   - Saves the processed tree in Newick format

2. **Cell-to-Leaf Mapping**
   - Processes complex lineage annotations (e.g., "ABp[xaplrvd]aapaap") into specific tree leaves
   - Handles ambiguous lineage patterns through systematic matching
   - Creates a mapping from individual cells to terminal lineage nodes

3. **Expression Data Processing**
   - Filters the original expression matrix (389,755 cells × 27,138 genes) 
   - Retains only C.elegans cells with valid lineage annotations (25,383 cells)
   - Outputs expression data in standardized CSV format

4. **Metadata Integration**
   - Merges cell annotations with lineage mapping information
   - Adds `leaf` column to link cells to their corresponding tree nodes
   - Preserves original metadata while adding tree-specific annotations

### Output Files

The notebook generates three files in the standard CellTreeBench data directory structure:

```
/workspaces/CellTreeBench/data/celegans_mid/
├── P0/
│   └── p0-topology_tree.nwk          # Processed lineage tree
└── raw/
    ├── c_elegans_expression_df.csv   # Filtered expression matrix
    └── c_elegans_cell_meta.csv       # Enhanced cell metadata
```

### Dataset Statistics
- **Original data**: 389,755 cells across multiple species
- **Filtered data**: 25,383 C.elegans cells with lineage annotations
- **Final dataset**: 18,514 cells mapped to 183 tree leaves
- **Genes**: 27,138 features

### Usage
Run `preprocess_celegans_mid.ipynb` when setting up the `celegans_mid` dataset for the first time or when updating the preprocessing pipeline. The output files are directly compatible with the `CElegansDatasetBase(dataset_name="celegans_mid")` class initialization.

## Acknowledgements

Some of the preprocessing steps follow similar approaches as described in the [PORCELAN](https://github.com/uhlerlab/porcelan) project:

```bibtex
@article{schluter2025integrating,
  title={Integrating representation learning, permutation, and optimization to detect lineage-related gene expression patterns},
  author={Schl{\"u}ter, Hannah M and Uhler, Caroline},
  journal={Nature Communications},
  volume={16},
  number={1},
  pages={1062},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
```

**Reference**: [https://github.com/uhlerlab/porcelan](https://github.com/uhlerlab/porcelan)
