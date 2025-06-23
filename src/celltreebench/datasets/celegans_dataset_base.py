import logging
import sys
import os
import pickle
import pandas as pd
from torch.utils.data import Dataset
from ete3 import Tree
from scipy.io import mmread
import numpy as np
import torch
import random
from itertools import combinations
from math import comb

from celltreebench.utils.reconstruction_eval import compare_trees
from celltreebench.utils.tree_operations import get_path_distance_matrix

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the minimum logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],  # This will output logs to the console
)

# Get a logger object
logger = logging.getLogger(__name__)

# Find the project root
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")


class CElegansDatasetBase(Dataset):
    _exprs_df_cache = None

    def __init__(
        self,
        dist_metric="euclidean",
        data_dir=None,
        out_dir=None,
        dataset_name="celegans_dev",  # "celegans_dev" or "celegans_packer"
        lineage_name="P0",
        subset_tree_with_leaves=None,
        # leaves_metadata_with_levels=True,
        quartet_sampling_method="random",  # "random" or "exhaustive"
        _permute_leaves=False,
        _permutation_seed=None,
    ):
        # Validate input parameters
        assert quartet_sampling_method in [
            "random",
            "exhaustive",
        ], "Invalid sampling method"
        self.sampling_method = quartet_sampling_method

        assert dataset_name in [
            # "celegans_dev",
            "celegans_small",
            "celegans_mid",
            "celegans_large",
            # "celegans_packer",
        ], "Invalid dataset name"
        self.dataset_name = dataset_name

        if data_dir is None:
            data_dir = DATA_ROOT

        self.data_dir = os.path.join(data_dir, dataset_name)
        # If data_dir does not exist, raise an error
        if not os.path.exists(self.data_dir):
            raise ValueError(f"Data directory does not exist: {self.data_dir}")

        # Initialize parameters
        self.lineage_name = lineage_name
        self.dist_metric = dist_metric
        self.out_dir = out_dir
        if not os.path.exists(self.out_dir):
            logger.info(f"Creating output directory: {self.out_dir}")
            os.makedirs(self.out_dir)

        # Initialize data attributes
        self.data_normalized = None
        self.data = None
        self.ref_dm = None
        self.dim = None
        self._permutation_seed = _permutation_seed
        # Load Data (1): Load the lineage tree
        self.topology_tree = self._create_topology_tree()
        if subset_tree_with_leaves:
            self._prune_tree(subset_tree_with_leaves)
        if _permute_leaves:
            logger.info(f"Permuting leaf names with see {self._permutation_seed}...")
            self._permute_leaf_names(self._permutation_seed)
        # self._erase_internal_node_names()
        self.topology_tree.standardize()
        self._save_tree_to_file()

        self.leave_names = [leaf.name for leaf in self.topology_tree.iter_leaves()]
        self.n_leaves = len(self.leave_names)

        # Load Data (2): Load metadata and expression data
        self.metadata_df = self._load_metadata()

        # Load Data (3): Load expression data
        if CElegansDatasetBase._exprs_df_cache is None:
            CElegansDatasetBase._exprs_df_cache = {}
        if self.dataset_name in CElegansDatasetBase._exprs_df_cache:
            logger.info("Using cached expression data...")
            self.exprs_df = CElegansDatasetBase._exprs_df_cache[self.dataset_name]
        else:
            self.exprs_df = self._load_expression_data()
            CElegansDatasetBase._exprs_df_cache[self.dataset_name] = self.exprs_df
            logger.info("Cached expression data as class statics for future use")

        # Prepare quartets for iteration

        self.total_quartets = comb(self.n_leaves, 4)
        logger.info(f"Calculated total quartets: {self.total_quartets}")
        # self.quartet_indices = list(combinations(range(self.n_leaves), 4))

        # Create and annotate leaves metadata
        logger.info("Creating leaves metadata...")
        self.leaves_metadata = self._create_leaves_metadata()
        # Compute the full reference distance matrix between leaves
        # self.leave_names have the same order as the rows and cols orders in self.ref_dm
        logger.info("Creating reference distance matrix...")
        self.ref_dm = get_path_distance_matrix(self.topology_tree, self.leave_names)

    def _erase_internal_node_names(self):
        """
        Removes the names of all internal nodes in the given ete3 tree.

        Parameters:
            tree (ete3.Tree): The input tree whose internal node names will be erased.

        Returns:
            None: The tree is modified in-place.
        """
        tree = self.topology_tree
        for node in tree.traverse():
            # Check if the node is an internal node (not a leaf)
            if not node.is_leaf():
                node.name = ""

    def _permute_leaf_names(self, seed=None):
        """
        Permutes the leaf names of a given ete3 tree in-place, optionally using a seed for reproducibility.

        Parameters:
            seed (int, optional): A seed for the random number generator. Defaults to None for non-deterministic behavior.

        Returns:
            None: The tree is modified in-place.
        """
        tree = self.topology_tree
        # Get all leaf nodes
        leaves = tree.get_leaves()

        # Extract leaf names
        leaf_names = [leaf.name for leaf in leaves]

        # Shuffle the leaf names using a seeded random generator
        rng = random.Random(seed)
        shuffled_names = leaf_names[:]
        rng.shuffle(shuffled_names)

        # Replace the original names with the shuffled names
        for leaf, new_name in zip(leaves, shuffled_names):
            leaf.name = new_name

    def select_features(self, feature_list):
        """Selects a subset of features based on the provided list."""
        self.data_normalized = self.data_normalized[feature_list]
        self.dim = self.data_normalized.shape[1]
        self.unshuffled_data_tensor = torch.tensor(
            self.data_normalized.values, dtype=torch.float32
        )

    def __len__(self):
        return self.total_quartets

    def __getitem__(self, idx):
        # raise not implementated error
        raise NotImplementedError("__getitem__ method is not implemented yet.")
        # quartet_idx = list(self.quartet_indices[idx])  # Indices of the four leaves
        # quartet_names = [self.leave_names[i] for i in quartet_idx]

        # # Fetch the data corresponding to these names
        # quartet_data = self.data_normalized.iloc[
        #     quartet_idx
        # ].values  # Shape: (4, n_features)

        # # Get the reference distance matrix for these quartet indices
        # quartet_ref_dm = self.ref_dm[np.ix_(quartet_idx, quartet_idx)]  # Shape: (4, 4)

        # # Convert data to tensors
        # quartet_data = torch.tensor(quartet_data, dtype=torch.float32)
        # quartet_idx_tensor = torch.tensor(quartet_idx, dtype=torch.long)
        # # quartet_ref_dm = torch.tensor(quartet_ref_dm, dtype=torch.float32)
        # return quartet_data, quartet_names, quartet_idx_tensor, quartet_ref_dm

    def _create_topology_tree(self):
        """Construct the topology tree from lineage tree dataframe."""
        logger.info(f"Creating topology tree for dataset [{self.dataset_name}]")

        if self.dataset_name == "celegans_small":
            file_name = f"tree_df-{self.lineage_name}.csv"
        elif (
            self.dataset_name == "celegans_mid" or self.dataset_name == "celegans_large"
        ):
            file_name = "p0-topology_tree.nwk"

        file_name = os.path.join(self.data_dir, self.lineage_name, file_name)
        if not os.path.exists(file_name):
            raise FileNotFoundError(f"Lineage tree file not found: {file_name}")

        tree = self._create_tree(file_name, self.dataset_name)
        return tree

    def _save_tree_to_file(self):
        """Save the topology tree to files in ASCII and pickle formats."""
        if self.out_dir:
            topology_txt = os.path.join(self.out_dir, "topology_tree-ncells.txt")
            topology_pickle = os.path.join(self.out_dir, "topology_tree.pickle")

            with open(topology_txt, "w") as f:
                f.write(self.topology_tree.get_ascii(attributes=["name", "n_cells"]))
            logger.info(f"Saved topology tree to {topology_txt}")

            with open(topology_pickle, "wb") as f:
                pickle.dump(self.topology_tree, f)
            logger.info(f"Saved topology tree pickle to {topology_pickle}")
        else:
            logger.warning("Output directory not specified. Tree files not saved.")

    def _load_metadata(self):
        """Load the metadata table."""
        if self.dataset_name == "celegans_small":
            file_name = "metadata.csv"
        if self.dataset_name == "celegans_large":
            file_name = "GSE126954_cell_annotation.csv"
        if self.dataset_name == "celegans_mid":
            file_name = "c_elegans_cell_meta.csv"

        file_name = os.path.join(self.data_dir, "raw", file_name)
        if not os.path.exists(file_name):
            raise FileNotFoundError(f"Metadata file not found: {file_name}")

        if self.dataset_name == "celegans_small":
            metadata_df = pd.read_csv(file_name, low_memory=False)
            metadata_df = metadata_df[metadata_df.lineage_packer.isin(self.leave_names)]
            metadata_df = metadata_df.set_index("cell", drop=False)
            # Prepare the leaf column
            metadata_df["leaf"] = metadata_df["lineage_packer"]
        elif self.dataset_name == "celegans_large":
            metadata_df = pd.read_csv(file_name, index_col=0)
            metadata_df["og_idx"] = range(len(metadata_df))

            # For parker dataset, the cells for the lineage are pre-selected in cell_to_leaf_df.
            # We need to load the file and merge it with the metadata_df.
            # So that we can map cells to leaves.
            # Only the cells belong to the lineage with have non-empty leaf column.
            file_name = f"{self.lineage_name}-cell_to_leaf_df.csv"
            file_name = os.path.join(self.data_dir, self.lineage_name, file_name)
            cell_to_leaf_df = pd.read_csv(file_name)
            # Merge "regex_joined" and "leaf" columns to metadata_df
            metadata_df = metadata_df.merge(cell_to_leaf_df, on="og_idx", how="left")
            # Only keep the cells that have a leaf
            metadata_df = metadata_df[metadata_df.leaf.notna()]
        elif self.dataset_name == "celegans_mid":
            metadata_df = pd.read_csv(file_name, index_col=0)
        return metadata_df

    def _load_expression_data(self):
        """Load gene expression data and cache it."""
        cache_file = "exprs_df_cache.pkl"
        cache_file = os.path.join(self.data_dir, self.lineage_name, cache_file)
        if os.path.exists(cache_file):
            logger.info(f"Loading cached gene expression data from {cache_file}")
            with open(cache_file, "rb") as f:
                exprs_df = pickle.load(f)
        else:
            logger.info("Loading gene expression data from files...")
            if self.dataset_name == "celegans_small":
                exprs_df = self._load_expression_data_small()
            elif self.dataset_name == "celegans_large":
                exprs_df = self._load_expression_data_large()
            elif self.dataset_name == "celegans_mid":
                exprs_df = self._load_expression_data_mid()
            print(
                f"exprs_df.shape: {exprs_df.shape}, metadata_df.shape: {self.metadata_df.shape}"
            )
            exprs_df = exprs_df.loc[self.metadata_df.cell]
            print(f"exprs_df.shape after filtering: {exprs_df.shape}")

            # Save to cache
            with open(cache_file, "wb") as f:
                pickle.dump(exprs_df, f)

            logger.info(f"Cached gene expression data to {cache_file}")

        # Check if the expression data index is the same order as the metadata cell column
        cell_list1 = exprs_df.index.tolist()
        cell_list2 = self.metadata_df.cell.tolist()
        if cell_list1 != cell_list2:
            logging.info(
                "Cells in expression data and metadata do not match due to subsetting the tree."
            )
            logging.info(
                f"Select cells based on metadata cell column: {len(cell_list1)} out of {len(cell_list2)}"
            )
            exprs_df = exprs_df.loc[self.metadata_df.cell]
        return exprs_df

    def _load_expression_data_mid(self):
        file_name = "c_elegans_expression_df.csv"
        file_name = os.path.join(self.data_dir, "raw", file_name)
        exprs_df = pd.read_csv(file_name, index_col=0)
        return exprs_df

    def _load_expression_data_large(self):
        file_name = "GSE126954_gene_by_cell_count_matrix.txt"
        file_name = os.path.join(self.data_dir, "raw", file_name)
        expression_adata = mmread(file_name)  # (n_genes, n_cells)
        org_idx = range(expression_adata.shape[1])

        file_name = "GSE126954_gene_annotation.csv"
        file_name = os.path.join(self.data_dir, "raw", file_name)
        all_gene_names = pd.read_csv(file_name)
        all_gene_names = all_gene_names["gene_short_name"].values

        file_name = os.path.join("GSE126954_cell_annotation.csv")
        file_name = os.path.join(self.data_dir, "raw", file_name)
        cell_annotation_df = pd.read_csv(file_name, index_col=0)
        all_cell_names = cell_annotation_df.index.tolist()

        exprs_df = pd.DataFrame(
            expression_adata.toarray().T,
            index=all_cell_names,
            columns=all_gene_names,
        )

        return exprs_df

    def _load_expression_data_small(self):
        """Load expression data from files and cache it."""
        # Load row names (cell barcodes)
        barcodes_file = os.path.join(self.data_dir, "raw", "cell_barcodes.csv")
        barcodes_list = pd.read_csv(barcodes_file, header=None).values.flatten()
        logger.info(f"Loaded barcodes from {barcodes_file}")

        # Load column names (genes)
        genes_file = os.path.join(self.data_dir, "raw", "genes.csv")
        genes_list = pd.read_csv(genes_file, header=None).values.flatten()
        logger.info(f"Loaded gene list from {genes_file}")

        # Load sparse matrix (gene expression data)
        exprs_file = os.path.join(self.data_dir, "raw", "exprs.mm")
        sparse_matrix = mmread(exprs_file)
        logger.info(f"Loaded sparse expression matrix from {exprs_file}")

        # Convert sparse matrix to DataFrame
        exprs_df = pd.DataFrame(
            sparse_matrix.toarray(), index=barcodes_list, columns=genes_list
        )
        # logging.info(exprs_df.head())
        return exprs_df

    def _create_leaves_metadata(self):
        """Create and return a table counting cells for each lineage node."""
        leaves_metadata = pd.DataFrame(columns=["lineage", "n_cells"])
        for leaf in self.topology_tree.iter_leaves():
            metadata_df_lineage = self.metadata_df[self.metadata_df.leaf == leaf.name]
            exprs_df_lineage = self.exprs_df.loc[metadata_df_lineage.cell]
            leaves_metadata = pd.concat(
                [
                    leaves_metadata,
                    pd.DataFrame(
                        {"lineage": [leaf.name], "n_cells": [exprs_df_lineage.shape[0]]}
                    ),
                ],
                ignore_index=True,
            )
            logger.debug(
                f"Added lineage {leaf.name} with {exprs_df_lineage.shape[0]} cells to leaves metadata"
            )

        # if self.annotate_levels:
        #     self._annotate_leaves_with_levels(leaves_metadata)

        return leaves_metadata

    def _annotate_leaves_with_levels(self, leaves_metadata):
        """Annotate leaves with the levels of parent nodes."""
        levels = {
            "level_2": ["AB", "P1"],
            "level_3": ["ABPx", "ABaxx", "EMS", "P2"],
            "level_4": ["ABpxp", "ABpxax", "ABarpx", "MSx", "Exx", "Cx", "Dx"],
            "level_5": [
                "ABpxpa",
                "ABpxpp",
                "ABpxapa",
                "ABpxaap",
                "ABarpaa",
                "ABarppx",
                "MSxa",
                "MSxp",
                "Eaxaa",
                "Epxpp",
                "Cxa",
                "Cxp",
                "Dxxa",
                "Dxxp",
            ],
        }

        for level_name, level_values in levels.items():
            for value in level_values:
                node = self.topology_tree.search_nodes(name=value)
                if node:
                    leaves = node[0].get_leaves()
                    leaves_names = [leaf.name for leaf in leaves]
                    self.leaves_metadata.loc[
                        self.leaves_metadata["lineage"].isin(leaves_names), level_name
                    ] = value

    def _determine_dim(self):
        """Determine the dimension of the dataset."""
        if self.data_normalized is not None:
            return self.data_normalized.shape[1]
        elif self.exprs_df is not None:
            return self.exprs_df.shape[1]
        return None

    def _prune_tree(self, leave_list):
        """Prune the topology tree to include only specific leaves."""
        logger.info(f"Pruning tree to keep only leaves: {leave_list}")
        mrca = self.topology_tree.get_common_ancestor(leave_list)
        subtree = mrca.copy()
        subtree.prune(leave_list, preserve_branch_length=False)
        self.topology_tree = subtree
        logger.info("Tree pruning completed")

    def _create_tree(self, tree_file_name, dataset_name):
        logger.info(f"Loading lineage tree from {tree_file_name}")

        if dataset_name == "celegans_small":
            tree_df = pd.read_csv(tree_file_name)
            tree = self._create_tree_from_dataframe(tree_df)
            self._remove_leaves_with_zero_cells(tree)
            return tree
        elif dataset_name == "celegans_large" or dataset_name == "celegans_mid":
            tree = Tree(tree_file_name, format=1)
        else:
            raise ValueError("Unknown dataset name")
        return tree

    def _create_tree_from_dataframe(self, tree_df):
        """Create a tree from the given dataframe."""
        node_dict = {}
        root_name = tree_df["Lineage"].iloc[0]
        tree = Tree(name=root_name)
        tree.add_feature("n_cells", tree_df["n_cells"].iloc[0])
        node_dict[root_name] = tree

        for _, row in tree_df.iterrows():
            lineage_name = row["Lineage"]
            parent_name = row["Parent"]

            if lineage_name == root_name:
                continue

            if lineage_name not in node_dict:
                node = Tree(name=lineage_name)
                node.add_feature("n_cells", row["n_cells"])
                node_dict[lineage_name] = node

            if parent_name in node_dict:
                node_dict[parent_name].add_child(node_dict[lineage_name])
            else:
                print(f"ERR: Parent node {parent_name} not found for {lineage_name}.")
        return tree

    def _remove_leaves_with_zero_cells(self, tree):
        """Remove all leaves from the tree that have zero cells."""
        while True:
            removed = False
            for leaf in tree.iter_leaves():
                if getattr(leaf, "n_cells", 0) == 0:
                    leaf.detach()
                    removed = True
                    logger.debug(f"Detached leaf {leaf.name} with zero cells")
            if not removed:
                break
        logger.info("Completed removal of leaves with zero cells")

    def _normalize(self, x, mean=None, std=None):
        if mean is None:
            mean = x.mean()
            std = x.std()
        return (x - mean) / std, mean, std

    def get_node_mtx(self):
        """
        Returns a dictionary containing the node matrix and node names.

        Returns:
            dict: A dictionary with 'node_mtx' as a NumPy array of the normalized data
            and 'node_names' as the corresponding index.
        """
        return {
            "node_mtx": self.data_normalized.to_numpy(),
            "node_names": self.data_normalized.index,
        }

    def compare_trees(self, tree1, ref_tree="topology_tree", unrooted=True):
        """
        Compare two trees using the specified reference tree and the unrooted flag.

        Args:
            tree1 (Tree): The tree to compare.
            ref_tree (str): The name of the reference tree, default is 'topology_tree'.
            unrooted (bool): Whether to compare the trees as unrooted. Default is True.

        Returns:
            float: A similarity score between the two trees.
        """
        logger.debug(f"Comparing trees with reference to {ref_tree}")
        if ref_tree == "topology_tree":
            tree2 = self.topology_tree
        else:
            raise ValueError("Unknown reference tree specified.")

        return compare_trees(tree1, tree2, unrooted_trees=unrooted)
