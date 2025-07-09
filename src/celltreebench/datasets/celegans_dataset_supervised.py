import logging
import os
import pickle
import torch
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
from ete3 import Tree
import logging
import sys
import random
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.io import mmread
from sklearn.decomposition import PCA
from math import comb

from celltreebench.datasets.celegans_dataset_base import CElegansDatasetBase

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the minimum logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],  # This will output logs to the console
)
# Get a logger object
logger = logging.getLogger(__name__)


class CElegansDatasetSupervised(CElegansDatasetBase):
    def __init__(
        self,
        dist_metric="euclidean",  # ------------------- Base parameters
        data_dir=None,
        out_dir=None,
        dataset_name=None,  # "celegans_dev" or "celegans_packer"
        lineage_name="P0",
        subset_tree_with_leaves=None,
        quartet_sampling_method="random",  # "random" or "exhaustive"
        train=False,  # ------------------- Supervised parameters
        supervised_data_dict=None,
        data_split_seed=None,
        lineage_node_sampling_strategy="biological",  # biological or technical
        _permute_leaves=False,
        _permutation_seed=None,
    ):
        super().__init__(
            dist_metric=dist_metric,
            data_dir=data_dir,
            out_dir=out_dir,
            dataset_name=dataset_name,
            lineage_name=lineage_name,
            subset_tree_with_leaves=subset_tree_with_leaves,
            quartet_sampling_method=quartet_sampling_method,
            _permute_leaves=_permute_leaves,
            _permutation_seed=_permutation_seed,
        )
        self.supervised_data_dict = supervised_data_dict
        self.train = train
        self.data_split_seed = data_split_seed
        self.dim = None
        self.sampling_strategy = lineage_node_sampling_strategy
        if supervised_data_dict is None:
            logger.info(f"Generating supervised data with seed: {self.data_split_seed}")
            self.prepare_supervised_data()
            self.supervised_data_dict = self._create_supervised_data_dict()
            # Save list of genes to a file
            # We filter out genes with low expression in the training data
            # The filter can be slightly different depending on the splitting seed.
            # We save the gene list to ensure reproducibility.
            self._save_genes_list()
        else:
            if self.train:
                logger.info(f"Generated supervised data provided. Training mode.")
            else:
                logger.info(f"Generated supervised data provided. Testing mode.")
            self._set_data_based_on_training_mode()
            self.dim = self._determine_dim()

    def _save_genes_list(self):
        gene_list = self.exprs_df.columns.tolist()
        gene_list_file = os.path.join(self.out_dir, "gene_list.pkl")
        with open(gene_list_file, "wb") as f:
            pickle.dump(gene_list, f)

    def _create_supervised_data_dict(self):
        """Creates a dictionary containing the train/test data and references."""
        return {
            "train": self.data_train,
            "test": self.data_test,
            "train_normalized": self.data_normalized_train,
            "test_normalized": self.data_normalized_test,
            "train_feature_stats": self.train_feature_stats,
            "test_feature_stats": self.test_feature_stats,
            "ref_dm": self.ref_dm,
            "topology_tree": self.topology_tree,
        }

    def _set_data_based_on_training_mode(self):
        """Sets train/test data based on training flag and applies PCA if required."""
        self.ref_dm = self.supervised_data_dict["ref_dm"]
        self.topology_tree = self.supervised_data_dict["topology_tree"]
        self.leave_names = self.topology_tree.get_leaf_names()
        self.n_leaves = len(self.leave_names)
        self.total_quartets = comb(self.n_leaves, 4)
        if self.train:
            self.data = self.supervised_data_dict["train"]
            self.data_normalized = self.supervised_data_dict["train_normalized"]
        else:
            self.data = self.supervised_data_dict["test"]
            self.data_normalized = self.supervised_data_dict["test_normalized"]

    def prepare_supervised_data(self):
        if self.data_split_seed > 0:
            # Generate expression for each lineage leaves and split into train/test
            logger.info(f"[prepare_supervised_data] Seed: {self.data_split_seed}")
            rng = np.random.default_rng(self.data_split_seed)
        else:
            logger.info("[prepare_supervised_data] Seed: None")
            rng = np.random.default_rng()

        self._filter_low_expressed_genes()
        self._split_and_aggregate_train_test(rng)

    def _filter_low_expressed_genes(self):
        """Filters out genes that expressed in fewer than 10 mRNAs across all cells."""
        gene_mask = self.exprs_df.sum(axis=0) >= 10
        logger.info(
            f"Genes have less than 10 umis in total: {len(gene_mask) - gene_mask.sum()} out of {len(gene_mask)}"
        )
        self.exprs_df = self.exprs_df.loc[:, gene_mask]
        logger.info(f"Remaining genes: {self.exprs_df.shape[1]}")
        # TODO: select genes based number of cells

    def _split_and_aggregate_train_test(self, rng):
        """Splits expression data into training and testing sets using the specified sampling strategy."""

        def _calculate_feature_statistics(exprs_df_lineage):
            """Helper function to calculate mean, median, std, CoV for a given lineage expression data."""

            mean_exprs = exprs_df_lineage.mean()
            median_exprs = exprs_df_lineage.median()
            std_exprs = exprs_df_lineage.std()
            cov_exprs = std_exprs / mean_exprs.replace(
                0, np.nan
            )  # Handle division by zero

            # Create a feature-by-4 DataFrame with statistics
            feature_stats_df = pd.DataFrame(
                {
                    "mean": mean_exprs,
                    "median": median_exprs,
                    "std": std_exprs,
                    "cov": cov_exprs,
                }
            )

            return feature_stats_df

        exprs_df_train, exprs_df_test = pd.DataFrame(), pd.DataFrame()
        train_feature_stats, test_feature_stats = (
            {},
            {},
        )

        if self.sampling_strategy not in ["biological", "technical"]:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")
        else:
            logger.info(f"Sampling strategy: {self.sampling_strategy}")
        logger.info(
            f"Calculated feature statistics for {len(self.leaves_metadata)} leaves"
        )

        for the_leaf in self.leave_names:
            lineage_cells = self.metadata_df[self.metadata_df.leaf == the_leaf][
                "cell"
            ].values
            exprs_df_lineage = self.exprs_df.loc[lineage_cells]

            if self.sampling_strategy == "biological":
                # Shuffle indices and split into train/test sets
                idx = rng.permutation(exprs_df_lineage.index)
                mid_point = len(idx) // 2
                idx_train, idx_test = idx[:mid_point], idx[mid_point:]

                # Compute mean expression for train and test
                avg_train, avg_test = (
                    exprs_df_lineage.loc[idx_train].mean(),
                    exprs_df_lineage.loc[idx_test].mean(),
                )

                # Append mean values to the train and test DataFrames
                exprs_df_train = pd.concat(
                    [exprs_df_train, avg_train.to_frame().T], ignore_index=True
                )
                exprs_df_test = pd.concat(
                    [exprs_df_test, avg_test.to_frame().T], ignore_index=True
                )

                # Calculate feature-level statistics for training and testing sets
                train_feature_stats[the_leaf] = _calculate_feature_statistics(
                    exprs_df_lineage.loc[idx_train]
                )
                test_feature_stats[the_leaf] = _calculate_feature_statistics(
                    exprs_df_lineage.loc[idx_test]
                )

            elif self.sampling_strategy == "technical":
                # Summing counts to simulate total expression for each gene
                counts_total = exprs_df_lineage.sum(axis=0).astype(int)

                # Simulate technical replicates using binomial distribution for downsampling
                counts_rep1 = rng.binomial(n=counts_total.values, p=0.75)
                counts_rep2 = rng.binomial(n=counts_total.values, p=0.5)

                # Convert replicates to DataFrames
                df_rep1 = pd.DataFrame([counts_rep1], columns=exprs_df_lineage.columns)
                df_rep2 = pd.DataFrame([counts_rep2], columns=exprs_df_lineage.columns)

                exprs_df_train = pd.concat([exprs_df_train, df_rep1], ignore_index=True)
                exprs_df_test = pd.concat([exprs_df_test, df_rep2], ignore_index=True)

                # Calculate feature-level statistics for technical replicates
                train_feature_stats[the_leaf] = self._calculate_feature_statistics(
                    df_rep1
                )
                test_feature_stats[the_leaf] = self._calculate_feature_statistics(
                    df_rep2
                )

        exprs_df_train.index, exprs_df_test.index = self.leave_names, self.leave_names

        # Filter out genes with zero variance in the training data
        gene_variance = exprs_df_train.var()
        exprs_df_train = exprs_df_train.loc[:, gene_variance > 0]
        exprs_df_test = exprs_df_test[exprs_df_train.columns]

        # Check for NA values and store the split data
        self._check_for_na_values(exprs_df_train, exprs_df_test)

        self.data_train = exprs_df_train
        self.data_test = exprs_df_test
        self.train_feature_stats = train_feature_stats
        self.test_feature_stats = test_feature_stats

        self._apply_normalization(exprs_df_train, exprs_df_test)

    def _check_for_na_values(self, train_df, test_df):
        """Checks for and warns about any NA values in the training or testing data."""
        if train_df.isnull().values.any():
            logger.warning("NA values found in the exprs_df_avg_train.")
        if test_df.isnull().values.any():
            logger.warning("NA values found in the exprs_df_avg_test.")

    def _apply_normalization(self, train_df, test_df):
        """Stores the processed train and test data and applies PCA if necessary."""

        combined_expr_df = pd.concat([train_df, test_df], axis=0)
        # combined_expr_df = combined_expr_df / combined_expr_df.sum(
        # axis=1
        # ).values.reshape(-1, 1)
        # combined_expr_df *= 10000
        combined_expr_df = np.log1p(combined_expr_df)
        combined_expr_df = combined_expr_df.loc[:, combined_expr_df.var() > 0]

        data_train_processed = combined_expr_df.iloc[: len(train_df)]
        data_test_processed = combined_expr_df.iloc[len(train_df) :]

        self.data_normalized_train, self.mean_train, self.std_train = self._normalize(
            data_train_processed
        )
        self.data_normalized_test, _, _ = self._normalize(
            data_test_processed, self.mean_train, self.std_train
        )
