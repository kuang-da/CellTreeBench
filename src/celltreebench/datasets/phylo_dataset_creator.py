import logging
import os
from ete3 import Tree
from Bio import SeqIO
import pandas as pd
from math import comb
import torch
import numpy as np

from celltreebench.utils.tree_operations import get_path_distance_matrix
from celltreebench.utils.reconstruction_eval import compare_trees
from celltreebench.datasets.phylo_dataset import PhyloDataset



# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the minimum logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],  # This will output logs to the console
)

logger = logging.getLogger(__name__)

# Find the project root
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")

class PhyloDatasetCreator():
    """
    Class for phylogenetic datasets.
    """
    def __init__(self, dataset, dataset_names=["default"], tree_directory="trees", msa_directory="msas", data_dir=DATA_ROOT, autosplit=False, seed=None):
        """
        Initialize the PhyloDataset.

        Args:
            datset (str): Name of master dataset.
            datasets (list or dict): Names of the datasets to create. If dict, keys are dataset names and values are proportions (float).
            tree_name (str): Name of the phylogenetic tree file in the dataset.
            data_dir (str): Directory where the dataset is stored.
            autosplit (bool): Whether to enable automatic splitting of the dataset or to read from seperate files.
        """
        self.tree_directory = tree_directory
        self.msa_directory = msa_directory
        self.data_dir = data_dir
        self.dataset = dataset
        self.created_datasets = {}
        if autosplit:

            if type(dataset_names) == list:
                dataset_proportions = {name: 1/len(dataset_names) for name in dataset_names}
            elif type(dataset_names) == dict:
                total = sum(dataset_names.values())
                dataset_proportions = {name: prop/total for name, prop in dataset_names.items()}
            else:   
                raise ValueError("dataset_names must be a list or dict")
            trees = self._read_all_trees()
            msas = self._read_all_msas()
            dataset_trees, dataset_msas = self._split_data(trees, msas, dataset_proportions, rng=np.random.default_rng(seed))
        
        if type(dataset_names) == dict:
            dataset_names = list(dataset_names.keys())
        elif type(dataset_names) != list:
            raise ValueError("dataset_names must be a list or dict")
        
        if not autosplit:
            dataset_msas = {} # {dataset name: [msa, msa, ...]}
            dataset_trees = {}# {dataset name: [tree, tree, ...]}
            for dataset in dataset_names:
                dataset_trees[dataset] = self._read_all_trees(dataset=dataset)
                dataset_msas[dataset] = self._read_all_msas(dataset=dataset)

        self.created_datasets = self._create_datasets(dataset_msas, dataset_trees) # {dataset name: dataset object}




    def _create_datasets(self, dataset_msas, dataset_trees):
        """
        Create PhyloDataset objects from the given MSA and tree data.

        Args:
            dataset_msas (dict): Dictionary of dataset MSAs.
            dataset_trees (dict): Dictionary of dataset trees.

        Returns:
            dict: Dictionary of created PhyloDataset objects.
        """
        created_datasets = {}
        max_length = max(df.shape[1] for msas in dataset_msas.values() for df in msas)
        for dataset in dataset_msas.keys():
            created_datasets[dataset] = PhyloDataset(
                dataset_msas[dataset],
                dataset_trees[dataset],
                max_length
                )
        return created_datasets
            
    def _split_data(self, trees, msas, props, rng):
        split_msas = {dataset_name: [] for dataset_name in props.keys()} # {dataset name: [msa, msa, ...]}
        split_trees = {dataset_name: [] for dataset_name in props.keys()} # {dataset name: [tree, tree, ...]}
        for tree, msa in zip(trees, msas):
            leaves = tree.get_leaf_names()
            idx = rng.permutation(leaves)
            prev_point = 0
            point = 0
            for dataset, prop in props.items():
                point += round(len(idx) * prop)
                idx_dataset = idx[prev_point:point]
                split_msas[dataset].append(msa.loc[idx_dataset])
                split_trees[dataset].append(self._create_sub_tree(tree, leaves=idx_dataset))
                prev_point = point

        return split_trees, split_msas

    def get_dataset(self, datasets=None):
        """
        Create the phylogenetic dataset.

        Args:
            datasets (list or str): Name(s) of the datasets to create (str). If None, all datasets will be created.
        
        Returns:
            dict or object: Dictionary of created dataset objects.
        """
        if datasets is None:
            return self.created_datasets
        elif type(datasets) == str:
            assert datasets in self.created_datasets, f"Dataset '{datasets}' not found"
            return self.created_datasets[datasets]
        elif type(datasets) == list:
            datasets_got = {}
            for dataset in datasets:
                assert dataset in self.created_datasets, f"Dataset '{dataset}' not found"
                datasets_got[dataset] = self.created_datasets[dataset]

            return datasets_got
        else:
            raise ValueError("datasets must be a list, str, or None")
        

    def _read_phylogenetic_tree(self, path, filename):
        """Read phylogenetic tree
    
        Args:
            tree_name (str): Name of the phylogenetic tree file in the dataset.
        
        Returns:
            Tree: ETE3 Tree object representing the phylogenetic tree.
        """
        tree = Tree(os.path.join(path, filename), name=filename)
        return tree
    
    def _read_all_trees(self, dataset=""):
        """Read all phylogenetic trees in the dataset directory.
        
        Args:
            tree_directory (str): Directory containing the phylogenetic tree files.
        
        Returns:
            list: List of ETE3 Tree objects representing the phylogenetic trees.
        """
        trees = []
        path = os.path.join(self.data_dir, self.dataset, dataset, self.tree_directory)
        for filename in os.listdir(path):
            if filename.endswith(".nwk"):
                tree = self._read_phylogenetic_tree(path, filename)
                trees.append(tree)
        return trees
    
    def _read_phylo_msa(self, path, filename, alphabet=b"ARNDCQEGHILKMFPSTWYVX-"):
        """Read phylogenetic MSA
        Args:
            msa_name (str): Name of the MSA file in the dataset.
        Returns:
            pd.DataFrame: DataFrame containing the MSA data (one-hot encoded)
        """
        msa_file = os.path.join(path, filename)
        sequences, ids = [], []
        lookup = {char: index for index, char in enumerate(alphabet)}
        with open(msa_file, "rb") as f:
            for line in f:
                line = line.strip()
                if line.startswith(b">"):
                    ids.append(line[1:].decode("utf8"))
                    sequences.append([])
                else:
                    for char in line:# iterate over each character in the sequence and convert to one-hot
                        i = lookup[char]
                        one_hot = [0] * len(alphabet)
                        one_hot[i] = 1
                        sequences[-1] += one_hot
        # Convert to DataFrame
        columns=[f"{site}_{letter}" for site in range(len(sequences[0]) // len(alphabet)) for letter in alphabet] 
        # column format is "[site number]_[protein letter]" so total # of columns is # of sites * # of letters
        return pd.DataFrame(sequences, index=ids, columns=columns)
    
    def _read_all_msas(self, dataset=""):
        """Read all phylogenetic MSAs in the dataset directory.
        
        Args:
            msa_directory (str): Directory containing the MSA files.
        
        Returns:
            list: List of DataFrames containing the MSA data (one-hot encoded).
        """
        msas = []
        path = os.path.join(self.data_dir, self.dataset, dataset, self.msa_directory)
        for filename in os.listdir(path):
            if filename.endswith(".fa"):
                msa = self._read_phylo_msa(path, filename)
                msas.append(msa)
        return msas
    
    def _create_sub_tree(self, tree, leaves=None):
        tree = tree.copy()
        if leaves is not None:
            tree.prune(leaves.tolist())
        return tree

def load_phylo_supervised_split(dataset_name=None, data_dir=DATA_ROOT, out_dir=None):
    """Load phylogenetic dataset with precomputed train/test splits. Very simple method that does not split. It only loads separate datasets."""
    
    train_dataset= PhyloDataset(
        dataset_name=os.path.join(dataset_name, "train"),
        tree_directory="trees",
        msa_directory="msas",
        data_dir=data_dir
    )
    test_dataset = PhyloDataset(
        dataset_name=os.path.join(dataset_name, "test"),
        tree_directory="trees",
        msa_directory="msas",
        data_dir=data_dir
        )
    max_length = max(df.shape[1] for df in train_dataset.data + test_dataset.data)
    train_dataset.create_data_normalized(max_length)
    test_dataset.create_data_normalized(max_length)
    train_dataset.create_ref_dm()
    test_dataset.create_ref_dm()

    return train_dataset, test_dataset