import logging
import os
from ete3 import Tree
import pandas as pd
import numpy as np

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
    This class creates all the phylogenetic datasets you need. Will read data. Will split data if necessary. Etc.
    """
    def __init__(self, dataset, dataset_names=[""], tree_directory="trees", msa_directory="msas", data_dir=DATA_ROOT, autosplit=None, seed=None):
        """
        Initialize the PhyloDatasetCreator.

        Args:
            dataset (str): Name of master dataset (directory).
            dataset_names (list or dict): Names of the datasets to create. If dict, keys are dataset names and values are proportions (float). If autosplit is disabled the names will be assumed to be directories for each dataset.
            tree_directory (str): Name of the phylogenetic tree directory in the dataset.
            msa_directory (str): Name of the multiple sequence alignment (MSA) directory in the dataset.
            data_dir (str): Directory where the master dataset is stored.
            autosplit (str): Whether to enable automatic splitting of the dataset or to read from seperate files and what method to use. Options are None, "sites", or "leaves".
        """
        self.tree_directory = tree_directory
        self.msa_directory = msa_directory
        self.data_dir = data_dir
        self.dataset = dataset
        self.created_datasets = {}

        if autosplit:

            if type(dataset_names) == list:
                dataset_proportions = {name: 1/len(dataset_names) for name in dataset_names} # if no dataset split proportions are provided datasets are split equally
            elif type(dataset_names) == dict:
                total = sum(dataset_names.values())
                dataset_proportions = {name: prop/total for name, prop in dataset_names.items()} # normalize dataset split proportions
            else:   
                raise ValueError("dataset_names must be a list or dict")
            
            trees = self._read_all_trees()
            msas = self._read_all_msas()

            if autosplit == "sites":
                dataset_msas = self._split_data_site(msas, dataset_proportions, rng=np.random.default_rng(seed)) # splits data by sites
                dataset_trees = {dataset_name: trees for dataset_name in dataset_proportions.keys()} # trees are not split when splitting by sites

            elif autosplit == "leaves":
                dataset_trees, dataset_msas = self._split_data_leaves(trees, msas, dataset_proportions, rng=np.random.default_rng(seed)) # splits data by leaves
        
        # split proportions are no longer necessary. Dataset names will now be a list of just the dataset names
        if type(dataset_names) == dict:
            dataset_names = list(dataset_names.keys())
        elif type(dataset_names) != list:
            raise ValueError("dataset_names must be a list or dict")
        
        # basic reading of seperate datasets if no autosplit
        if not autosplit: 
            dataset_msas = {} # {dataset name: [msa, msa, ...]}
            dataset_trees = {}# {dataset name: [tree, tree, ...]}
            for dataset in dataset_names:
                dataset_trees[dataset] = self._read_all_trees(dataset=dataset)
                dataset_msas[dataset] = self._read_all_msas(dataset=dataset)

        # create the actual dataset objects
        self.created_datasets = self._create_datasets(dataset_msas, dataset_trees) # {dataset name: PhyloDataset dataset object}




    def _create_datasets(self, dataset_msas, dataset_trees):
        """
        Create PhyloDataset objects from the given MSAs and trees.

        Args:
            dataset_msas (dict): Dictionary of dataset MSAs.
            dataset_trees (dict): Dictionary of dataset trees.

        Returns:
            dict: Dictionary of created PhyloDataset objects. {name (str): dataset object (PhyloDataset)}
        """
        created_datasets = {}
        max_length = max(df.shape[1] for msas in dataset_msas.values() for df in msas) # ensures all MSAs have the same length (between and inside datasets). Will zero-pad to max_length if not.
        
        for dataset in dataset_msas.keys():
            created_datasets[dataset] = PhyloDataset(
                dataset_msas[dataset],
                dataset_trees[dataset],
                name=dataset,
                max_length=max_length
                )
        return created_datasets
            
    def _split_data_leaves(self, trees, msas, props, rng):
        """
        Split the data by leaves into sets based on the given proportions.

        Args:
            trees (list): List of ETE3 Tree objects.
            msas (list): List of pd DataFrames containing the MSA data.
            props (dict): Dictionary containing the proportions for each dataset.
            rng (np.random.Generator): Random number generator.

        Returns:
            Tuple: Two dictionaries containing the split MSAs and trees for each dataset.
        """
        split_msas = {dataset_name: [] for dataset_name in props.keys()} # {dataset name: [msa, msa, ...]}
        split_trees = {dataset_name: [] for dataset_name in props.keys()} # {dataset name: [tree, tree, ...]}
        for tree, msa in zip(trees, msas): # iterate over tree and msa. (each dataset gets its proportion of each tree/msa)
            leaves = tree.get_leaf_names()
            idx = rng.permutation(leaves)
            prev_point = 0
            point = 0
            for dataset, prop in props.items():
                point += round(len(idx) * prop) # calculate the endpoint for the current dataset based on permuted index and proportion
                idx_dataset = idx[prev_point:point] # get selected leaf names
                split_msas[dataset].append(msa.loc[idx_dataset]) # msa with only the selected leaves
                split_trees[dataset].append(self._create_sub_tree(tree, leaves=idx_dataset)) # generate tree with only the selected leaves
                prev_point = point

        return split_trees, split_msas
    


    def _split_data_site(self, msas, props, rng):
        """
        Split the data by sites into sets based on the given proportions.

        Args:
            msas (list): List of pd DataFrames containing the MSA data.
            props (dict): Dictionary containing the proportions for each dataset.
            rng (np.random.Generator): Random number generator.

        Returns:
            Dictionary: containing the split MSAs each dataset.
        """
        split_msas = {dataset_name: [] for dataset_name in props.keys()} # {dataset name: [msa, msa, ...]}
        for msa in msas: # iterate over msa. (each dataset gets its proportion of each msa)
            num_sites = msa.shape[1]//22 # each site has 22 columns (20 amino acids + unknown + gap)
            idx = rng.permutation(num_sites) # permute site indices and convert to column indices
            prev_point = 0
            point = 0
            for dataset, prop in props.items(): 
                point += round(num_sites * prop) # calculate the endpoint for the current dataset based on permuted index and proportion
                idx_dataset = []

                for i in idx[prev_point:point]:
                    idx_dataset += range(22*i, 22*(i+1)) # get selected site indices

                split_msas[dataset].append(msa.iloc[:, idx_dataset]) # msa with only the selected sites
                prev_point = point

        return split_msas

    def get_dataset(self, datasets=None):
        """
        Get a/multiple phylogenetic dataset(s) that were created.

        Args:
            datasets (list or str): Name(s) of the datasets to get (str). If None, all datasets will be returned.
        
        Returns:
            dict or PhyloDataset: Dictionary of created dataset objects. If PhyloDataset, then only object is returned
        """
        if datasets is None: # defaults to returning all datasets
            return self.created_datasets
        elif type(datasets) == str:
            assert datasets in self.created_datasets, f"Dataset '{datasets}' not found" # ensure datasets are created/exist
            return self.created_datasets[datasets]
        elif type(datasets) == list:
            datasets_got = {}
            for dataset in datasets:
                assert dataset in self.created_datasets, f"Dataset '{dataset}' not found" # ensure datasets are created/exist
                datasets_got[dataset] = self.created_datasets[dataset]
            return datasets_got
        else:
            raise TypeError("datasets must be a list, str, or None")
        

    def _read_phylogenetic_tree(self, path, filename):
        """Read phylogenetic tree
    
        Args:
            path (str): Path to the directory containing the tree file.
            filename (str): Name of the tree file.

        Returns:
            Tree: ETE3 Tree object representing the phylogenetic tree.
        """
        tree = Tree(os.path.join(path, filename), name=filename)
        return tree
    
    def _read_all_trees(self, dataset=""):
        """Read all phylogenetic trees in the dataset directory.
        
        Args:
            dataset (str): Name of the dataset to read trees from.

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
    
    def _read_phylo_msa(self, path, filename, alphabet="ARNDCQEGHILKMFPSTWYVX-"):
        """
        Read phylogenetic MSA and parse into pd dataframe with one-hot encoding
        
        
        Args:
            path (str): Path to the directory containing the MSA file.
            filename (str): Name of the MSA file.
            alphabet (bytes): Alphabet used in the MSA sequences. Default is "ARNDCQEGHILKMFPSTWYVX-" (20 standard amino acids + unknown + gap).
        
        Returns:
            pd.DataFrame: DataFrame containing the MSA data (one-hot encoded).
                Rows correspond to taxons, columns correspond to amino acid positions.
                    Column format is "[site number]_[protein letter]" so total # of columns is # of sites * # of letters

        """
        msa_file = os.path.join(path, filename)
        sequences, ids = [], []
        lookup = {char: index for index, char in enumerate(alphabet)}
        with open(msa_file, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"): # get leaf name/id
                    ids.append(line[1:])
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
            dataset (str): Name of the dataset to read MSAs from.

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
        """
        Creates a new Ete3 tree with only the selected leaves

        Args:
            tree (Tree): The original ETE3 tree.
            leaves (list): List of leaf names to keep in the sub-tree.

        Returns:
            Tree: A new ETE3 tree containing only the selected leaves.
        """
        tree = tree.copy()
        if leaves is not None:
            tree.prune(leaves.tolist(), preserve_branch_length=True)
        return tree