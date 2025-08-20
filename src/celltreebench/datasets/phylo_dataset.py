import logging
import os
from torch.utils.data import Dataset
from ete3 import Tree
from Bio import SeqIO
import pandas as pd
from math import comb

from celltreebench.utils.tree_operations import get_path_distance_matrix
from celltreebench.utils.reconstruction_eval import compare_trees



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

class PhyloDataset(Dataset):
    """
    Class for phylogenetic datasets.
    """
    def __init__(self, dataset_name, tree_directory="trees", msa_directory="msas", data_dir=DATA_ROOT):
        """
        Initialize the PhyloDataset.

        Args:
            dataset_name (str): Name of the dataset.
            tree_name (str): Name of the phylogenetic tree file in the dataset.
            data_dir (str): Directory where the dataset is stored.        
        """
        super().__init__()
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.topology_trees = self._read_all_trees(tree_directory)
        self.data = self._read_all_msas(msa_directory)

        self.leave_names = [leaf.name for leaf in self.topology_tree.iter_leaves()]
        self.n_leaves = len(self.leave_names)
        self.ref_dm = [get_path_distance_matrix(tree, self.leave_names) for tree in self.topology_trees]

        
        self.total_quartets = comb(self.n_leaves, 4)
        self.data_normalized = self._zero_pad(self.data) # skipping normalization (other than zero padding). I don't think it makes sense with one-hot encoding

    def _read_phylogenetic_tree(self, filename, tree_directory):
        """Read phylogenetic tree
    
        Args:
            tree_name (str): Name of the phylogenetic tree file in the dataset.
        
        Returns:
            Tree: ETE3 Tree object representing the phylogenetic tree.
        """
        tree = Tree(os.path.join(self.data_dir, self.dataset_name, tree_directory, filename))
        return tree
    
    def _read_all_trees(self, tree_directory):
        """Read all phylogenetic trees in the dataset directory.
        
        Args:
            tree_directory (str): Directory containing the phylogenetic tree files.
        
        Returns:
            list: List of ETE3 Tree objects representing the phylogenetic trees.
        """
        trees = []
        for filename in os.listdir(os.path.join(self.data_dir, self.dataset_name, tree_directory)):
            if filename.endswith(".nwk"):
                tree = self._read_phylogenetic_tree(filename, tree_directory)
                trees.append(tree)
        return trees
    
    def _read_phylo_msa(self, filename, msa_directory, alphabet=b"ARNDCQEGHILKMFPSTWYVX-"):
        """Read phylogenetic MSA
        Args:
            msa_name (str): Name of the MSA file in the dataset.
        Returns:
            pd.DataFrame: DataFrame containing the MSA data (one-hot encoded)
        """
        msa_file = os.path.join(self.data_dir, self.dataset_name, msa_directory, filename)
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
    
    def _read_all_msas(self, msa_directory):
        """Read all phylogenetic MSAs in the dataset directory.
        
        Args:
            msa_directory (str): Directory containing the MSA files.
        
        Returns:
            list: List of DataFrames containing the MSA data (one-hot encoded).
        """
        msas = []
        for filename in os.listdir(os.path.join(self.data_dir, self.dataset_name, msa_directory)):
            if filename.endswith(".fa"):
                msa = self._read_phylo_msa(filename, msa_directory)
                msas.append(msa)
        return msas
    
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
    
    def get_proportions(self):
        """
        Returns the proportions of each tree (based on number of leaves) in the dataset.

        """
        return [comb(4, len(tree.get_leaves())) for tree in self.topology_trees]
    
    def __len__(self):
        return sum(self.get_proportions)
    
    def _zero_pad(self, data):
        """
        Zero pad the data to ensure all MSAs have the same length.
        
        Args:
            data (list): List of DataFrames containing the MSA data.
        
        Returns:
            list: List of DataFrames with zero padding applied.
        """
        max_length = max(df.shape[1] for df in data)
        padded_data = []
        for df in data:
            if df.shape[1] < max_length:
                padding = pd.DataFrame(0, index=df.index, columns=[f"pad_{i}" for i in range(max_length - df.shape[1])])
                df = pd.concat([df, padding], axis=1)
            padded_data.append(df)
        return padded_data
    
    def get_node_mtx(self):
        """
        Returns a dictionary containing the node matrix and node names.

        Returns:
            dict: A dictionary with 'node_mtx' as a NumPy array of the normalized data
            and 'node_names' as the corresponding index.
        """
        return [{
            "node_mtx": data.to_numpy(),
            "node_names": data.index
        } for data in self.data_normalized]
    
    def compare_trees(self, tree1, i, ref_tree="topology_tree", unrooted=True):
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
            tree2 = self.topology_trees[i]
        else:
            raise ValueError("Unknown reference tree specified.")

        return compare_trees(tree1, tree2, unrooted_trees=unrooted)
                                                                                                                                                                                                            
def load_phylo_supervised_split(dataset_name=None, data_dir=DATA_ROOT, out_dir=None):
    """Load phylogenetic dataset with precomputed train/test splits. Very simple method that does not split. It only loads separate datasets."""
    train_dataset= PhyloDataset(
        dataset_name=dataset_name,
        tree_directory="trees",
        msa_directory="msas",
        data_dir=os.path.join(data_dir, "train")
    )
    test_dataset = PhyloDataset(
        dataset_name=dataset_name,
        tree_directory="trees",
        msa_directory="msas",
        data_dir=os.path.join(data_dir, "test")
        )
    return train_dataset, test_dataset