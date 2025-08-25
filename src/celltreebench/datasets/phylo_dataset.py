import logging
import os
from torch.utils.data import Dataset
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

class PhyloDataset(Dataset):
    """
    Class for phylogenetic datasets.
    """
    def __init__(self, msas, trees, max_length=0):
        """
        Initialize the PhyloDataset.

        Args:
            msas (list): List of DataFrames containing the MSA data.
            trees (list): List of ETE3 Tree objects representing the phylogenetic trees.
        """
        self.data = msas
        self.data_normalized = self._zero_pad(self.data, max_length) # skipping normalization (other than zero padding). I don't think it makes sense with one-hot encoding
        self.topology_trees = trees
        self.create_ref_dm()

    def get_proportions(self):
        """
        Returns the proportions of each tree (based on number of leaves) in the dataset.

        """
        return [comb(len(tree.get_leaves()), 4) for tree in self.topology_trees]
    
    def __len__(self):
        return sum(self.get_proportions())
    
    def _zero_pad(self, data, max_length):
        """
        Zero pad the data to ensure all MSAs have the same length.
        
        Args:
            data (list): List of DataFrames containing the MSA data.
        
        Returns:
            list: List of DataFrames with zero padding applied.
        """
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

    def create_ref_dm(self):
        self.ref_dm = []  # List of reference distance matrices for each topology tree
        for i, tree in enumerate(self.topology_trees):
            leave_names = self.data[i].index
            self.ref_dm.append(get_path_distance_matrix(tree, leave_names))