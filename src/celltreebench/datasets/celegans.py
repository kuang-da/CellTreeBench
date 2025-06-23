# src/celltreebench/datasets/celegans.py

from .celegans_dataset_supervised import CElegansDatasetSupervised


def load_celegans_supervised_split(
    dataset_name="celegans_dev",
    lineage_name="P0",
    data_dir=None,
    out_dir=None,
    sampling_method="biological",  # or "technical"
    seed=5,
    subset_leaves=None,
    quartet_sampling_method="random",
):
    """
    Load CElegans dataset with precomputed supervised train/test splits.

    Returns:
        (train_dataset, test_dataset): two CElegansSupervisedSplit instances
    """
    # Step 1: Create initial dataset to generate splits
    dataset_init = CElegansDatasetSupervised(
        data_dir=data_dir,
        out_dir=out_dir,
        dataset_name=dataset_name,
        lineage_name=lineage_name,
        data_split_seed=seed,
        train=True,  # this only determines which part it gets, not critical
        quartet_sampling_method=quartet_sampling_method,
        lineage_node_sampling_strategy=sampling_method,
        subset_tree_with_leaves=subset_leaves,
    )

    # Step 2: Reuse splits
    train_dataset = CElegansDatasetSupervised(
        data_dir=data_dir,
        out_dir=out_dir,
        dataset_name=dataset_name,
        lineage_name=lineage_name,
        supervised_data_dict=dataset_init.supervised_data_dict,
        train=True,
        quartet_sampling_method=quartet_sampling_method,
        lineage_node_sampling_strategy=sampling_method,
        subset_tree_with_leaves=subset_leaves,
    )

    test_dataset = CElegansDatasetSupervised(
        data_dir=data_dir,
        out_dir=out_dir,
        dataset_name=dataset_name,
        lineage_name=lineage_name,
        supervised_data_dict=dataset_init.supervised_data_dict,
        train=False,
        quartet_sampling_method=quartet_sampling_method,
        lineage_node_sampling_strategy=sampling_method,
        subset_tree_with_leaves=subset_leaves,
    )

    return train_dataset, test_dataset
