# %%
from celltreebench.datasets.celegans import load_celegans_supervised_split

# %%
dataset_name = "celegans_small"
lineage_name = "P0"
base_dir = "/workspaces/1-phydist/main/CellTreeBench"
data_dir = f"{base_dir}/data"
out_dir = (
    f"{base_dir}/examples/out/supervised_split_example_{dataset_name}_{lineage_name}"
)

# Load train and test splits with pre-defined supervised strategy
train_dataset, test_dataset = load_celegans_supervised_split(
    dataset_name=dataset_name,
    lineage_name=lineage_name,
    data_dir=data_dir,
    out_dir=out_dir,
    sampling_method="biological",
    seed=42,
)

print(f"Train shape: {train_dataset.data_normalized.shape}")
print(f"Test shape: {test_dataset.data_normalized.shape}")
print(f"Number of leaves: {train_dataset.n_leaves}")


# %%
dataset_name = "celegans_mid"
lineage_name = "P0"
base_dir = "/workspaces/1-phydist/main/CellTreeBench"
data_dir = f"{base_dir}/data"
out_dir = (
    f"{base_dir}/examples/out/supervised_split_example_{dataset_name}_{lineage_name}"
)

# Load train and test splits with pre-defined supervised strategy
train_dataset, test_dataset = load_celegans_supervised_split(
    dataset_name=dataset_name,
    lineage_name=lineage_name,
    data_dir=data_dir,
    out_dir=out_dir,
    sampling_method="biological",
    seed=42,
)

print(f"Train shape: {train_dataset.data_normalized.shape}")
print(f"Test shape: {test_dataset.data_normalized.shape}")
print(f"Number of leaves: {train_dataset.n_leaves}")


# %%
dataset_name = "celegans_large"
lineage_name = "P0"
base_dir = "/workspaces/1-phydist/main/CellTreeBench"
data_dir = f"{base_dir}/data"
out_dir = (
    f"{base_dir}/examples/out/supervised_split_example_{dataset_name}_{lineage_name}"
)

# Load train and test splits with pre-defined supervised strategy
train_dataset, test_dataset = load_celegans_supervised_split(
    dataset_name=dataset_name,
    lineage_name=lineage_name,
    data_dir=data_dir,
    out_dir=out_dir,
    sampling_method="biological",
    seed=42,
)

print(f"Train shape: {train_dataset.data_normalized.shape}")
print(f"Test shape: {test_dataset.data_normalized.shape}")
print(f"Number of leaves: {train_dataset.n_leaves}")
