# %%
import pandas as pd
import numpy as np
import pickle as pk
from ete3 import Tree
from pathlib import Path
from scipy.io import mmread, mmwrite
from tqdm import tqdm
import dendropy

# Set up project paths
project_root = Path('/workspaces/CellTreeBench')
data_dir = project_root / 'data' / 'cbriggsae_mid'
raw_dir = data_dir / 'raw'
p0_dir = data_dir / 'P0'

# Ensure output directories exist
raw_dir.mkdir(parents=True, exist_ok=True)
p0_dir.mkdir(parents=True, exist_ok=True)

print(f"Project root: {project_root}")
print(f"Data directory: {data_dir}")
print(f"Raw data directory: {raw_dir}")
print(f"P0 directory: {p0_dir}")
print()

print("C.briggsae Dataset Preprocessing")
print("="*50)

# %%
# Load the reference lineage tree
print("Loading reference lineage tree...")
tree_file = project_root / "data" / "raw" / "celegans_packer" / "celegans_1.nwk"

if not tree_file.exists():
    print(f"❌ Error: Tree file not found: {tree_file}")
    raise FileNotFoundError(f"Tree file not found: {tree_file}")

ref_lineage_tree = Tree(str(tree_file), format=1)
print(f"✅ Loaded reference tree with {len(ref_lineage_tree.get_leaf_names())} leaves")
print("Reference tree structure (first few levels):")
print(ref_lineage_tree.get_ascii(show_internal=True)[:1000] + "...")

# %%
# Load cell metadata
print("Loading cell metadata...")
metadata_file = project_root / "data" / "raw" / "celegans_chris" / "coldata_df.csv"

if not metadata_file.exists():
    print(f"❌ Metadata file not found: {metadata_file}")
    raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

cell_meta = pd.read_csv(metadata_file, index_col=0, low_memory=False)
print(f"✅ Loaded metadata for {len(cell_meta)} cells")

# Show species distribution
species_counts = cell_meta['species'].value_counts()
print(f"\nSpecies distribution:")
for species, count in species_counts.items():
    print(f"  {species}: {count:,} cells")

# Filter metadata for quality
print("\nFiltering metadata...")
pruned_meta = cell_meta[~cell_meta['lineage_broad'].isna()]
pruned_meta = pruned_meta[pruned_meta['lineage_broad'] != 'unassigned']

print(f"After filtering:")
for species in pruned_meta['species'].unique():
    count = len(pruned_meta[pruned_meta['species'] == species])
    print(f"  {species}: {count:,} cells with lineage annotations")

# %%
# Select C.briggsae species and process lineage patterns
print("Processing C.briggsae data...")
the_species = 'C.briggsae'
species_meta = pruned_meta[pruned_meta['species'] == the_species].copy()
print(f"Selected {len(species_meta)} C.briggsae cells")

# Process lineage patterns
print("Processing lineage patterns...")
meta_lineages = species_meta['lineage_broad'].str.split('/', expand=True)
meta_lineages.index = species_meta.index

# Extract unique lineage patterns
lineage_patterns = meta_lineages.values.flatten()
lineage_patterns = lineage_patterns[~pd.isna(lineage_patterns)]
lineage_patterns = np.unique(lineage_patterns)
print(f"Found {len(lineage_patterns)} unique lineage patterns")

# Show some examples
print(f"Example lineage patterns: {lineage_patterns[:10].tolist()}")

# %%
# Create regex patterns and find matches (FIXED VERSION)
print("Creating regex patterns and finding matches...")

lineages = pd.Series(lineage_patterns)
regex_lineages = lineages.str.replace('x', '[xaplrvd]')

# FIXED: Create matches matrix with proper DataFrame initialization
print("Computing pattern matches...")

# Initialize DataFrame with proper dtype specification
matches = pd.DataFrame(index=lineages, columns=regex_lineages, dtype=bool)

# Fill the matrix using .loc to ensure proper assignment
for i, rexp in enumerate(tqdm(regex_lineages, desc="Processing regex patterns")):
    try:
        # Get the match results as boolean array
        match_results = lineages.str.fullmatch(rexp).values
        
        # Assign the entire column at once using .loc
        matches.loc[:, rexp] = match_results
        
    except Exception as e:
        print(f"Error processing regex {i}: {rexp}")
        print(f"Error: {e}")
        # Fill with False for failed matches
        matches.loc[:, rexp] = False

# Verify the matches matrix is properly filled
print(f"Matches matrix shape: {matches.shape}")
print(f"Matches matrix dtype: {matches.dtypes.iloc[0]}")
print(f"Has NaN values: {matches.isna().any().any()}")

# Check for unmatched patterns
unmatched_count = (matches.sum(axis=1) == 0).sum()
print(f"Unmatched patterns: {unmatched_count}")

# Check for multiple matches
match_counts = matches.sum(axis=0)
multi_match = match_counts[match_counts > 1]
print(f"Patterns with multiple matches: {len(multi_match)}")

if len(multi_match) > 0:
    print("Multiple matches found:", multi_match.index.tolist()[:5])

# %%
# Process lineage to regex mapping
print("\nProcessing lineage to regex mapping...")

# Find rows matching just one regex (themselves)
row_sums = matches.sum(axis=1)
print(f"Row sums statistics:")
print(f"  Min: {row_sums.min()}")
print(f"  Max: {row_sums.max()}")
print(f"  Mean: {row_sums.mean():.2f}")
print(f"  Rows with sum == 1: {(row_sums == 1).sum()}")

good_lineages_mask = (matches.sum(axis=1) == 1)
good_lineages = matches.index[good_lineages_mask]
print(f"Good lineages found: {len(good_lineages)}")

if len(good_lineages) == 0:
    print("No lineages with exactly one match found!")
    print("Using alternative approach: direct pattern-to-regex mapping...")
    
    # Alternative approach: create mapping directly
    metalineage_to_regex = {}
    for pattern in lineages:
        # For patterns without 'x', they should match themselves exactly
        if 'x' not in pattern:
            metalineage_to_regex[pattern] = pattern
        else:
            # For patterns with 'x', use the regex version
            regex_pattern = pattern.replace('x', '[xaplrvd]')
            metalineage_to_regex[pattern] = regex_pattern
    
    print(f"Created {len(metalineage_to_regex)} lineage-to-regex mappings using alternative approach")

else:
    # Original approach for when we have good lineages
    metalineage_to_regex = {}
    for lineage in good_lineages:
        # Find the column that matches for this lineage
        matching_cols = matches.columns[matches.loc[lineage, :]]
        if len(matching_cols) == 1:
            metalineage_to_regex[lineage] = matching_cols[0]
    
    print(f"Created {len(metalineage_to_regex)} lineage-to-regex mappings")

# Apply regex mapping to metadata lineages
meta_regex_lineages = meta_lineages.replace(metalineage_to_regex)

# Join lineage patterns for each cell
meta_regex_lineages_joined = meta_regex_lineages.apply(
    lambda x: '|'.join(x.dropna().astype(str)),
    axis=1
)

# Get unique regex patterns
meta_regex_lineages_joined_set = meta_regex_lineages_joined.unique()
print(f"Found {len(meta_regex_lineages_joined_set)} unique regex pattern groups")

# Show some examples
print(f"Example regex patterns: {meta_regex_lineages_joined_set[:5].tolist()}") 

# %%
# Map cells to tree leaves using dendropy
print("Mapping cells to tree leaves...")

# Load tree with dendropy for processing
tree_file_path = project_root / "data" / "raw" / "celegans_packer" / "celegans_1.nwk"
pruned_tree = dendropy.Tree.get(path=str(tree_file_path), schema='newick')

# Add taxa to internal nodes
for node in pruned_tree.internal_nodes():
    node.taxon = dendropy.Taxon(node.label)

# Get leaf labels
leaf_labels = np.array([t.taxon.label for t in pruned_tree.leaf_nodes()])
leaf_labels = pd.Series(leaf_labels).sort_values()
print(f"Tree has {len(leaf_labels)} leaves")

# Create leaf matching matrix (FIXED VERSION)
print("Creating leaf matching matrix...")

# FIXED: Initialize DataFrame with proper dtype specification
leaf_matches = pd.DataFrame(
    index=leaf_labels, 
    columns=meta_regex_lineages_joined_set, 
    dtype=bool
)

# FIXED: Match regex patterns to tree leaves using .loc assignment
for i, rexp in enumerate(tqdm(meta_regex_lineages_joined_set, desc="Matching patterns to leaves")):
    try:
        # Get match results and assign using .loc
        match_results = leaf_labels.str.fullmatch(str.upper(rexp)).values
        leaf_matches.loc[:, rexp] = match_results
    except Exception as e:
        print(f"Error processing regex {i}: {rexp}")
        print(f"Error: {e}")
        # Fill with False for failed matches
        leaf_matches.loc[:, rexp] = False

print("Initial matching complete")

# %%
# Iterative tree pruning and cell-to-leaf mapping
print("Starting iterative tree pruning and leaf mapping...")

meta_regex_lineages_joined_set_to_leaf = {}
matched_leaves = set()

# Perform iterative matching and pruning
iteration = 1
while True:
    print(f"\nIteration {iteration}:")
    
    # Find matches in current iteration
    new_matches_found = False
    
    for reg in leaf_matches.columns:
        if leaf_matches[reg].any():
            leaf = leaf_matches.index[np.argmax(leaf_matches[reg])]
            if reg not in meta_regex_lineages_joined_set_to_leaf:
                meta_regex_lineages_joined_set_to_leaf[reg] = leaf
                matched_leaves.add(leaf)
                new_matches_found = True
    
    if not new_matches_found:
        print("No new matches found, stopping iteration")
        break
    
    # Remove matched columns and rows
    remaining_cols = [col for col in leaf_matches.columns 
                     if col not in meta_regex_lineages_joined_set_to_leaf.keys()]
    remaining_rows = [idx for idx in leaf_matches.index if idx not in matched_leaves]
    
    if not remaining_cols or not remaining_rows:
        print("No more patterns or leaves to match")
        break
    
    # Prune tree
    to_prune = [leaf for leaf in leaf_matches.index if leaf not in matched_leaves and leaf not in remaining_rows]
    if to_prune:
        pruned_tree.prune_taxa_with_labels(to_prune)
        pruned_tree.update_taxon_namespace()
    
    # Update leaf labels
    current_leaf_labels = np.array([t.taxon.label for t in pruned_tree.leaf_nodes()])
    remaining_leaf_labels = pd.Series(current_leaf_labels).sort_values()
    
    print(f"  Matches found: {sum(1 for col in leaf_matches.columns if col in meta_regex_lineages_joined_set_to_leaf)}")
    print(f"  Remaining leaves: {len(remaining_leaf_labels)}")
    print(f"  Remaining patterns: {len(remaining_cols)}")
    
    if len(remaining_leaf_labels) == 0 or len(remaining_cols) == 0:
        break
    
    # Create new matching matrix for next iteration (FIXED VERSION)
    leaf_matches = pd.DataFrame(
        index=remaining_leaf_labels,
        columns=remaining_cols,
        dtype=bool
    )
    
    # Re-match patterns using .loc assignment
    for rexp in remaining_cols:
        try:
            match_results = remaining_leaf_labels.str.fullmatch(str.upper(rexp)).values
            leaf_matches.loc[:, rexp] = match_results
        except Exception as e:
            print(f"Error in iteration {iteration} with regex {rexp}: {e}")
            leaf_matches.loc[:, rexp] = False
    
    # Check if any matches exist
    if not leaf_matches.any().any():
        print("No more matches possible")
        break
    
    iteration += 1
    if iteration > 10:  # Safety break
        print("Maximum iterations reached")
        break

# Final statistics
final_leaves = len([t.taxon.label for t in pruned_tree.leaf_nodes()])
print(f"\nFinal results:")
print(f"  Total regex patterns mapped: {len(meta_regex_lineages_joined_set_to_leaf)}")
print(f"  Unique leaves mapped to: {len(set(meta_regex_lineages_joined_set_to_leaf.values()))}")
print(f"  Final tree leaves: {final_leaves}")

# %%
# Create cell-to-leaf mapping dataframe
print("Creating cell-to-leaf mapping...")

cell_to_leaf_df = pd.DataFrame(meta_regex_lineages_joined, columns=['regex_joined'])
cell_to_leaf_df.index.name = 'og_idx'

# Filter to only include successfully mapped patterns
cell_to_leaf_df = cell_to_leaf_df.loc[
    cell_to_leaf_df['regex_joined'].isin(meta_regex_lineages_joined_set_to_leaf.keys())
]

# Add leaf mapping
cell_to_leaf_df['leaf'] = cell_to_leaf_df['regex_joined'].replace(meta_regex_lineages_joined_set_to_leaf)

print(f"Cell-to-leaf mapping created:")
print(f"  Total cells mapped: {len(cell_to_leaf_df)}")
print(f"  Unique leaves: {cell_to_leaf_df['leaf'].nunique()}")

# Show some examples
print(f"\nExample mappings:")
print(cell_to_leaf_df.head())

# Save the pruned tree
print(f"\nSaving pruned tree...")
species_name = the_species.replace('.', '_').lower()
tree_output_file = p0_dir / "p0-topology_tree.nwk"
pruned_tree.write(path=str(tree_output_file), schema='newick')
print(f"✅ Saved pruned tree: {tree_output_file}")

# %%
# Load and process expression data
print("Loading expression data...")

# Load expression data from the source
data_path = project_root / "data" / "raw" / "celegans_chris"
expression_file = data_path / "cds_filt_exprs.mm"

if not expression_file.exists():
    print(f"❌ Expression file not found: {expression_file}")
    raise FileNotFoundError(f"Expression file not found: {expression_file}")

expression_adata = mmread(str(expression_file))
print(f"Loaded expression matrix: {expression_adata.shape}")

# Load cell barcodes
barcodes_file = data_path / "cds_filt_cell_barcodes.csv"
barcodes_list = pd.read_csv(barcodes_file, header=None).values.flatten()
print(f"Loaded {len(barcodes_list)} cell barcodes")

# Load gene names  
genes_file = data_path / "cds_filt_gene_names.csv"
all_gene_names = pd.read_csv(genes_file, header=None, names=['id'])
all_gene_names = all_gene_names.id.values
print(f"Loaded {len(all_gene_names)} genes")

# Create expression dataframe
expression_df = pd.DataFrame(expression_adata.toarray())
expression_df.index = barcodes_list
expression_df.columns = all_gene_names

# Filter expression data to mapped cells only
expression_df = expression_df.loc[cell_to_leaf_df.index]
print(f"Filtered expression data: {expression_df.shape}")

# Prepare metadata
filtered_metadata = species_meta.loc[cell_to_leaf_df.index].copy()
filtered_metadata['leaf'] = cell_to_leaf_df['leaf']

print(f"Final dataset statistics:")
print(f"  Species: {the_species}")
print(f"  Cells: {len(filtered_metadata):,}")
print(f"  Genes: {expression_df.shape[1]:,}")
print(f"  Lineages: {filtered_metadata['leaf'].nunique()}")
print(f"  Tree leaves: {len(set(meta_regex_lineages_joined_set_to_leaf.values()))}")

# %%
# Save processed data in standard CellTreeBench format
print("Saving processed data...")

# 1. Save cell-to-leaf mapping
cell_to_leaf_file = p0_dir / "P0-cell_to_leaf_df.csv"
cell_to_leaf_df.to_csv(cell_to_leaf_file)
print(f"✅ Saved cell-to-leaf mapping: {cell_to_leaf_file}")

# 2. Save filtered metadata
metadata_file = raw_dir / f"{species_name}_cell_meta.csv"
filtered_metadata.to_csv(metadata_file)
print(f"✅ Saved metadata: {metadata_file} ({len(filtered_metadata)} cells)")

# 3. Save expression data
expression_file = raw_dir / f"{species_name}_expression_df.csv"
expression_df.to_csv(expression_file)
print(f"✅ Saved expression data: {expression_file} {expression_df.shape}")

print(f"\n{'='*60}")
print("C.BRIGGSAE PREPROCESSING COMPLETE!")
print(f"{'='*60}")
print("Generated files for CBriggsaeDataset:")
print(f"📁 {p0_dir}/")
print(f"   ├── p0-topology_tree.nwk           # Processed lineage tree")
print(f"   └── P0-cell_to_leaf_df.csv         # Cell-to-lineage mapping")
print(f"📁 {raw_dir}/")
print(f"   ├── {species_name}_cell_meta.csv       # Cell metadata with lineage mapping")
print(f"   └── {species_name}_expression_df.csv   # Expression matrix")
print()
print("Dataset statistics:")
print(f"  📊 Species: {the_species}")
print(f"  📊 Cells: {len(filtered_metadata):,}")
print(f"  📊 Genes: {expression_df.shape[1]:,}")
print(f"  📊 Lineages: {filtered_metadata['leaf'].nunique()}")
print(f"  📊 Tree leaves: {len(set(meta_regex_lineages_joined_set_to_leaf.values()))}")
print()
print("Ready for dataset loading with appropriate dataset class!")

# %%
# End of C.briggsae preprocessing workflow
# All core processing steps completed above


