# UMAP Dimension Reduction

Tan_Gen_data_choose.py: This script randomly selects structures with guidance = 1 and guidance = 5, computes the distance map for each selected structure, and prepares them as input for UMAP dimensionality reduction.

umap_dimension_reduction.py: This script implements the UMAP training process using the distance maps from all structures as input.

transform_new_data.py: This script applies the trained UMAP dimensionality reduction model to transform new distance maps from the test dataset.

# UMAP Implementation Guide

We demonstrate the implementation of UMAP using the above scripts.

## Step 1: Choose the Training Dataset
To select the training dataset, run the following command:

```
python Tan_Gen_data_choose.py 0.5 800
```

Here, 0.5 specifies the fraction of structures with guidance = 1, and 800 is the total number of generated structures selected for a specific region.

## Step 2: Train the UMAP Dimensionality Reduction Model
To train the UMAP model, use the command:

```
python umap_dimension_reduction.py
```

## Step 3: Transform New Data
To transform new distance maps from the test dataset using the trained UMAP model, run:

```
python transform_new_data.py
```
