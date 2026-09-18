# Notebooks

This folder is reserved for exploratory data analysis (EDA), prototype experiments, and model performance inspection before batch execution.

## Recommended Exploration Workflow
1. `01_eda_and_distribution.ipynb`: Inspect CT image dimensions, slice intensity distributions, and class imbalance.
2. `02_model_architecture_inspection.ipynb`: Dry-run forward pass through CNN backbone -> Sequence Reshape -> BiLSTM -> Attention Layer -> Classifier.
3. `03_gradcam_visualizations.ipynb`: Interactive visualization of heatmaps over tumor regions.

All production training, evaluation, and inference code is maintained modularly in the `src/` directory.
