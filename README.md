# Deep Learning for Predicting Chronic Wasting Disease (CWD) Spread in U.S. Deer Populations

**University of Virginia: DS 6050 Deep Learning**  
**Team:** Alexander Owens, Samuel Delaney, Tyler Kellogg  
**Instructor:** Heman Shakeri, PhD  
**Semester:** Fall 2025  

---

## Project Overview
This project develops a reproducible machine learning and deep learning framework for forecasting next-year Chronic Wasting Disease (CWD) emergence at the county level across the United States. Using publicly available USGS surveillance data and environmental context, we compare logistic regression, random forest, and a small multilayer perceptron (MLP).

The workflow mirrors the constraints of real-world wildlife surveillance and uses a strict time-based train/validation/test split to avoid information leakage across years.

---

## Dataset
**Source:**  
USGS CWD Distribution ver. 3.0 (June 2025):  
https://www.usgs.gov/data/chronic-wasting-disease-distribution-united-states-state-and-county-ver-30-june-2025

**Features included:**
- Historical detection summaries  
- Spatial neighbor exposure indicators  
- Environmental variables (land cover fractions, temperature, elevation)  
- Human population density and related context variables  

**Target:**  
A binary indicator denoting whether a county reports a new CWD detection in the following year.

Feature engineering and preprocessing steps are handled through dedicated scripts in the `code/` directory.

---

## Reproducibility

This project is designed so that any user can reproduce the full modeling pipeline from raw data processing through model evaluation. All scripts, configuration files, and utilities are contained in the public repository.

### Environment Setup
Install dependencies listed in `requirements.txt`:

```
pip install -r requirements.txt
```

Random seeds are fixed where applicable to ensure deterministic output.

---

## Data Loading and Feature Engineering
Raw USGS county-year detections and context variables are loaded using:

`01_build_features.py`

This script constructs all engineered predictors described in the paper:
- Historical detection indicators  
- Spatial neighbor summaries  
- Environmental and demographic attributes  
- Standardized numeric feature matrix with imputation  

The preprocessing step ensures reproducibility and consistent feature matrices across machines.

---

## Time-Based Train/Validation/Test Split
To reflect operational forecasting workflows, we follow the strict chronological split described in the paper:

- **Training:** all years prior to the designated validation year  
- **Validation:** the next chronological year  
- **Test:** the final available year  

This structure prevents future information from leaking into past predictions and mirrors how wildlife agencies make prospective decisions.

---

## Model Training
Three supervised models are implemented in modular training scripts:

- `02_train_logreg.py`  
- `03_train_rf.py`  
- `04_train_mlp.py`  

Each model is trained only on the designated training years.  
The MLP follows a standard tabular deep learning design with ReLU activations and a sigmoid output layer.

---

## Evaluation Framework
The evaluation pipeline is handled by:

- `05_eval_models.py`  
- `09_visuals.py`  

Metrics and diagnostics include:
- ROC curves  
- Precision–Recall curves  
- Average Precision (PR-AUC)  
- Confusion matrices  
- Calibration curves  
- F1-optimized decision thresholds (selected on validation data)  
- Training and validation loss curves for the MLP  

These outputs match the figures and evaluation procedures described in the paper and shown in the appendices.

---

## Ablation Studies
Feature-block ablations are implemented in:

`06_ablations.py`

These experiments remove specific engineered feature classes (e.g., adjacency summaries) to evaluate their contribution to predictive performance.

---

## End-to-End Execution
Run the complete pipeline for preprocessing, time-based splitting, training, evaluation, and figure generation with:

```bash
python3 code/08_main.py
```

This mirrors the pipeline diagram described in the paper and supports clean, direct reproducibility.

---

## Extensibility
The repository is fully modular. New feature blocks, additional model architectures (e.g., graph neural networks), or alternative decision-analytic thresholding procedures can be added without modifying the surrounding pipeline.

## Repository Structure

The repository is organized so that each stage of the pipeline is modular and aligned with the workflow described in the project paper.

* **code/**
    * `01_build_features.py`: Data loading, cleaning, feature engineering
    * `02_train_logreg.py`: Logistic regression training + preprocessing
    * `03_train_rf.py`: Random forest training + prediction
    * `04_train_mlp.py`: Multilayer perceptron (PyTorch) training
    * `05_eval_models.py`: Evaluation metrics, PR-AUC, ROC, F1 thresholds
    * `06_ablations.py`: Feature-block ablation experiments
    * `07_utils.py`: Random seed logic, shared utilities
    * `08_main.py`: End-to-end pipeline execution script
    * `09_visuals.py`: Loss curves, ROC, PR, calibration plots
* **data/**: Raw and processed data (not tracked in GitHub)
* `config.yaml`: File paths and runtime configuration
* `requirements.txt`: Python dependencies for reproducibility