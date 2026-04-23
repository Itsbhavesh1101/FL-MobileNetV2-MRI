# Federated Learning with MobileNetV2 for MRI Brain Tumor Classification

This repository contains the implementation of a Federated Learning framework utilizing the MobileNetV2 architecture for the classification of brain tumors from MRI scans. The system trains local models across 5 simulated clients and aggregates them using a global meta-learner (ensemble approach) over 20 communication rounds.

## Dataset

The dataset used for training and evaluating this model is the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) available on Kaggle.

## Architecture

The project leverages **MobileNetV2** as the base architecture for feature extraction and classification. The federated learning cycle involves decentralized training where each client updates the model locally, and the central server aggregates these updates to refine the global model.

### Federated Learning Cycle
![FL Cycle](assets/FL%20Cycle.png)

### Model Architecture
![Model Architecture](assets/Model%20Architecture.png)

## Performance Metrics

The final global + ensemble local meta-learners achieved an outstanding test accuracy of **99.57%**.

### Classification Report (Final Model)
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Glioma | 0.9938 | 0.9938 | 0.9938 | 162 |
| Meningioma | 0.9939 | 1.0000 | 0.9969 | 165 |
| No Tumor | 1.0000 | 0.9900 | 0.9949 | 200 |
| Pituitary | 0.9943 | 1.0000 | 0.9971 | 176 |
| **Accuracy** | | | **0.9957** | **703** |
| **Macro Avg**| 0.9955 | 0.9959 | 0.9957 | 703 |
| **Weighted Avg**| 0.9957 | 0.9957 | 0.9957 | 703 |

### Federated Training Process
- **Clients:** 5
- **Rounds:** 20
- **Final Round (20) Validation Accuracy:** 98.86%
- **Final Round (20) Macro Avg F1-score:** 98.84%

Detailed round-by-round metrics can be found in `results/metrics.txt`.

## Repository Structure

```
FL-MobileNetV2-MRI/
├── assets/
│   ├── FL Cycle.png                  # Federated Learning Cycle Architecture
│   ├── Model Architecture.png        # MobileNetV2 Model Architecture
│   └── IMG-*.jpg                     # Various outputs and plots
├── results/
│   └── metrics.txt                   # Detailed round-by-round training metrics
├── mobilenetv2-bt-classification.ipynb # Main Jupyter Notebook with training code
└── README.md                         # This file
```

## Getting Started

1. Clone the repository.
2. Install the necessary dependencies (e.g., standard ML libraries like TensorFlow/PyTorch, numpy, pandas, scikit-learn).
3. Run the Jupyter Notebook `mobilenetv2-bt-classification.ipynb` to execute the federated training cycle.