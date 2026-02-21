import os

# --------------------------------------------------
# CONFIGURATION & CONSTANTS
# --------------------------------------------------

# File Paths
MODEL_PATHS = {
    "VGG19": r"C:\Programs\Brain_stroke_app\vgg19_best_model.keras",
    "ResNet50": r"C:\Programs\Brain_stroke_app\resnet_best_model.keras",
    "EfficientNetB0": r"C:\Programs\Brain_stroke_app\efficientnet_best_model.keras",
    "Autoencoder" : r"C:\Programs\Brain_stroke_app\brain_stroke_autoencoder.keras"
}

# Target Layers for XAI (Backend use only)
MODEL_LAYERS = {
    "VGG19": "block5_conv4",
    "ResNet50": "conv5_block3_out",
    "EfficientNetB0": "top_activation" 
}

# Assets for Visualizations
CONFUSION_MATRICES = {
    "VGG19": "assets/vgg19.png",
    "ResNet50": "assets/resnet50.png",
    "EfficientNetB0": "assets/effiB0.png"
}

IMG_SIZE = (224, 224)

# --------------------------------------------------
# MODEL CARD DATA (Sanitized & Standardized)
# --------------------------------------------------
MODEL_META = {
    "VGG19": {
        "type": "Convolutional Neural Network (CNN)",
        "architecture": "VGG19 (Visual Geometry Group, 19 Layers)",
        "dataset_info": "Trained on the 'Brain Stroke MRI' dataset (Kaggle), containing ~2,500 anonymized T2-weighted MRI scans. Data split: 70% Train, 15% Val, 15% Test. Class imbalance addressed via data augmentation.",
        "intended_use": "Primary clinical decision support for identifying ischemic and hemorrhagic stroke patterns.",
        "limitations": "May exhibit reduced specificity on scans with significant motion artifacts or non-standard contrast settings.",
        "accuracy": "97.4%",
        "sensitivity": "97.0%",  # Recall
        "specificity": "98.0%",  # Estimated from Precision
        "auc": "1.00",
        "params": "20.0M"
    },
    "ResNet50": {
        "type": "Residual Neural Network",
        "architecture": "ResNet50 (50 Layers, Skip Connections)",
        "dataset_info": "Trained on 'Brain Stroke MRI' dataset (Kaggle). Utilizes skip connections to mitigate vanishing gradient problems in deep feature extraction.",
        "intended_use": "Secondary validation tool for stroke classification.",
        "limitations": "Lower sensitivity compared to VGG19; recommended for use in conjunction with other models.",
        "accuracy": "94.0%",
        "sensitivity": "89.0%",
        "specificity": "99.0%",
        "auc": "0.94",
        "params": "24.1M"
    },
    "EfficientNetB0": {
        "type": "Efficient Neural Network",
        "architecture": "EfficientNetB0 (Compound Scaling)",
        "dataset_info": "Trained on 'Brain Stroke MRI' dataset (Kaggle). Optimized for computational efficiency and low-latency inference.",
        "intended_use": "Rapid triage in resource-constrained environments (e.g., mobile/edge devices).",
        "limitations": "Highest false-negative rate among the three models; use with caution for critical rule-out diagnosis.",
        "accuracy": "91.0%",
        "sensitivity": "95.0%",
        "specificity": "88.0%",
        "auc": "0.96",
        "params": "4.3M"
    }
}