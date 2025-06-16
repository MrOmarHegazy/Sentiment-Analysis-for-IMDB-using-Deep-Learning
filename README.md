# Sentiment Analysis with Custom Loss

This repository contains an end-to-end sentiment analysis pipeline built with PyTorch and Hugging Face. It demonstrates a custom loss function that weights samples by review length and prediction confidence, a full training/validation/test workflow with progress bars, caching, checkpointing, and rich visualizations of model performance.

---

## Project Overview

- **Task**: Classify IMDB movie reviews as positive or negative.
- **Backbone**: `bert-base-uncased` (extendable to any HF model).
- **Custom Loss**: `SentimentWeightedLoss` that applies:
  - **Length-based weighting**: longer reviews contribute more to the gradient.
  - **Confidence-based penalty**: highly confident wrong predictions incur a larger loss.
- **Key Features**:
  - Data caching with **pickle** to avoid repeated downloads.
  - Resumeable training with epoch-level **checkpointing**.
  - Device-agnostic (MPS/CUDA/CPU) with mixed-precision on MPS.
  - **Progress bars** (`tqdm`) for both training and evaluation loops.
  - Automated **plot generation** of metrics, ROC curve, confusion matrix, and epoch timings.

---

## Installation

```bash
# Clone the repo
git clone https://github.com/MrOmarHegazy/Sentiment-Analysis-for-IMDB-using-Deep-Learning.git
cd sentiment-analysis

# Create a venv and activate
python -m venv venv
# macOS/Linux
source venv/bin/activate
# Windows
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

**Note**: Requires PyTorch ≥1.12 for MPS support on Apple Silicon.

---

## Usage Examples

### 1. Train the model

```bash
python scripts/train.py
```

- Downloads & tokenizes the IMDB dataset (cached under data/).
- Trains for 5 epochs (resumeable via checkpoints/train_ckpt.pkl).
- Saves the best model to checkpoints/best-model.pt.
- Records per-epoch metrics and timings in checkpoints/history.pkl.

### 2. Evaluate On Test Set

```bash
python scripts/test.py
```

- Loads best-model.pt and runs inference on the IMDB test split.
- Displays & caches test accuracy and F1 in checkpoints/test_results.pkl.

### 3. Generate Plots

```bash
python scripts/plots.py
```

Generates a plots folder containing:

- plots/train_loss.png
- plots/val_metrics.png
- plots/smoothed_val_f1.png
- plots/epoch_time.png
- plots/roc_curve.png
- plots/confusion_matrix.png

---

## Explanation of the Custom Loss Function

The custom loss function is designed to improve sentiment classification by combining several weighting strategies on top of the standard binary cross-entropy:

- **Base Loss**  
  Uses binary cross-entropy with logits as the foundational loss term.

- **Length-based Weighting**  
  Assigns higher weight to longer reviews, under the assumption that they carry more information and should have greater influence on model updates.

- **Confidence-based Penalty**  
  Penalizes highly confident but incorrect predictions more heavily, encouraging the model to be better calibrated.

- **Combined Weighting**  
  The final loss for each sample is computed by multiplying the base loss by both the length weight and the confidence penalty, then averaging across the batch.

---

##  Model Performance Results

- **Validation**: Accuracy = 0.9172 | F1 = 0.9568  
- **Test**: Accuracy = 0.91528 | F1 = 0.91315  

**Comparison to State-of-the-Art**

| Model                                              | Accuracy | F1 Score | Source                                  |
|----------------------------------------------------|----------|----------|-----------------------------------------|
| Space-XLNet (Papers With Code SOTA)                | 94.88%   | —        | :contentReference[oaicite:0]{index=0} |
| textattack/xlnet-base-cased-imdb (Hugging Face)    | 95.352%  | 95.04%   | :contentReference[oaicite:1]{index=1} |
| finetuning-xlnet-base-cased-on-imdb (Hugging Face) | 95.06%   | 95.04%   | :contentReference[oaicite:2]{index=2} |

Despite trailing the top transformer-based models by ~3–4 percentage points, our BERT-based classifier with a custom weighted loss demonstrates strong performance and benefits from efficient MPS training.

**Performance Plots**

![Training Loss over Epochs](plots/train_loss.png)  
![Validation Accuracy & F1 over Epochs](plots/val_metrics.png)  
![Smoothed Validation F1](plots/smoothed_val_f1.png)  
![Epoch Duration per Epoch](plots/epoch_time.png)  
![ROC Curve](plots/roc_curve.png)  
![Confusion Matrix](plots/confusion_matrix.png)  


---

## Potential Improvements

- **Hyperparameter Optimization**  
  Automate search over learning rate, batch size, dropout rate, and loss-weight parameters (e.g. with Optuna).

- **Alternative Model Backbones**  
  Experiment with lighter or more efficient transformer variants (e.g. DistilBERT, RoBERTa) to speed up training and inference.

- **Data Augmentation**  
  Incorporate techniques like back-translation or synonym replacement to diversify examples and improve generalization.

- **Comprehensive Testing**  
  Add unit tests for data loading, the custom loss logic, and end-to-end training/inference workflows to ensure reliability.

- **Interactive Demo**  
  Build a simple web interface (Streamlit, FastAPI) for live inference and visualization of metrics such as ROC curves.

- **Embedding Visualization**  
  Project model representations at various training checkpoints using t-SNE or UMAP to inspect class separation.

- **Extended Evaluation Metrics**  
  Include precision-recall curves, calibration plots, and detailed confusion matrices for a deeper performance analysis.
