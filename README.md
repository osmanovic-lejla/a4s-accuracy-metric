## AI Security Project – Brier Score (Calibration Metric)

Course: AI and Cybersecurity – University of Luxembourg (2025/2026)

Name: Lejla Osmanovic

GitHub Link: https://github.com/osmanovic-lejla/a4s-accuracy-metric

This repository contains my work for the AI Security course project.
The goal is to implement a custom reliability metric (Brier Score) and integrate it into the A4S evaluation framework, and to evaluate it on both tabular and image classification models.

⸻

# 1. Metric: Brier Score

Location:
a4s_eval/metrics/model_metrics/brier_score_metric.py

Registry decorator:
@model_metric(name="brier_score")

Idea:
The Brier Score measures how close a model’s predicted probabilities are to the true labels.
It evaluates probabilistic accuracy / calibration:
	•	Lower Brier Score → better calibrated, more reliable probabilities
	•	Higher Brier Score → overconfident or poorly calibrated predictions

The metric supports:
	•	Binary and multiclass classification
	•	Tabular and image models
	•	Models with predict_proba, with fallback to predict if needed

Formula:

Brier Score = mean( (p - y)^2 )

Where:
	•	p = predicted probability
	•	y = true label (0/1 or one-hot encoded for multiclass)

The metric returns:
A single Measure object containing:
	•	name = “brier_score”
	•	score = 
	•	time = 

Applies to:
Any classification model integrated into A4S, including:
	•	Logistic Regression
	•	Random Forest
	•	Gradient Boosting
	•	SVC with probabilities
	•	ResNet models (via FunctionalModel on CIFAR-10)

⸻

# 2. Metric Computation Details

For each batch of data:
	1.	Features (X) and target (y) are extracted using the A4S DataShape and Dataset.
	2.	Inputs are converted from pandas DataFrame to a torch.Tensor for PyTorch models.
	3.	Probabilities are obtained using (in this order):
	•	functional_model.predict_proba(Xb)
	•	functional_model.predict_proba(model, Xb)
	•	fallback to predict / predict(model, Xb) if predict_proba is not available
	4.	If the output is:
	•	1D → treated as binary probability
	•	2D → treated as multiclass probability matrix and compared to one-hot labels
	5.	The mean squared error between probabilities and true labels is accumulated.
	6.	The final Brier Score is the dataset-wide average.

⸻

# 3. Tabular Data Metric (Breast Cancer, A4S Integration)

A4S automatically discovers and runs the metric using its model metric registry.

Additional validation and experiment logic are provided in:

Files:
	•	tests/metrics/model_metrics/test_brier_score.py
	•	tests/metrics/model_metrics/test_brier_score_breast_cancer.py

Models tested:
	•	Logistic Regression
	•	Random Forest
	•	Gradient Boosting
	•	SVC (with probability=True)

Run tests inside project root:

uv sync
uv run pytest -s tests/metrics/model_metrics/test_brier_score_breast_cancer.py

Expected output example:

Logistic Regression:  brier_score=0.0253
Random Forest:        brier_score=0.0255
Gradient Boosting:    brier_score=0.0328
SVC (probabilities):  brier_score=0.0340

Results saved to:
tests/data/measures/brier_score_breast_cancer.csv

Each row contains:
	•	model
	•	accuracy
	•	brier_score

This confirms that the metric works correctly on tabular data and that all tests pass.

⸻

# 4. CIFAR-10 Brier Score Evaluation (Image Extension)

To analyze calibration on deep learning models, a second evaluation is performed on CIFAR-10.

File:
tests/metrics/model_metrics/test_brier_score_cifar.py

Models tested:
	•	ResNet18
	•	ResNet34
	•	ResNet50

Each model is evaluated on CIFAR-10 to compute:
	•	accuracy
	•	Brier Score

Run command:

uv run pytest -s tests/metrics/model_metrics/test_brier_score_cifar.py

Example output:

Evaluating ResNet18...
ResNet18 - accuracy=..., brier_score=1.0575

Evaluating ResNet34...
ResNet34 - accuracy=..., brier_score=1.1616

Evaluating ResNet50...
ResNet50 - accuracy=..., brier_score=0.9285

Results saved to:
tests/data/measures/brier_score_cifar_resnets.csv

Visual results:
Optionally, example CIFAR-10 images and their predictions may be saved under:

image_brier_flips_cifar10/<model_name>/

Note:
CIFAR-10 is downloaded automatically via
torchvision.datasets.CIFAR10(download=True)

⸻

# 5. Notebook

File used for visualization:
brier_score_breast_cancer.ipynb

The notebook visualizes Brier Score results for both tabular and image experiments.

This notebook is used to generate the visuals for the presentation.

⸻

# 6. Experimental Findings
	•	The Brier Score reveals how well-calibrated different models are.

Breast Cancer (Tabular)
	•	Logistic Regression has the lowest Brier Score (best calibration).
	•	Random Forest performs similarly well.
	•	Gradient Boosting and SVC show worse calibration (higher Brier scores).

CIFAR-10 (Image)
	•	ResNet50 achieves the best Brier Score among the tested ResNets.
	•	ResNet34 shows the highest Brier Score (worst calibration).

General conclusions
	•	High accuracy does NOT always mean good calibration.
	•	Deep models tend to be overconfident — the Brier Score exposes this clearly.

⸻

# 7. How to Reproduce

All tests and metrics can be run directly from the project root.

1. Install dependencies

uv sync

2. Run all metric-related tests (tabular + CIFAR)

uv run pytest -s tests/metrics/model_metrics

Or run the full test suite

uv run pytest -s

Results saved to:

tests/data/measures/
    • brier_score.csv
    • brier_score_breast_cancer.csv
    • brier_score_cifar_resnets.csv

Optional image outputs:

image_brier_flips_cifar10/

Notebook for visualization:

brier_score_breast_cancer.ipynb
