# M-CARE: Maternal Causal Actionable Risk Engine

**M-CARE** is an advanced AI-driven framework designed to assess maternal health risks during pregnancy. Unlike traditional "black box" predictive models, M-CARE focuses on **Interpretability** and **Actionability**. It not only predicts if a patient is at "High Risk" but also explains *why* (using SHAP-IQ) and prescribes *how* to lower that risk (using DiCE counterfactuals).

## 🚀 Key Features

  * **Foundation Model for Tabular Data:** Utilizes **TabPFN** (Tabular Prior-Fitted Network), a Transformer-based model pre-trained on synthetic datasets. It enables state-of-the-art accuracy on small-to-medium datasets without the need for extensive hyperparameter tuning.
  * **High-Order Interpretability:** Uses **SHAP-IQ** to visualize complex feature interactions, offering a deeper "Diagnosis" than standard feature importance methods.
  * **Actionable Prescriptions:** Integrates **DiCE** (Diverse Counterfactual Explanations) to act as an "AI Doctor." It generates medically valid, personalized plans (Counterfactuals) to transition a patient from "High Risk" to "Low Risk."
  * **Medical Safety Bounds:** The prescriptive engine is constrained by real-world clinical ranges to ensure suggestions are physiologically possible (e.g., keeping Body Temp between 97°F-99°F).

## 📊 Dataset

The model is trained on the **Maternal Health Risk Assessment Dataset** (sourced from Mendeley Data). It contains clinical and physiological data points including:

  * **Age:** Age in years.
  * **SystolicBP / DiastolicBP:** Blood pressure levels (mmHg).
  * **BS:** Blood Sugar levels (mmol/L).
  * **BodyTemp:** Body temperature (°F).
  * **HeartRate:** Heart rate (bpm).
  * **Risk Level:** Target variable (Mapped to: 0 = Low Risk, 1 = High Risk).

## 🛠️ Tech Stack

  * **Python 3.10+**
  * **Modeling:** `TabPFN` (Transformer for Tabular Classification)
  * **Explainability:** `shapiq` (Shapley Interaction values), `dice-ml` (Counterfactuals)
  * **Data Processing:** `pandas`, `numpy`, `scikit-learn`
  * **Visualization:** `matplotlib`

## 📦 Installation

To replicate this environment, install the required dependencies:

```bash
pip install tabpfn dice-ml shapiq pandas scikit-learn matplotlib numpy
```

*Note: TabPFN runs significantly faster on a machine with a GPU (e.g., NVIDIA T4 via Google Colab).*

## Workflow

1.  **Data Preprocessing:**

      * Loads the maternal health dataset.
      * Maps risk levels to binary classification (High Risk vs. Low Risk).
      * Splits data into training and testing sets.

2.  **Model Training:**

      * Initializes the `TabPFNClassifier`.
      * Fits the model using In-Context Learning (no traditional iterative training epochs required).

3.  **Diagnosis (SHAP-IQ):**

      * Selects a specific "High Risk" patient from the test set.
      * Calculates Shapley values to identify which features (e.g., high Blood Sugar, Age) contributed most to the prediction.
      * Generates **Force Plots** and **Waterfall Plots** for visualization.

4.  **Prescription (DiCE):**

      * Defines medical safety bounds (e.g., Systolic BP cannot drop below 90).
      * Optimizes for the minimal set of changes required to flip the prediction from "High" to "Low."
      * Outputs an actionable plan (e.g., *"Reduce Systolic BP by 12.2 mmHg"*).

## 📈 Performance

In the provided notebook run, M-CARE achieved exceptional performance:

  * **Accuracy:** **99.16%** on the test set.
  * **Confusion Matrix:** Demonstrated high precision and recall for determining risk categories.

## 🩺 Example Output

**Patient Diagnosis:**

> Current Prediction: **HIGH RISK**

**AI Prescription (Counterfactual):**

>   * **Reduce** Systolic BP by 12.2 (Target: 117.8)
>   * **Reduce** BMI by 6.5 (Target: 21.0)

## 📄 Acknowledgements

  * **Dataset:** Mojumdar, Mayen Uddin et al. (2024), “Maternal Health Risk Assessment Dataset”, Mendeley Data, V1.
  * **Libraries:** This project leans heavily on the open-source contributions of the [TabPFN](https://github.com/automl/TabPFN), [DiCE](https://github.com/interpretml/DiCE), and [shapiq](https://www.google.com/search?q=https://github.com/shapiq/shapiq) teams.

### Let's get connected

*https://www.linkedin.com/in/ocama-mohamed*
