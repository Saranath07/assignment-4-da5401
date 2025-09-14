# DA5401 Assignment 4: GMM-Based Synthetic Sampling for Imbalanced Data

**Project Phoenix - Enhancing Fraud Detection with Probabilistic Modeling**

*Submitted by: Saranath P (DA25E003)*  
*Date: 15-Sep-2025*

## 📁 Project Structure

```
da5401-data-analysis-laboratory/assignment-4-da5401/
├── README.md                 # Project documentation (this file)
├── .gitignore               # Git ignore rules
├── creditcard.csv           # Credit card fraud dataset
├── da25e003.ipynb          # Main Jupyter notebook with analysis
└── DA5401 A4 GMM.pdf       # PDF report of the analysis
```

## 🎯 Project Overview

This project addresses the critical challenge of **class imbalance in fraud detection** using advanced probabilistic modeling techniques. The work simulates a real-world scenario for SecureBank Inc., where the goal is to improve fraud detection capabilities by leveraging **Gaussian Mixture Models (GMMs)** for synthetic data generation.

### Key Problem Statement
- **Severe Class Imbalance**: Only 0.1727% (492 out of 284,807) transactions are fraudulent
- **Business Impact**: Traditional models miss ~36% of fraudulent transactions
- **Solution**: Use GMM-based oversampling to generate high-quality synthetic fraud data

## 🔬 Technical Approach

### Part A: Baseline Analysis
- **Dataset**: Credit card transactions with PCA-transformed features
- **Preprocessing**: Feature scaling for `Time` and `Amount` columns
- **Baseline Model**: Logistic Regression on imbalanced data
- **Key Metrics**: Focus on Precision, Recall, and F1-Score for fraud detection

### Part B: GMM-Based Synthetic Sampling
- **Theory**: Probabilistic approach vs. geometric methods (SMOTE)
- **Model Selection**: Optimal number of components using AIC/BIC criteria
- **Approaches Tested**:
  1. Pure GMM Oversampling (1:1 balance)
  2. Hybrid CBU + GMM (Clustering-Based Undersampling + Oversampling)

### Part C: Performance Comparison
- **Evaluation**: All models tested on same imbalanced test set
- **Focus**: Maximize Recall (fraud detection rate) while managing Precision trade-offs

## 📊 Key Results

| Model | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Baseline | 82.9% | 64.3% | 0.724 |
| GMM-Oversampling | 8.3% | **90.8%** | 0.153 |
| Hybrid (CBU+GMM) | 4.5% | 88.8% | 0.086 |

### 🎯 Business Impact
- **Fraud Detection Improvement**: From missing 36% to catching 90%+ of fraudulent transactions
- **Strategic Trade-off**: Lower precision accepted for dramatically higher recall
- **Financial Justification**: Cost of missed fraud >> Cost of investigating false positives

## 🛠️ Technologies Used

- **Python Libraries**:
  - `pandas`, `numpy` - Data manipulation
  - `matplotlib`, `seaborn` - Visualization
  - `scikit-learn` - Machine learning models and preprocessing
  - `imblearn` - Imbalanced learning techniques
- **Key Algorithms**:
  - Gaussian Mixture Models (GMM)
  - Logistic Regression
  - Clustering-Based Undersampling (CBU)
  - AIC/BIC model selection

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

### Running the Analysis
1. **Clone/Download** the project files
2. **Ensure** `creditcard.csv` is in the project directory
3. **Open** [`da25e003.ipynb`](da25e003.ipynb) in Jupyter Notebook/Lab
4. **Run** all cells sequentially to reproduce the analysis

### Dataset Requirements
- The `creditcard.csv` file should contain:
  - PCA-transformed features (V1-V28)
  - `Time` and `Amount` columns
  - `Class` column (0=Normal, 1=Fraud)

## 📈 Key Insights

### Why GMM Over SMOTE?
1. **Multi-modal Capability**: Can model different types of fraud patterns
2. **Probabilistic Foundation**: Learns true data distribution rather than geometric interpolation
3. **Noise Robustness**: Less susceptible to outlier amplification

### Model Selection Process
- **AIC/BIC Analysis**: Determined optimal k=3 components for fraud data
- **Business Context**: Prioritized Recall over Precision due to fraud cost implications
- **Recommendation**: Pure GMM-Oversampling for best balance of performance

## 📋 Project Highlights

- ✅ **Comprehensive Analysis**: From data exploration to business recommendations
- ✅ **Advanced Techniques**: GMM-based synthetic data generation
- ✅ **Real-world Context**: Framed as business consulting project
- ✅ **Rigorous Evaluation**: Multiple models compared on consistent test set
- ✅ **Actionable Results**: Clear recommendations with business justification

## 📖 Additional Resources

- **Full Report**: See [`DA5401 A4 GMM.pdf`](DA5401%20A4%20GMM.pdf) for detailed analysis
- **Code Implementation**: All code available in [`da25e003.ipynb`](da25e003.ipynb)
- **Course**: DA5401 - Data Analysis Laboratory

---

**Note**: This project demonstrates advanced techniques in handling imbalanced datasets for fraud detection, showcasing the power of probabilistic models in generating high-quality synthetic data for machine learning applications.