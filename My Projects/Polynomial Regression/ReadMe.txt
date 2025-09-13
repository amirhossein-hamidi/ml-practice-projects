## Project 1: Basic Polynomial Regression

**Description:**  
- This project generates synthetic data for `pageSpeed` (time in seconds) and `purchaseAmount` (in dollars).  
- The purchase amount is inversely related to page speed to simulate realistic user behavior.  
- A scatter plot visualizes the data, and a 4th-degree polynomial regression is applied to fit the relationship.  
- The model's R² score is computed to measure goodness-of-fit.

**Key Features:**  
- Data generation using NumPy (`np.random.normal`)  
- Scatter plot visualization with Matplotlib  
- Polynomial regression using `np.polyfit` and `np.poly1d`  
- Evaluation using R² score from `sklearn.metrics`  

**Output:**  
- Scatter plot with fitted polynomial curve  
- R² score for the polynomial model

---

## Project 2: Advanced Polynomial Regression with Cross-Validation

**Description:**  
- This project extends the first one by testing multiple polynomial degrees (1 to 5) and comparing their fits.  
- Cross-validation (5-fold) is implemented to avoid overfitting and select the best model degree.  
- Detailed visualizations show how each polynomial degree fits the data.  

**Key Features:**  
- Conversion of features to polynomial terms with `PolynomialFeatures`  
- Multiple polynomial regression models  
- Model evaluation with both training R² and 5-fold cross-validated R²  
- Data visualization using Seaborn and Matplotlib  
- Selection of the best polynomial degree based on cross-validation  

**Output:**  
- Scatter plot of raw data  
- Polynomial fit curves for each degree with R² scores  
- Best polynomial degree based on cross-validated R²

---

## Requirements

- Python 3.8+  
- Libraries:  
  - numpy  
  - pandas  
  - matplotlib  
  - seaborn  
  - scikit-learn  

Install required packages using:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn