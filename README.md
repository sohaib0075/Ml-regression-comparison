# 🤖 Machine Learning Regression & Neural Network — Comparative Study

A machine learning project that explores and compares multiple regression techniques on custom datasets. Covers classical ML models (Random Forest, Decision Tree, Lasso) with feature engineering, hyperparameter tuning, and a custom **PyTorch Neural Network** for regression.

---

## 📌 Overview

This project is structured into **3 tasks**:

| Task | Description |
|---|---|
| **Task 1a** | Random Forest Regression with polynomial feature engineering |
| **Task 1c** | Lasso Regression with GridSearchCV hyperparameter tuning |
| **Task 2** | Decision Tree Regression with GridSearchCV tuning |
| **Bonus** | Custom PyTorch Neural Network (3-layer MLP) |

---

## 📁 Datasets

| File | Used In |
|---|---|
| `Data.npz` | Tasks 1 & 2 — features `x`, target `y` |
| `Data_bonus.npz` | Bonus Task — features `x`, target `y` |

> Both datasets are NumPy `.npz` files loaded via `np.load()`.

---

## 🧪 Tasks Breakdown

### Task 1a — Random Forest Regression
- StandardScaler feature normalization
- Polynomial features (degree=2) concatenated with original features
- 80/20 train-test split
- **Result:** Training MSE ≈ 0.307

### Task 1c — Lasso Regression with Grid Search
- Pipeline: `PolynomialFeatures → Lasso`
- Grid search over polynomial degree `[2, 3]` and alpha `[0.0001, 0.001, 0.01, 0.1]`
- 5-fold cross-validation
- **Result:** Best model MSE ≈ 1.94 (higher due to L1 regularization sensitivity)

### Task 2 — Decision Tree Regression
- GridSearchCV over `max_depth`, `min_samples_split`, `min_samples_leaf`
- Best hyperparameters: `max_depth=15`, `min_samples_leaf=2`, `min_samples_split=5`
- Retrained on full dataset after tuning
- **Result:** Full dataset MSE ≈ 0.812

### Bonus — PyTorch Neural Network (MLP)
- 3-layer fully connected network: `128 → 64 → 1`
- ReLU activations, Adam optimizer (lr=0.001), MSELoss
- Trained for **100 epochs**, batch size 32
- StandardScaler normalization + DataLoader pipeline
- **Result:** Test MSE ≈ 0.0125

---

## 📏 Results Summary

| Model | MSE |
|---|---|
| Random Forest (train) | 0.307 |
| Lasso Regression (best) | 1.940 |
| Decision Tree (full data) | 0.812 |
| Neural Network (test) | **0.0125** ✅ |

---

## 🛠️ Tech Stack

- **Python 3.11**
- **scikit-learn** — Random Forest, Decision Tree, Lasso, GridSearchCV, Pipelines
- **PyTorch** — Custom MLP Neural Network, DataLoader, Adam optimizer
- **NumPy** — Data loading and preprocessing
- **Pandas** — Data handling

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install numpy pandas scikit-learn torch
```

### 2. Add Datasets

Place `Data.npz` and `Data_bonus.npz` in the project root directory.

### 3. Run the Notebook

```bash
jupyter notebook rechecked_ipynb.ipynb
```

---

## 📂 Project Structure

```
ml-regression-comparison/
├── rechecked_ipynb.ipynb   # Main notebook (all tasks)
├── Data.npz                # Dataset for Tasks 1 & 2 (not included)
├── Data_bonus.npz          # Dataset for Bonus Task (not included)
└── README.md
```

---

## 👤 Author

**sohaib0075**  
[GitHub Profile](https://github.com/sohaib0075)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
