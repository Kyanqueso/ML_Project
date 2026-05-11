# Steel Industry Energy Consumption — MLP vs TCN

Predicts 15-minute electrical energy consumption (`Usage_kWh`) in a steel plant using two neural network architectures: a **Multilayer Perceptron (MLP)** and a **Temporal Convolutional Network (TCN)**.

## Results Summary

| Model | MAE (kWh) | RMSE (kWh) | R²     |
|-------|-----------|------------|--------|
| MLP   | 1.0460    | 1.5830     | 0.9975 |
| TCN   | 1.2380    | 1.9819     | 0.9960 |

Both models achieve R² > 0.99. The MLP outperforms the TCN, likely because the categorical context features (load type, weekday) are already highly informative, leaving little room for temporal modelling to add value.

---

## Project Structure

```
project/
├── Project_steel_energy_mlp_vs_tcn.ipynb   # Main notebook (annotated)
├── Steel_industry_data.csv                 # Raw dataset
└── README.md
```

---

## Dataset

- **Source**: UCI Steel Industry Energy Consumption dataset
- **File**: `Steel_industry_data.csv`
- **Shape**: 35,040 rows × 11 columns (15-minute intervals, full year 2018)
- **Target**: `Usage_kWh`
- **Features**: reactive power (lagging/leading), CO2 emissions, power factors, NSM (seconds from midnight), week status, day of week, load type
- **Missing values**: none

---

## Requirements

### Python version

Python **3.10 – 3.11** is recommended for TensorFlow 2.10.x compatibility.  
The notebook was originally executed with:

| Package       | Version used |
|---------------|-------------|
| tensorflow    | 2.10.1      |
| scikit-learn  | 1.x         |
| pandas        | 1.x / 2.x  |
| numpy         | 1.x         |
| matplotlib    | 3.x         |

> **Note**: The current environment has TensorFlow 2.21.0 and Python 3.13. Results should be numerically close but minor floating-point differences are expected across TF versions. Random seeds are fixed (`numpy` seed 42, `tf.random.set_seed(42)`) to maximise reproducibility within a single environment.

### Install dependencies

```bash
pip install tensorflow==2.10.1 scikit-learn pandas numpy matplotlib
```

If you are on an Apple Silicon Mac or need a newer TensorFlow:

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib
```

---

## Reproducing Results

### 1. Clone / download the project

Ensure all three files are in the same directory:
- `Project_steel_energy_mlp_vs_tcn.ipynb`
- `Steel_industry_data.csv`

### 2. Launch Jupyter

```bash
jupyter notebook
# or
jupyter lab
```

Open `Project_steel_energy_mlp_vs_tcn.ipynb`.

### 3. Run all cells in order

Use **Kernel → Restart & Run All** to guarantee a clean run from top to bottom.

Do **not** run cells out of order — the notebook is stateful: later sections depend on variables set in earlier ones (scaler, train/test splits, windowed arrays).

### 4. Expected outputs

| Section | What you should see |
|---------|---------------------|
| Section 1 | Shape `(35040, 11)`, 0 missing values |
| Section 2 | Scaled training mean ≈ 0, std ≈ 1 |
| Section 3 | MLP training stops before epoch 100; loss/scatter plots appear |
| Section 4 | TCN training stops before epoch 100; loss/scatter plots appear |
| Section 5 | Comparison table with MLP winning on MAE, RMSE, and R² |

---

## Key Design Decisions

| Decision | Reason |
|----------|--------|
| No shuffle in train/test split | Time-series data — shuffling would let the model see the future |
| Scaler fit only on training data | Prevents data leakage from test set statistics |
| TCN window size W = 24 | 24 × 15 min = 6 hours of history per prediction |
| Dilation rates [1, 2, 4, 8] | Exponentially growing receptive field without extra depth |
| EarlyStopping patience = 10 | Stops training if validation loss stalls, restores best weights |
| MLP aligned to TCN test slice | Both models evaluated on the exact same 6,984 test targets |
