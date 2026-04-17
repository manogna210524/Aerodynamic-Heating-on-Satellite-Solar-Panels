import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import (mean_absolute_error, mean_squared_error,
                                     r2_score)

from sklearn.neighbors  import KNeighborsRegressor
from sklearn.svm        import SVR
from sklearn.linear_model import LinearRegression
from sklearn.tree       import DecisionTreeRegressor
from sklearn.ensemble   import AdaBoostRegressor, RandomForestRegressor

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401

# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv('/content/drive/MyDrive/SAT_Lab-11032026/APEX_Master_Dataset_noisy_v2.csv')
df.columns = df.columns.str.strip().str.lower().str.replace(r'[^a-z0-9_]', '_', regex=True)
print("Detected columns:", df.columns.tolist())

FEATURE_COLS = [c for c in df.columns if c != 'max_temp_k']
TARGET_COL   = 'max_temp_k'
IRR_COL      = FEATURE_COLS[0]   # irradiation_wm2
AMB_COL      = FEATURE_COLS[1]   # ambient_temp_k

X = df[FEATURE_COLS].values
y = df[TARGET_COL].values

print(f"\nDataset shape : {df.shape}")
print(f"Features      : {FEATURE_COLS}")
print(f"Target        : {TARGET_COL}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  CORRELATION MATRIX
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 55)
print("  CORRELATION MATRIX")
print("=" * 55)
corr = df.corr()
print(corr.to_string(), "\n")

fig_corr, ax_corr = plt.subplots(figsize=(6, 5))
cax = ax_corr.imshow(corr.values, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
plt.colorbar(cax, ax=ax_corr, label='Pearson r')

labels = [c.replace('_', '\n') for c in corr.columns]
ax_corr.set_xticks(range(len(labels))); ax_corr.set_xticklabels(labels, fontsize=9)
ax_corr.set_yticks(range(len(labels))); ax_corr.set_yticklabels(labels, fontsize=9)
ax_corr.set_title('Feature Correlation Matrix', fontsize=13, fontweight='bold', pad=12)

# Annotate cells with correlation values
for i in range(len(corr)):
    for j in range(len(corr)):
        val = corr.values[i, j]
        color = 'white' if abs(val) > 0.6 else 'black'
        ax_corr.text(j, i, f'{val:.3f}', ha='center', va='center',
                     fontsize=10, fontweight='bold', color=color)

plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved → correlation_matrix.png\n")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  TRAIN / TEST SPLIT  +  SCALING
# ─────────────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print(f"Train samples : {len(X_train)}")
print(f"Test  samples : {len(X_test)}\n")

scaler   = StandardScaler()
Xs_train = scaler.fit_transform(X_train)
Xs_test  = scaler.transform(X_test)

# ─────────────────────────────────────────────────────────────────────────────
# 4.  MODEL DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────
models = {
    "KNN (k=5)"            : (KNeighborsRegressor(n_neighbors=5),                    True),
    "SVR (RBF kernel)"     : (SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1),      True),
    "Multiple Linear Reg." : (LinearRegression(),                                     False),
    "Decision Tree"        : (DecisionTreeRegressor(random_state=42),                False),
    "AdaBoost"             : (AdaBoostRegressor(n_estimators=100, random_state=42),  False),
    "Random Forest"        : (RandomForestRegressor(n_estimators=200, random_state=42), False),
}

MODEL_COLORS = ["#3266AD", "#1D9E75", "#D85A30", "#7F77DD", "#BA7517", "#D4537E"]

# ─────────────────────────────────────────────────────────────────────────────
# 5.  TRAIN + EVALUATE
# ─────────────────────────────────────────────────────────────────────────────
kf = KFold(n_splits=5, shuffle=True, random_state=42)
results, all_preds = [], {}

HEADER = (f"\n{'Model':<26} {'CV R²(mean)':>12} {'CV R²(std)':>11} "
          f"{'Test R²':>9} {'Test MAE':>10} {'Test RMSE':>11}")
SEP = "─" * len(HEADER)

print(HEADER)
print(SEP)

for name, (model, use_scaled) in models.items():
    Xtr = Xs_train if use_scaled else X_train
    Xte = Xs_test  if use_scaled else X_test

    cv_r2  = cross_val_score(model, Xtr, y_train, cv=kf, scoring='r2')
    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    all_preds[name] = y_pred

    row = dict(
        Model       = name,
        CV_R2_Mean  = round(cv_r2.mean(), 4),
        CV_R2_Std   = round(cv_r2.std(),  4),
        Test_R2     = round(r2_score(y_test, y_pred), 4),
        Test_MAE    = round(mean_absolute_error(y_test, y_pred), 4),
        Test_RMSE   = round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
    )
    results.append(row)

    print(f"{name:<26} {cv_r2.mean():>12.4f} {cv_r2.std():>11.4f} "
          f"{row['Test_R2']:>9.4f} {row['Test_MAE']:>10.4f} {row['Test_RMSE']:>11.4f}")

print(SEP)

results_df = pd.DataFrame(results).sort_values('Test_R2', ascending=False).reset_index(drop=True)
results_df.index += 1
print("\n📊  MODEL RANKING  (by Test R²)\n")
print(results_df.to_string())

best = results_df.iloc[0]
print(f"\n🏆  Best model : {best['Model']}  |  Test R² = {best['Test_R2']}")

# ─────────────────────────────────────────────────────────────────────────────
# 6.  BUILD PREDICTION GRID  (for 3D surfaces)
# ─────────────────────────────────────────────────────────────────────────────
irr_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 50)
amb_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 50)
irr_grid, amb_grid = np.meshgrid(irr_range, amb_range)
grid_raw = np.c_[irr_grid.ravel(), amb_grid.ravel()]
grid_scl = scaler.transform(grid_raw)


def get_surface(model, use_scaled):
    g = grid_scl if use_scaled else grid_raw
    return model.predict(g).reshape(irr_grid.shape)


# ─────────────────────────────────────────────────────────────────────────────
# 7A.  3D SCATTER – RAW DATA
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(10, 7))
ax  = fig.add_subplot(111, projection='3d')
sc  = ax.scatter(X[:, 0], X[:, 1], y,
                 c=y, cmap='viridis', s=15, alpha=0.65)
plt.colorbar(sc, ax=ax, shrink=0.5, label='Max Temp (K)')
ax.set_xlabel('Irradiation (W/m²)', labelpad=8)
ax.set_ylabel('Ambient Temp (K)',   labelpad=8)
ax.set_zlabel('Max Temp (K)',       labelpad=8)
ax.set_title('3D Scatter – All Data Points', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('3d_scatter_raw.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved → 3d_scatter_raw.png")

# ─────────────────────────────────────────────────────────────────────────────
# 7B.  3D PREDICTION SURFACES – ALL 6 MODELS  (2 × 3 grid)
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 12))
fig.suptitle('3D Prediction Surfaces — All Models\n(X = Irradiation  |  Y = Ambient Temp  |  Z = Max Temp)',
             fontsize=14, fontweight='bold', y=1.01)

for idx, ((name, (model, use_scaled)), color) in enumerate(zip(models.items(), MODEL_COLORS)):
    ax = fig.add_subplot(2, 3, idx + 1, projection='3d')
    Z  = get_surface(model, use_scaled)
    r2v = next(r['Test_R2'] for r in results if r['Model'] == name)

    # Prediction surface
    surf = ax.plot_surface(irr_grid, amb_grid, Z,
                           cmap='plasma', alpha=0.70, linewidth=0,
                           antialiased=True)

    # Actual test points (red)
    ax.scatter(X_test[:, 0], X_test[:, 1], y_test,
               c='red', s=8, alpha=0.55, zorder=5, label='Actual')

    # Predicted test points (cyan)
    ax.scatter(X_test[:, 0], X_test[:, 1], all_preds[name],
               c='cyan', s=8, alpha=0.45, marker='^', zorder=4, label='Predicted')

    ax.set_title(f'{name}\nR² = {r2v:.4f}', fontsize=9, fontweight='bold')
    ax.set_xlabel('Irr (W/m²)', fontsize=7, labelpad=4)
    ax.set_ylabel('Amb T (K)',  fontsize=7, labelpad=4)
    ax.set_zlabel('Max T (K)',  fontsize=7, labelpad=4)
    ax.tick_params(labelsize=6)

    if idx == 0:
        ax.legend(fontsize=6, loc='upper left')

plt.tight_layout()
plt.savefig('3d_prediction_surfaces_all_models.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved → 3d_prediction_surfaces_all_models.png")

# ─────────────────────────────────────────────────────────────────────────────
# 7C.  INDIVIDUAL 3D PLOTS – one full-size figure per model
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating individual 3D plots for each model...")
for (name, (model, use_scaled)), color in zip(models.items(), MODEL_COLORS):
    Z   = get_surface(model, use_scaled)
    r2v = next(r['Test_R2'] for r in results if r['Model'] == name)
    mae = next(r['Test_MAE']  for r in results if r['Model'] == name)

    fig = plt.figure(figsize=(11, 8))
    ax  = fig.add_subplot(111, projection='3d')

    # Smooth prediction surface
    surf = ax.plot_surface(irr_grid, amb_grid, Z,
                           cmap='plasma', alpha=0.72, linewidth=0,
                           antialiased=True)
    plt.colorbar(surf, ax=ax, shrink=0.45, pad=0.08, label='Predicted Max Temp (K)')

    # Actual test points
    ax.scatter(X_test[:, 0], X_test[:, 1], y_test,
               c='red', s=20, alpha=0.70, zorder=5, label='Actual (test)')

    # Predicted test points
    ax.scatter(X_test[:, 0], X_test[:, 1], all_preds[name],
               c='lime', s=20, alpha=0.55, marker='^', zorder=4, label='Predicted (test)')

    ax.set_xlabel('Irradiation (W/m²)', fontsize=10, labelpad=10)
    ax.set_ylabel('Ambient Temp (K)',   fontsize=10, labelpad=10)
    ax.set_zlabel('Max Temp (K)',       fontsize=10, labelpad=10)
    ax.set_title(
        f'3D Prediction Surface — {name}\nTest R² = {r2v:.4f}  |  Test MAE = {mae:.4f} K',
        fontsize=12, fontweight='bold', pad=15
    )
    ax.legend(fontsize=9, loc='upper left')
    ax.view_init(elev=22, azim=225)   # nice viewing angle

    fname = f"3d_{name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('=','')}.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved → {fname}")

# ─────────────────────────────────────────────────────────────────────────────
# 7D.  RESIDUAL 3D PLOTS – all 6 models
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 12))
fig.suptitle('3D Residual Analysis — All Models  (Actual − Predicted)',
             fontsize=14, fontweight='bold', y=1.01)

for idx, ((name, _), color) in enumerate(zip(models.items(), MODEL_COLORS)):
    ax  = fig.add_subplot(2, 3, idx + 1, projection='3d')
    res = y_test - all_preds[name]
    lim = abs(res).max()

    sc = ax.scatter(X_test[:, 0], X_test[:, 1], res,
                    c=res, cmap='coolwarm', s=18, alpha=0.80,
                    vmin=-lim, vmax=lim)
    plt.colorbar(sc, ax=ax, shrink=0.40, label='Residual (K)')

    # Zero-plane reference
    xx, yy = np.meshgrid(
        [X_test[:, 0].min(), X_test[:, 0].max()],
        [X_test[:, 1].min(), X_test[:, 1].max()]
    )
    ax.plot_surface(xx, yy, np.zeros_like(xx),
                    color='gray', alpha=0.15, linewidth=0)

    r2v = next(r['Test_R2'] for r in results if r['Model'] == name)
    ax.set_title(f'{name}\nR² = {r2v:.4f}', fontsize=9, fontweight='bold')
    ax.set_xlabel('Irr (W/m²)', fontsize=7, labelpad=4)
    ax.set_ylabel('Amb T (K)',  fontsize=7, labelpad=4)
    ax.set_zlabel('Residual (K)', fontsize=7, labelpad=4)
    ax.tick_params(labelsize=6)

plt.tight_layout()
plt.savefig('3d_residuals_all_models.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved → 3d_residuals_all_models.png")

# ─────────────────────────────────────────────────────────────────────────────
# 7E.  MODEL COMPARISON BAR CHART  (2D summary)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold')

metrics = [
    ('Test_R2',   'Test R²  ↑ (higher = better)'),
    ('Test_MAE',  'Test MAE ↓ (K)'),
    ('Test_RMSE', 'Test RMSE ↓ (K)'),
]
names_ranked = results_df['Model'].values

for ax, (col, label) in zip(axes, metrics):
    vals  = [next(r[col] for r in results if r['Model'] == n) for n in names_ranked]
    bars  = ax.barh(names_ranked, vals, color=MODEL_COLORS[:len(names_ranked)],
                    edgecolor='white', height=0.6)
    ax.set_xlabel(label, fontsize=10)
    ax.set_title(label,  fontsize=10, fontweight='bold')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + max(vals) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'{v:.4f}', va='center', fontsize=8)
    ax.invert_yaxis()
    ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig('model_comparison_bar.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved → model_comparison_bar.png")

# ─────────────────────────────────────────────────────────────────────────────
# 8.  SAVE RESULTS CSV
# ─────────────────────────────────────────────────────────────────────────────
results_df.to_csv('model_comparison_results.csv', index=True)
print("\nResults saved → model_comparison_results.csv")
print("\n── All outputs ─────────────────────────────────────────────────────────")
print("  correlation_matrix.png")
print("  3d_scatter_raw.png")
print("  3d_prediction_surfaces_all_models.png")
print("  3d_knn_k5.png  |  3d_svr_rbf_kernel.png  |  3d_multiple_linear_reg.png")
print("  3d_decision_tree.png  |  3d_adaboost.png  |  3d_random_forest.png")
print("  3d_residuals_all_models.png")
print("  model_comparison_bar.png")
print("  model_comparison_results.csv")