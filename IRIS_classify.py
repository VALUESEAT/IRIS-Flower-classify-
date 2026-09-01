import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Handle scikit-learn version differences for calibration
try:
    from sklearn.frozen import FrozenEstimator
except ImportError:
    FrozenEstimator = None

# 1. Load the data
iris_data = pd.read_csv('IRIS.csv')

print("--- First 5 Rows ---")
print(iris_data.head())

print("\n--- Dataset Information ---")
iris_data.info()

print("\n--- Summary Statistics ---")
print(iris_data.describe())

print("\n--- Missing Values ---")
print(iris_data.isnull().sum())

print("\n--- Dataset Shape ---")
print(iris_data.shape)

print("\n--- Value Counts ---")
print(iris_data['species'].value_counts())

# 2. Data Visualizations and plot setting 
sns.set_theme()

# Count plot
plt.figure()
sns.countplot(x='species', data=iris_data)
plt.title('Species Count')
plt.show()

# plot for sepal length
plt.figure()
sns.barplot(x='sepal_length', data=iris_data)
plt.title('Sepal Length Bar Plot')
plt.show()

# Sepal Length vs Petal Length
sns.FacetGrid(iris_data, hue='species', height=5)\
   .map(plt.scatter, 'sepal_length', 'petal_length')\
   .add_legend()
plt.show()

#  Sepal Width vs Petal Width
sns.FacetGrid(iris_data, hue='species', height=5)\
   .map(plt.scatter, 'sepal_width', 'petal_width')\
   .add_legend()
plt.show()

# 3. Feature Matrix and Target Setup
selected_columns = ['sepal_width', 'sepal_length', 'petal_width', 'petal_length']
x = iris_data[selected_columns].values
y = iris_data['species'].values

print("\nFeature matrix (X) shape:", x.shape)
print("Target vector (Y) shape:", y.shape)

# 4. Label Encoding for converting data into numerical values 
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 5. First Split: 20% Testing Set
X_temp, X_test, y_temp, y_test = train_test_split(
    x, y, test_size=0.20, random_state=42, stratify=y
)

# 6. Second Split: 70% Training / 10% Calibration
X_train, X_cal, y_train, y_cal = train_test_split(
    X_temp, y_temp, test_size=0.125, random_state=42, stratify=y_temp
)

# 7. Standardize Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_cal_scaled = scaler.transform(X_cal)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining samples (70%): {len(X_train)}")
print(f"Calibration samples (10%): {len(X_cal)}")
print(f"Testing samples (20%): {len(X_test)}\n")

# 8. Define Base Models
base_models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='rbf', probability=True, random_state=42)
}

# 9. Train, Calibrate, and Evaluate
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

for idx, (name, base_model) in enumerate(base_models.items()):
    # Select feature sets (scaled for SVM, raw for tree-based models)
    X_tr = X_train_scaled if name == "SVM" else X_train
    X_ca = X_cal_scaled if name == "SVM" else X_cal
    X_te = X_test_scaled if name == "SVM" else X_test

    # Fit base model on training split
    base_model.fit(X_tr, y_train)

    # Calibrate probability predictions
    if FrozenEstimator is not None:
        calibrated_model = CalibratedClassifierCV(
            estimator=FrozenEstimator(base_model),
            method='isotonic'
        )
    else:
        calibrated_model = CalibratedClassifierCV(
            estimator=base_model,
            method='isotonic',
            cv='prefit'
        )

    calibrated_model.fit(X_ca, y_cal)

    # Evaluation
    y_pred = calibrated_model.predict(X_te)
    y_proba = calibrated_model.predict_proba(X_te)

    acc = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_proba)

    print(f"=== {name} ===")
    print(f"Accuracy: {acc * 100:.1f}% | Log Loss: {loss:.3f}")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    print("-" * 50)

    # Confusion Matrix setuo
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        cbar=False,
        ax=axes[idx],
        xticklabels=label_encoder.classes_, 
        yticklabels=label_encoder.classes_
    )
    axes[idx].set_title(f"{name}\nAcc: {acc*100:.1f}% | Log Loss: {loss:.3f}")
    axes[idx].set_xlabel("Predicted Label")
    axes[idx].set_ylabel("True Label")

plt.tight_layout()
plt.savefig("iris_confusion_matrices.png", dpi=300)
plt.show()
