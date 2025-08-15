import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.metrics import confusion_matrix, average_precision_score
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib style
plt.style.use('default')
sns.set_palette("husl")

# === 1. Load Data ===
base_path = r"C:\Users\patru\OneDrive\Desktop\ML Practice\data"

print("Loading data files...")
admissions = pd.read_csv(f"{base_path}\\admissions.csv")
patients = pd.read_csv(f"{base_path}\\patients.csv")
diagnoses = pd.read_csv(f"{base_path}\\diagnoses_icd.csv")
d_icd = pd.read_csv(f"{base_path}\\d_icd_diagnoses.csv")

print(f"Loaded files:")
print(f"  Admissions: {admissions.shape}")
print(f"  Patients: {patients.shape}")
print(f"  Diagnoses: {diagnoses.shape}")
print(f"  D_ICD: {d_icd.shape}")

# === 2. Data Processing (same as before) ===
print("\nProcessing data...")

# Fix merge keys
diagnoses_cols = set(diagnoses.columns)
d_icd_cols = set(d_icd.columns)
merge_cols = list(diagnoses_cols.intersection(d_icd_cols))
merge_keys = ['icd_code']
if 'icd_version' in merge_cols:
    merge_keys.append('icd_version')

# Merge diagnoses
diagnoses = diagnoses.merge(d_icd, on=merge_keys, how="left")

# Create sepsis labels
sepsis_icd_codes = ['99591','99592','78552','R6520','R6521','A419']
diagnoses['sepsis'] = diagnoses['icd_code'].isin(sepsis_icd_codes).astype(int)
sepsis_labels = diagnoses.groupby('subject_id')['sepsis'].max().reset_index()

# Fix subject_id data types
admissions['subject_id'] = admissions['subject_id'].astype(str)
patients['subject_id'] = patients['subject_id'].astype(str)
sepsis_labels['subject_id'] = sepsis_labels['subject_id'].astype(str)

# Merge all data
df = admissions.merge(patients, on="subject_id", how="inner")
df = df.merge(sepsis_labels, on="subject_id", how="inner")

# Clean and prepare features
print("Cleaning and preparing features...")
df_clean = df.dropna(subset=['gender', 'anchor_age', 'sepsis']).copy()

# Convert data types properly
print("Converting data types...")
print(f"anchor_age dtype before: {df_clean['anchor_age'].dtype}")
print(f"Sample anchor_age values: {df_clean['anchor_age'].head()}")

# Convert anchor_age to numeric, handling any non-numeric values
df_clean['anchor_age'] = pd.to_numeric(df_clean['anchor_age'], errors='coerce')
print(f"anchor_age dtype after: {df_clean['anchor_age'].dtype}")

# Convert gender to numeric
df_clean['gender'] = df_clean['gender'].map({'M': 0, 'F': 1})

# Convert sepsis to numeric if it's not already
df_clean['sepsis'] = pd.to_numeric(df_clean['sepsis'], errors='coerce')

# Remove any rows with NaN values after conversion
df_clean = df_clean.dropna(subset=['gender', 'anchor_age', 'sepsis'])

# Check for valid age values
print(f"Age range: {df_clean['anchor_age'].min()} - {df_clean['anchor_age'].max()}")
print(f"Invalid ages (< 0 or > 120): {((df_clean['anchor_age'] < 0) | (df_clean['anchor_age'] > 120)).sum()}")

# Remove unrealistic ages
df_clean = df_clean[(df_clean['anchor_age'] >= 0) & (df_clean['anchor_age'] <= 120)]

X = df_clean[['gender', 'anchor_age']]
y = df_clean['sepsis']

print(f"\nFinal dataset shape: {df_clean.shape}")
print(f"Sepsis cases: {y.sum()} ({100*y.mean():.2f}%)")

# === 3. DATA EXPLORATION PLOTS ===
print("\nCreating exploratory data analysis plots...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Sepsis Dataset - Exploratory Data Analysis', fontsize=16, fontweight='bold')

# Plot 1: Class Distribution
ax1 = axes[0, 0]
sepsis_counts = y.value_counts()
colors = ['lightblue', 'salmon']
bars = ax1.bar(['No Sepsis', 'Sepsis'], sepsis_counts.values, color=colors, alpha=0.7, edgecolor='black')
ax1.set_title('Class Distribution', fontweight='bold')
ax1.set_ylabel('Count')
# Add value labels on bars
for bar, count in zip(bars, sepsis_counts.values):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01*height, f'{count}',
             ha='center', va='bottom', fontweight='bold')

# Plot 2: Age Distribution by Sepsis
ax2 = axes[0, 1]
age_no_sepsis = df_clean[df_clean['sepsis'] == 0]['anchor_age']
age_sepsis = df_clean[df_clean['sepsis'] == 1]['anchor_age']
ax2.hist([age_no_sepsis, age_sepsis], bins=20, alpha=0.7, 
         label=['No Sepsis', 'Sepsis'], color=['lightblue', 'salmon'], edgecolor='black')
ax2.set_title('Age Distribution by Sepsis Status', fontweight='bold')
ax2.set_xlabel('Age')
ax2.set_ylabel('Frequency')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Gender Distribution by Sepsis
ax3 = axes[0, 2]
gender_sepsis = pd.crosstab(df_clean['gender'].map({0: 'Male', 1: 'Female'}), 
                           df_clean['sepsis'].map({0: 'No Sepsis', 1: 'Sepsis'}))
gender_sepsis.plot(kind='bar', ax=ax3, color=['lightblue', 'salmon'], alpha=0.7, edgecolor='black')
ax3.set_title('Gender Distribution by Sepsis Status', fontweight='bold')
ax3.set_xlabel('Gender')
ax3.set_ylabel('Count')
ax3.legend(title='Sepsis Status')
ax3.tick_params(axis='x', rotation=0)

# Plot 4: Age vs Gender Scatter (colored by sepsis)
ax4 = axes[1, 0]
scatter_no_sepsis = ax4.scatter(df_clean[df_clean['sepsis'] == 0]['gender'] + np.random.normal(0, 0.02, sum(df_clean['sepsis'] == 0)),
                               df_clean[df_clean['sepsis'] == 0]['anchor_age'],
                               alpha=0.6, color='lightblue', label='No Sepsis', s=20)
scatter_sepsis = ax4.scatter(df_clean[df_clean['sepsis'] == 1]['gender'] + np.random.normal(0, 0.02, sum(df_clean['sepsis'] == 1)),
                            df_clean[df_clean['sepsis'] == 1]['anchor_age'],
                            alpha=0.8, color='salmon', label='Sepsis', s=30)
ax4.set_title('Age vs Gender (Colored by Sepsis)', fontweight='bold')
ax4.set_xlabel('Gender (0=Male, 1=Female)')
ax4.set_ylabel('Age')
ax4.set_xticks([0, 1])
ax4.set_xticklabels(['Male', 'Female'])
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: Sepsis Rate by Age Groups
ax5 = axes[1, 1]
try:
    # Ensure anchor_age is numeric and create age groups
    df_plot = df_clean.copy()
    df_plot['age_group'] = pd.cut(df_plot['anchor_age'], 
                                 bins=[0, 30, 50, 70, 100], 
                                 labels=['<30', '30-50', '50-70', '70+'],
                                 include_lowest=True)
    
    sepsis_rate_by_age = df_plot.groupby('age_group')['sepsis'].agg(['mean', 'count'])
    bars = ax5.bar(sepsis_rate_by_age.index, sepsis_rate_by_age['mean'], 
                   alpha=0.7, color='coral', edgecolor='black')
    ax5.set_title('Sepsis Rate by Age Group', fontweight='bold')
    ax5.set_xlabel('Age Group')
    ax5.set_ylabel('Sepsis Rate')
    ax5.tick_params(axis='x', rotation=45)
    
    # Add count labels
    for bar, count in zip(bars, sepsis_rate_by_age['count']):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.001, f'n={count}',
                 ha='center', va='bottom', fontsize=9)
                 
except Exception as e:
    print(f"Error creating age group plot: {e}")
    # Fallback: simple age histogram
    ax5.hist(df_clean['anchor_age'], bins=20, alpha=0.7, color='coral', edgecolor='black')
    ax5.set_title('Age Distribution', fontweight='bold')
    ax5.set_xlabel('Age')
    ax5.set_ylabel('Frequency')

# Plot 6: Correlation Heatmap
ax6 = axes[1, 2]
corr_data = df_clean[['gender', 'anchor_age', 'sepsis']].corr()
sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, ax=ax6, 
            square=True, linewidths=0.5)
ax6.set_title('Feature Correlation Matrix', fontweight='bold')

plt.tight_layout()
plt.show()

# === 4. TRAIN MODEL ===
if len(y.unique()) > 1 and y.sum() >= 5:
    print("\nTraining model...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    
    # === 5. MODEL EVALUATION PLOTS ===
    print("Creating model evaluation plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Model Evaluation Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Confusion Matrix
    ax1 = axes[0, 0]
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['No Sepsis', 'Sepsis'], 
                yticklabels=['No Sepsis', 'Sepsis'])
    ax1.set_title('Confusion Matrix', fontweight='bold')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    
    # Plot 2: ROC Curve
    ax2 = axes[0, 1]
    if len(y_test.unique()) > 1:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        ax2.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curve', fontweight='bold')
        ax2.legend(loc="lower right")
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Precision-Recall Curve
    ax3 = axes[0, 2]
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    ap_score = average_precision_score(y_test, y_pred_proba)
    ax3.plot(recall, precision, color='blue', lw=2,
             label=f'PR curve (AP = {ap_score:.3f})')
    ax3.axhline(y=y_test.mean(), color='red', linestyle='--', 
                label=f'Baseline (AP = {y_test.mean():.3f})')
    ax3.set_xlabel('Recall')
    ax3.set_ylabel('Precision')
    ax3.set_title('Precision-Recall Curve', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Feature Importance
    ax4 = axes[1, 0]
    feature_importance = clf.feature_importances_
    features = X.columns
    bars = ax4.bar(features, feature_importance, color='skyblue', alpha=0.7, edgecolor='black')
    ax4.set_title('Feature Importance', fontweight='bold')
    ax4.set_xlabel('Features')
    ax4.set_ylabel('Importance')
    # Add value labels
    for bar, importance in zip(bars, feature_importance):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.001, f'{importance:.3f}',
                 ha='center', va='bottom', fontweight='bold')
    
    # Plot 5: Prediction Distribution
    ax5 = axes[1, 1]
    ax5.hist(y_pred_proba[y_test == 0], bins=20, alpha=0.7, label='No Sepsis', 
             color='lightblue', edgecolor='black', density=True)
    ax5.hist(y_pred_proba[y_test == 1], bins=20, alpha=0.7, label='Sepsis', 
             color='salmon', edgecolor='black', density=True)
    ax5.set_xlabel('Predicted Probability')
    ax5.set_ylabel('Density')
    ax5.set_title('Distribution of Predicted Probabilities', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Model Performance Metrics Bar Chart
    ax6 = axes[1, 2]
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred)
    }
    
    if len(y_test.unique()) > 1:
        metrics['ROC AUC'] = roc_auc_score(y_test, y_pred_proba)
    
    bars = ax6.bar(metrics.keys(), metrics.values(), 
                   color=['lightcoral', 'lightblue', 'lightgreen', 'gold', 'plum'][:len(metrics)],
                   alpha=0.7, edgecolor='black')
    ax6.set_title('Model Performance Metrics', fontweight='bold')
    ax6.set_ylabel('Score')
    ax6.set_ylim(0, 1)
    ax6.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, (metric, value) in zip(bars, metrics.items()):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{value:.3f}',
                 ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # === 6. ADDITIONAL ANALYSIS PLOTS ===
    print("Creating additional analysis plots...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Additional Model Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Prediction Confidence vs Accuracy
    ax1 = axes[0]
    confidence = np.maximum(y_pred_proba, 1 - y_pred_proba)
    correct_predictions = (y_pred == y_test)
    
    # Bin by confidence levels
    confidence_bins = np.linspace(0.5, 1.0, 6)
    bin_centers = []
    bin_accuracies = []
    
    for i in range(len(confidence_bins) - 1):
        mask = (confidence >= confidence_bins[i]) & (confidence < confidence_bins[i+1])
        if mask.sum() > 0:
            bin_centers.append((confidence_bins[i] + confidence_bins[i+1]) / 2)
            bin_accuracies.append(correct_predictions[mask].mean())
    
    if bin_centers:
        ax1.plot(bin_centers, bin_accuracies, 'bo-', linewidth=2, markersize=8)
        ax1.plot([0.5, 1.0], [0.5, 1.0], 'r--', alpha=0.7, label='Perfect Calibration')
        ax1.set_xlabel('Prediction Confidence')
        ax1.set_ylabel('Actual Accuracy')
        ax1.set_title('Model Calibration', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Error Analysis by Age
    ax2 = axes[1]
    test_data = X_test.copy()
    test_data['y_true'] = y_test
    test_data['y_pred'] = y_pred
    test_data['error'] = (y_test != y_pred).astype(int)
    
    try:
        # Create age bins for error analysis
        age_bins = pd.cut(test_data['anchor_age'], bins=5, precision=0)
        error_rate_by_age = test_data.groupby(age_bins)['error'].mean()
        
        bars = ax2.bar(range(len(error_rate_by_age)), error_rate_by_age.values, 
                       color='lightcoral', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Age Groups')
        ax2.set_ylabel('Error Rate')
        ax2.set_title('Error Rate by Age Group', fontweight='bold')
        ax2.set_xticks(range(len(error_rate_by_age)))
        ax2.set_xticklabels([f'{int(interval.left)}-{int(interval.right)}' 
                            for interval in error_rate_by_age.index], rotation=45)
    except Exception as e:
        print(f"Error creating age-based error analysis: {e}")
        # Fallback: overall error rate
        overall_error_rate = test_data['error'].mean()
        ax2.bar(['Overall'], [overall_error_rate], color='lightcoral', alpha=0.7, edgecolor='black')
        ax2.set_title('Overall Error Rate', fontweight='bold')
        ax2.set_ylabel('Error Rate')
    
    # Plot 3: Learning Curve (simplified)
    ax3 = axes[2]
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_scores = []
    
    for train_size in train_sizes:
        n_samples = int(train_size * len(X_train))
        if n_samples > 10:  # Minimum samples needed
            clf_temp = RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced')
            clf_temp.fit(X_train[:n_samples], y_train[:n_samples])
            score = clf_temp.score(X_test, y_test)
            train_scores.append(score)
        else:
            train_scores.append(0)
    
    ax3.plot(train_sizes, train_scores, 'bo-', linewidth=2, markersize=6)
    ax3.set_xlabel('Training Set Size (fraction)')
    ax3.set_ylabel('Test Accuracy')
    ax3.set_title('Learning Curve', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # === 7. PRINT RESULTS ===
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Dataset size: {len(df_clean)} patients")
    print(f"Sepsis cases: {y.sum()} ({100*y.mean():.2f}%)")
    print(f"Test set size: {len(y_test)} patients")
    print(f"Test sepsis cases: {y_test.sum()} ({100*y_test.mean():.2f}%)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    if len(y_test.unique()) > 1:
        print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
        print(f"Average Precision Score: {average_precision_score(y_test, y_pred_proba):.4f}")
    
    print(f"\nFeature Importance:")
    for feature, importance in zip(X.columns, clf.feature_importances_):
        print(f"  {feature}: {importance:.4f}")

else:
    print("Cannot train model: insufficient positive cases or only one class present")
    print("Consider:")
    print("1. Checking if the ICD codes match your data format")
    print("2. Using ICD-9 codes if your data uses ICD-9")
    print("3. Expanding the list of sepsis-related ICD codes")