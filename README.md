#ED frequent users - RF Training section:

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RandomizedSearchCV, train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# JoinAll was created in data cleaning phase. It has been defined.
X = JoinAll.loc[:, JoinAll.columns != 'ERCOUNT']
y = JoinAll.loc[:, JoinAll.columns == 'ERCOUNT']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Apply SMOTE for balancing
os = SMOTE(random_state=0)
X_train_resampled, y_train_resampled = os.fit_resample(X_train, y_train)

# Define RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=1400, max_depth=26, max_features='auto')

# Hyperparameter tuning (RandomizedSearchCV)
param_grid = {
    'n_estimators': [int(x) for x in np.linspace(200, 2000, 10)],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [int(x) for x in np.linspace(100, 500, 11)] + [None]
}
rfc_random = RandomizedSearchCV(classifier, param_grid, n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)
rfc_random.fit(X_train_resampled, y_train_resampled)

# Print best hyperparameters
print(f"Best parameters: {rfc_random.best_params_}")

# Train the classifier with the best hyperparameters
best_classifier = rfc_random.best_estimator_

# Model evaluation
y_pred = best_classifier.predict(X_test)
print(classification_report(y_test, y_pred))

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, best_classifier.predict_proba(X_test)[:, 1])
plt.plot(fpr, tpr, label=f'Random Forest (AUC = {roc_auc_score(y_test, y_pred):.2f})')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

# Feature importance plot
feature_imp = pd.Series(best_classifier.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10, len(X.columns) * 0.6))
sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.tight_layout()
plt.show()
