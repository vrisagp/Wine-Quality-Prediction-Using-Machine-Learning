import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import RFE


path = "/Users/vrisagpatel/Downloads/wine+quality/"


red_wine = pd.read_csv(path + "winequality-red.csv", sep=';')
white_wine = pd.read_csv(path + "winequality-white.csv", sep=';')


red_wine['type'] = 'red'
white_wine['type'] = 'white'


wine_data = pd.concat([red_wine, white_wine], ignore_index=True)


print(wine_data.head())


print(wine_data.info())
print(wine_data.describe())
print(wine_data['quality'].value_counts())


sns.countplot(data=wine_data, x='quality', hue='type', palette='pastel')
plt.title("Wine Quality Distribution (Red vs White)")
plt.xlabel("Quality Score")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(12, 8))
sns.heatmap(wine_data.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap of Wine Attributes")
plt.show()


wine_data['type'] = LabelEncoder().fit_transform(wine_data['type'])

X = wine_data.drop('quality', axis=1)
y = wine_data['quality']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC()
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

    print(f"\n🧠 {name}")
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))


X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)


rf_model = models["Random Forest"]
importances = rf_model.feature_importances_
features = X_scaled_df.columns


sorted_idx = importances.argsort()[::-1]


plt.figure(figsize=(10, 6))
sns.barplot(x=importances[sorted_idx], y=features[sorted_idx])
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.tight_layout()
plt.show()


selector = RFE(LogisticRegression(max_iter=1000), n_features_to_select=5)
X_train_rfe = selector.fit_transform(X_train, y_train)
X_test_rfe = selector.transform(X_test)


rfe_results = {}

for name, model in models.items():
    model.fit(X_train_rfe, y_train)
    y_pred_rfe = model.predict(X_test_rfe)
    
    acc_rfe = accuracy_score(y_test, y_pred_rfe)
    rfe_results[name] = acc_rfe
    
    print(f"\n🧠 {name} (with RFE)")
    print(f"Accuracy: {acc_rfe:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred_rfe))


print("\nPerformance with all features:")
for name, model in models.items():
    scores = cross_val_score(model, X_scaled, y, cv=3, scoring='accuracy', n_jobs=-1)
    print(f"{name}: Mean Accuracy = {scores.mean():.4f}, Std = {scores.std():.4f}")


X_selected = X.iloc[:, selector.support_]
print("\nPerformance with selected features:")
for name, model in models.items():
    scores = cross_val_score(model, X_selected, y, cv=3, scoring='accuracy', n_jobs=-1)
    print(f"{name}: Mean Accuracy = {scores.mean():.4f}, Std = {scores.std():.4f}")
    

rf = RandomForestClassifier(random_state=42)


param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}


grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)


best_rf = grid_search.best_estimator_


print(f"Best hyperparameters: {grid_search.best_params_}")


from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])


y_score = best_rf.predict_proba(X_test)

fpr = {}
tpr = {}
roc_auc = {}

for i in range(y_test_bin.shape[1]):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])


plt.figure(figsize=(8, 6))

for i in range(y_test_bin.shape[1]):
    plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} (area = {roc_auc[i]:0.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for Multiclass')
plt.legend(loc="lower right")
plt.show()