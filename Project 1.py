############ Step 1
# Read Data
import pandas as pd
data = pd.read_csv("Project 1 Data.csv")

# Some quick look at the data
print(data.head())
print(data.columns)
print(data.count())
print(data.describe())


# #Removing missing Data
# data = data.dropna().reset_index(drop=True)

############ Step 2
# data.hist()

import matplotlib.pyplot as plt

x       = data['X']
y       = data['Y']
z       = data['Z']
step    = data['Step']

plt.scatter(x, step, color='red')
plt.ylabel('Step')
plt.xlabel('X')
plt.grid(True)
plt.show()

plt.scatter(y, step,color='blue')
plt.ylabel('Step')
plt.xlabel('Y')
plt.grid(True)
plt.show()

plt.scatter(z, step,color='green')
plt.ylabel('Step')
plt.xlabel('Z')
plt.grid(True)
plt.show()

#Create a new figure and add a 3D subplot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Plot the 3D scatter plot
ax.scatter(x, y, z)

# Set labels for the axes
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')

# Set a title for the plot
ax.set_title('3D Scatter Plot')

ax.legend(title='Total', bbox_to_anchor=(1.05, 1), loc='upper left')


## Another 3D scatter plot with colors
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

steps = data['Step'].unique()
cmap = plt.cm.get_cmap('Spectral', len(steps))

for i, s in enumerate(steps):
        subset = data[data['Step'] == s]
        
        ax.scatter(subset['X'], subset['Y'], subset['Z'], 
               color=cmap(i),        
               label=f'Step {s}',    
               marker='o', 
               alpha=0.8)
        
# Set labels for the axes
ax.set_xlabel('X-axis', fontsize=12)
ax.set_ylabel('Y-axis', fontsize=12)
ax.set_zlabel('Z-axis', fontsize=12)
ax.set_title('3D Scatter Plot of Maintenance Steps by Coordinates', fontsize=16)


ax.legend(title='Maintenance Step', bbox_to_anchor=(1.05, 1), loc='upper left')

# Display the plot
plt.show()


 ############ Step 3
import seaborn as sns


correlation_matrix = data.corr(method='pearson')

print(correlation_matrix)
print("r(X, Step):", correlation_matrix.loc['X', 'Step'])
print("r(Y, Step):", correlation_matrix.loc['Y', 'Step'])
print("r(Z, Step):", correlation_matrix.loc['Z', 'Step'])


plt.figure(figsize=(6, 5))
sns.heatmap(correlation_matrix, 
            annot=True,         # shorw the "r" value in the box
            linewidths=.5)      # Cell boundry
plt.title('Feature Correlation Matrix (Pearson)')
plt.show()


 ############ Step 4
 
# Data splitting


# # Normal Data splitting but this won't be used
# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(x, y,test_size = 0.2,random_state = 0)


from sklearn.model_selection import StratifiedShuffleSplit

# Define X and Y
X = data[['X', 'Y', 'Z']]
Y = data['Step']

splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)

for train_index, test_index in splitter.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

print("\n--- Data Split results ---")
print(f"Train Data Size: {X_train.shape}")
print(f"Test Data Size: {X_test.shape}")
print(f"Train Data Step Portion:\n{Y_train.value_counts(normalize=True).sort_index()}")
print(f"Test Data Step Portion:\n{Y_test.value_counts(normalize=True).sort_index()}")

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC 
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint

# Dictionary for saving optimized models
optimized_models = {}

# --- Three GridSearchCV Model---

# KNN (K-Nearest Neighbors)
print("\n--- KNN Grid Search Started ---")
knn_model = KNeighborsClassifier()
knn_param_grid = {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
knn_grid_search = GridSearchCV(knn_model, knn_param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
knn_grid_search.fit(X_train, Y_train)
optimized_models['KNN_Grid'] = knn_grid_search.best_estimator_
print(f"KNN Best Hyperparameter: {knn_grid_search.best_params_}")


# Decision Tree
print("\n--- Decision Tree Grid Search Started ---")
dt_model = DecisionTreeClassifier(random_state=0)
dt_param_grid = {'max_depth': [5, 10, 15], 'min_samples_split': [2, 5]}
dt_grid_search = GridSearchCV(dt_model, dt_param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
dt_grid_search.fit(X_train, Y_train)
optimized_models['DT_Grid'] = dt_grid_search.best_estimator_
print(f"DT Best Hyperparameter: {dt_grid_search.best_params_}")


# SVC (Support Vector Classifier)
print("\n--- SVC Grid Search Started ---")
svc_model = SVC(random_state=0)
svc_param_grid = {'C': [1, 10], 'gamma': ['scale', 'auto']}
svc_grid_search = GridSearchCV(svc_model, svc_param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
svc_grid_search.fit(X_train, Y_train)
optimized_models['SVC_Grid'] = svc_grid_search.best_estimator_
print(f"SVC Best Hyperparameter: {svc_grid_search.best_params_}")


# --- RandomizedSearchCV Using Random Forest) ---

# Random Forest
print("\n--- Random Forest Randomized Search Started---")
rf_model = RandomForestClassifier(random_state=0)
rf_param_dist = {'n_estimators': randint(low=50, high=200), 'max_depth': randint(low=5, high=20)}

rf_random_search = RandomizedSearchCV(rf_model, rf_param_dist, n_iter=10, cv=5, random_state=0, scoring='f1_macro', n_jobs=-1)
rf_random_search.fit(X_train, Y_train)
optimized_models['RF_Random'] = rf_random_search.best_estimator_
print(f"RF Best Hyperparameter (Randomized): {rf_random_search.best_params_}")


############ Step 5

from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix

results = {}

print("\n--- Evaluating Model Performance ---")

def evaluate_model(model, X_test, Y_test):
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    precision = precision_score(Y_test, Y_pred, average='macro', zero_division=0)
    f1 = f1_score(Y_test, Y_pred, average='macro', zero_division=0)
    return accuracy, precision, f1, Y_pred

# Save Model results
for name, model in optimized_models.items():
    acc, prec, f1, Y_pred = evaluate_model(model, X_test, Y_test)
    results[name] = {
        'Accuracy': acc, 
        'Precision (Macro)': prec, 
        'F1 Score (Macro)': f1, 
        'Predictions': Y_pred
    }
    print(f"| {name:<10} | Accuracy: {acc:.4f} | Precision: {prec:.4f} | F1 Score: {f1:.4f} |")

# Visualize the results using DataFrame
results_df = pd.DataFrame({
    k: {k2: v2 for k2, v2 in v.items() if k2 != 'Predictions'} 
    for k, v in results.items()
}).T

print("\n--- Fnal Model Summary (Based on F1 Score with Descending Order) ---")
print(results_df.sort_values(by='F1 Score (Macro)', ascending=False))

# --- Visualizing final Model ---

# Pick the highest F1 Score Model
best_model_name = results_df['F1 Score (Macro)'].idxmax()
final_model = optimized_models[best_model_name]
final_predictions = results[best_model_name]['Predictions']

print("\n========================================================")
print(f" Final Selection: {best_model_name} (F1 Score: {results_df.loc[best_model_name, 'F1 Score (Macro)']:.4f})")
print("========================================================")

# Using Confusion Matirx on the final Model
cm = confusion_matrix(Y_test, final_predictions)
cm_df = pd.DataFrame(cm, 
                     index=[f'Actual Step {s}' for s in final_model.classes_], 
                     columns=[f'Predicted Step {s}' for s in final_model.classes_])

plt.figure(figsize=(12, 10))
sns.heatmap(cm_df, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            cbar=True)
plt.xlabel('Predicted Step', fontsize=14)
plt.ylabel('True Step', fontsize=14)
plt.title(f'Confusion Matrix for Selected Model: {best_model_name}', fontsize=16)
plt.show()


############ Step 6

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

# 1. Define Base Estimators
# Bring the best Model(RF_Random, SVC_Grid, KNN_Grid)
base_estimators = [
    ('rf', optimized_models['RF_Random']), 
    ('svc', optimized_models['SVC_Grid']),
    ('knn', optimized_models['KNN_Grid'])
]

# 2. Define Final Estimator Using LogisticeRegression
final_estimator = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=0)


# 3. StackingClassifier Training
# cv=5: 내부적으로 5-Fold 교차 검증을 사용하여 기본 모델의 예측을 메타 모델의 입력으로 만듭니다.
stack_model = StackingClassifier(
    estimators=base_estimators, 
    final_estimator=final_estimator,
    cv=5, 
    n_jobs=-1  # using Full CPU Core
)

print("\n--- Step 6: Stacking Model Training Started ---")
stack_model.fit(X_train, Y_train)
print("Stacking Model Training Completed!!.")

# 4. Assess Stacking Model
Y_pred_stack = stack_model.predict(X_test)
stack_acc = accuracy_score(Y_test, Y_pred_stack)
stack_prec = precision_score(Y_test, Y_pred_stack, average='macro', zero_division=0)
stack_f1 = f1_score(Y_test, Y_pred_stack, average='macro', zero_division=0)

print("\n--- Stacking Model Performance Result ---")
print(f"| Stacking   | Accuracy: {stack_acc:.4f} | Precision: {stack_prec:.4f} | F1 Score: {stack_f1:.4f} |")

# 5. Compare the best Model from Step 5 and Stacking
best_f1_step5 = results_df.loc[best_model_name, 'F1 Score (Macro)']

print("\n--- Step 5 vs Step 6 Performance Comparison ---")
print(f"Step 5 Best Model ({best_model_name}) F1 Score: {best_f1_step5:.4f}")
print(f"Step 6 Stacking Model F1 Score: {stack_f1:.4f}")

if stack_f1 > best_f1_step5:
    print(f" Stacking Model이 Step 5 Best Model F1 Score is higher for about {stack_f1 - best_f1_step5:.4f} . Accuracy Increased!")
else:
    print(f" There were no increased of accuracy of Stacking Model (delta: {stack_f1 - best_f1_step5:.4f})")

############ Step 7

import joblib
import numpy as np

# 1. Best Model Selection (Step 5 Best vs Step 6 stacked)
if 'stack_f1' in locals() and stack_f1 > best_f1_step5:
    true_final_model = stack_model
    final_model_name = "Stacking_Model"
else:
    true_final_model = final_model # From Step 5
    final_model_name = best_model_name

print("\n--- Step 7: Final Model Packaging---")
print(f"Final selected Model: {final_model_name}")

# 2. Save into Joblib Format
model_filename = f"{final_model_name}_Final_Model.joblib"
joblib.dump(true_final_model, model_filename)
print(f"Model {model_filename} has been succesfully saved.")

# 3. Data Point
new_coordinates = np.array([
    [9.375, 3.0625, 1.51], 
    [6.995, 5.125, 0.3875], 
    [0, 3.0625, 1.93], 
    [9.4, 3, 1.8], 
    [9.4, 3, 1.3]
])

print("\n--- Data to predict ---")
for i, coord in enumerate(new_coordinates):
    print(f"coordinates {i+1}: {coord}")


# 4. Call saved model and run
loaded_model = joblib.load(model_filename)
new_predictions = loaded_model.predict(new_coordinates)

print("\n--- Result of the Prediction ---")

prediction_results = []
for coord, step in zip(new_coordinates, new_predictions):
    prediction_results.append(f"Coordinate {coord} => Predicted as Step: {step}")
    
for result in prediction_results:
    print(result)

print("\n Step 7 Finished !!.")


























