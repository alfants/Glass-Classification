import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import make_pipeline
import pickle
import numpy as np

# Load The CSV File
data = pd.read_csv("glass.csv")

data.head()

# Select Independent & Dependent Variabel
X = data.drop(columns="Type", axis=1)
Y = data["Type"]

# Random Sampling
resamp = RandomOverSampler()
balX, balY = resamp.fit_resample(X, Y)

# Data Normalization
scaler = StandardScaler()
standardized = scaler.fit_transform(balX)

# Data Dimension Reduction
n_components = 9  # Ganti dengan jumlah komponen yang diinginkan
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(standardized)

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Tentukan parameter yang akan diuji dalam Grid Search
param_grid = {'criterion':['gini', 'entropy', 'log_loss'], 'max_depth':np.arange(1,10)}

# Inisialisasi Grid Search
grid_search = GridSearchCV(clf, param_grid=param_grid, cv=5, scoring='accuracy')
# Latih model dengan data pelatihan
grid_search.fit(X_pca, balY)

print("Best Parameters (Accuracy):", grid_search.best_params_)
print("Best Score (Accuracy):", grid_search.best_score_)

# Make Pipeline
pipe = make_pipeline(RandomOverSampler(), StandardScaler(), PCA(n_components=9), DecisionTreeClassifier(criterion='gini', max_depth=9))

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

# Make pickle file of our model
pickle.dump(pipe, open("model.pkl", "wb"))