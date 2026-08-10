import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression #using this for classification
from sklearn.neighbors import KNeighborsClassifier #usinng this for classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
df = pd.read_csv('IRIS.csv')
df.head()
df.columns.tolist()
df
df.info()
df.isnull().sum()
df.filter(['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']).describe()
sns.pairplot(df, hue='species')
plt.title('Pairplot of IRIS dataset')
plt.xlabel('Features')
plt.ylabel('count')
plt.show()
# Split the data into features and target variable
X = df.drop(columns=['species']) 
y = df['species']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"train set size: {X_train.shape[0]}")
print(f"test set size: {X_test.shape[0]}")
# Train the Logistic Regression model
lr_model = LogisticRegression(max_iter=200)
lr_model.fit(X_train, y_train)
# Predict on the test set
lr_predictions = lr_model.predict(X_test)
# Evaluate the Logistic Regression model
lr_pred = lr_model.predict(X_test)
accuracy = accuracy_score(y_test, lr_pred)
print("Logistic Regression Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, lr_pred))
cm = confusion_matrix(y_test, lr_pred, labels=lr_model.classes_)
discm = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=lr_model.classes_)
discm.plot()
print("Confusion Matrix:\n", cm)
print(cm)
plt.title('Confusion Matrix')
plt.show()
# Unke samne ye data run karke dikhao
live_test = [[5.1, 3.5, 1.4, 0.2]] # Yeh Setosa ke real measurements hain
prediction = lr_model.predict(live_test)
print(f"Live Input Prediction: {prediction[0]}")