# AIML
31)
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
print(f"Test accuracy: {model.evaluate(x_test, y_test)[1]:.4f}")

plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.legend(), plt.show()




 

30)
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])

model = Sequential([
    Dense(8, activation='relu', input_shape=(2,)),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(X, y, epochs=500, verbose=0)

print("Result:", model.predict(np.array([[1, 0]])))
plt.plot(y, 'ro-', label='Actual')
plt.plot(model.predict(X), 'bs--', label='Predicted')
plt.legend(), plt.grid(), plt.show()

 

29)
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import ExpectationMaximization
import matplotlib.pyplot as plt
import networkx as nx

# Dataset
data = pd.DataFrame({
    'Rain':      [1, 0, 1, 1, 0, 1, 1, 0, 1, 1],
    'Sprinkler': [1, 0, 1, 1, 0, 1, 0, 0, 0, 1],
    'GrassWet':  [1, 0, 1, 1, 0, 1, 1, 0, 0, 1]
}, dtype="int")

# Define model structure
model = BayesianNetwork([
    ('Rain', 'Sprinkler'),
    ('Rain', 'GrassWet'),
    ('Sprinkler', 'GrassWet')
])

# Learn parameters using EM
em = ExpectationMaximization(model, data)
cpds = em.get_parameters()

# Print learned CPDs
for cpd in cpds:
    print(cpd)

# Plot network
nx.draw(model, with_labels=True, node_size=2000, node_color='lightgreen', font_size=12, arrowsize=20)
plt.title("Bayesian Network Structure")
plt.show()

 

28)
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

data = np.array([[100], [200], [300], [400]])
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)

print("Scaled for LSTM:", scaled.flatten())

plt.plot(data.flatten(), label='Original')
plt.plot(scaled.flatten(), label='Scaled (0–1)')
plt.title("Data Scaling for LSTM")
plt.legend()
plt.grid()
plt.show()

 
27)
import tensorflow as tf
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

X, y = load_diabetes(return_X_y=True)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])
model.compile('adam', 'mse')
model.fit(X_train, y_train, epochs=100, verbose=0)

print("Loss:", model.evaluate(X_test, y_test, verbose=0))
pred = model.predict(X_test).flatten()

plt.scatter(y_test, pred, c='purple')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Prediction vs Actual - Diabetes")
plt.grid()
plt.show()

Loss: 3000.45 
Graph need to import
26)
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])

model = tf.keras.Sequential([
  tf.keras.layers.Dense(16, activation='relu', input_shape=(2,)),
  tf.keras.layers.Dense(8, activation='relu'),
  tf.keras.layers.Dense(1)
])

model.compile('adam', 'mse')
model.fit(X, y, epochs=100, verbose=0)

print("Prediction for [1,1]:", model.predict([[1,1]]))
plt.plot(y, 'ro-', label='Actual')
plt.plot(model.predict(X), 'bo--', label='Predicted')
plt.legend();
 plt.show();

 

25)
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

model = DiscreteBayesianNetwork([('Age', 'Risk'), ('Cholesterol', 'Risk')])
model.add_cpds(
    TabularCPD('Age', 2, [[0.6], [0.4]]),
    TabularCPD('Cholesterol', 2, [[0.7], [0.3]]),
    TabularCPD('Risk', 2,
               [[0.9, 0.6, 0.5, 0.2],
                [0.1, 0.4, 0.5, 0.8]],
               evidence=['Age', 'Cholesterol'], evidence_card=[2, 2])
)
infer = VariableElimination(model)
print(infer.query(['Risk'], evidence={'Age': 1, 'Cholesterol': 1}))


 

24)

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

X = np.array([[20,100],[22,110],[35,300],[40,310],[60,500]])
kmeans = KMeans(2, n_init=10).fit(X)

print("Customer Segment Labels:", kmeans.labels_)
plt.scatter(X[:,0], X[:,1], c=kmeans.labels_, cmap='viridis', s=100)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=50, c='k', marker='X', label='Centroids')
plt.title("Customer Segmentation using KMeans")
plt.xlabel("Age"); 
plt.ylabel("Spending")
plt.legend();
 plt.grid();
 plt.show()

graph needed to import

23)
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

X = [[1,20],[2,15],[3,5],[4,0]]
y = [1,1,0,0]
model = VotingClassifier([
    ('nb', GaussianNB()),
    ('lr', LogisticRegression()),
    ('dt', DecisionTreeClassifier())
], voting='hard').fit(X, y)

pred = model.predict([[2,10]])[0]
print("User Type Prediction:", pred)

for x, label in zip(X, y):
    plt.scatter(*x, c='red' if label else 'blue')
plt.scatter(2, 10, c='green', marker='x', s=100, label='New Data')
plt.title("Voting Classifier - User Segmentation")
plt.xlabel("Feature 1"); plt.ylabel("Feature 2")
plt.legend(); plt.grid(); plt.show()

 

22)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt

X = [[1,200,5],[5,50,1],[0,500,10],[7,20,0]]
y = [1,0,1,0]

scaler = StandardScaler().fit(X)
X_s = scaler.transform(X)
model = SVC().fit(X_s, y)

pred = model.predict(scaler.transform([[2,150,3]]))[0]
print("Purchase Made:", pred)

colors = ['red' if i==0 else 'green' for i in y]
plt.scatter([x[0] for x in X], [x[1] for x in X], c=colors)
plt.title("Customer Data (Red: No, Green: Yes)")
plt.xlabel("Feature 1"); plt.ylabel("Feature 2")
plt.grid(True)
plt.show()

 
21)
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

X = [[100,50,30],[80,70,20],[90,40,60],[60,100,30]]
y = ['A','B','A','B']

model = RandomForestClassifier().fit(X, y)
pred = model.predict([[95,60,20]])[0]
print("Predicted Majority Winner:", pred)

labels = ['A','B']
counts = [y.count('A'), y.count('B')]
plt.bar(labels, counts, color=['orange','cyan'])
plt.title('Vote Distribution Before Prediction')
plt.ylabel('Votes Count')
plt.show()
 

20)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

data = {
    'age': [45,50,65,38,52,61],
    'cholesterol': [230,250,300,180,240,310],
    'blood_pressure': [130,140,150,120,135,155],
    'heart_rate': [150,145,130,170,160,125],
    'risk': [1,1,1,0,0,1]
}

df = pd.DataFrame(data)
X_train, X_test, y_train, y_test = train_test_split(df.drop('risk', axis=1), df['risk'], test_size=0.3, random_state=1)

clf = DecisionTreeClassifier().fit(X_train, y_train)
print(f"Accuracy: {clf.score(X_test, y_test)*100:.2f}%")

plt.figure(figsize=(10,6))
plot_tree(clf, feature_names=X_train.columns, class_names=["Low Risk","High Risk"], filled=True)
plt.title("Simple Heart Attack Risk Decision Tree")
plt.show()

 
19)

from sklearn.linear_model import LinearRegression
import numpy as np, matplotlib.pyplot as plt

X = np.array([[1],[2],[3],[4]])
y = np.array([100,120,150,170])

model = LinearRegression().fit(X, y)
pred = model.predict([[5]])[0]
print(f"Predicted customers in month 5: {int(pred)}")

plt.scatter(X, y, c='blue', label='Actual')
plt.plot(X, model.predict(X), c='green', label='Regression Line')
plt.scatter(5, pred, c='red', label='Prediction')
plt.xlabel('Month'); plt.ylabel('Customers')
plt.title('Customer Growth Prediction')
plt.legend(); plt.grid(True)
plt.show()
 
18)
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt

model = DiscreteBayesianNetwork([('A','B'),('B','C')])
model.add_cpds(
    TabularCPD('A', 2, [[0.5], [0.5]]),
    TabularCPD('B', 2, [[0.7,0.2],[0.3,0.8]], evidence=['A'], evidence_card=[2]),
    TabularCPD('C', 2, [[0.9,0.4],[0.1,0.6]], evidence=['B'], evidence_card=[2])
)

infer = VariableElimination(model)
probs = [infer.query(['C'], evidence={'A': i}).values[1] for i in range(2)]
for i, p in enumerate(probs): print(f"P(C=1 | A={i}): {p:.2f}")

plt.bar(['A=0','A=1'], probs, color='skyblue')
plt.title('P(C=1 | A)'); plt.ylabel('Probability'); plt.show()
 
17)
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

data = ["Free money now", "Hi friend", "Earn cash fast", "Hello there", "Cheap loan offer", "Let's meet"]
labels = [1, 0, 1, 0, 1, 0]

X = CountVectorizer().fit_transform(data).toarray()
X_train, X_test, y_train, y_test = train_test_split(X, labels, random_state=42)

model = GaussianNB().fit(X_train, y_train)
print("Accuracy:", model.score(X_test, y_test))

ConfusionMatrixDisplay.from_predictions(y_test, model.predict(X_test), display_labels=["Not Spam", "Spam"])
plt.title("Confusion Matrix - Spam Detection")
plt.show()


graph needed to be import
16)
def astar_logo(goal_logo="🎾"): 
    from queue import PriorityQueue 
    nodes = ["⚽", "🏀", "🎾", "🏈"] 
    h = {"⚽": 3, "🏀": 2, "🎾": 0, "🏈": 1} 
    graph = {
        "⚽": ["🏀", "🏈"], 
        "🏀": ["🎾"], 
        "🎾": [], 
        "🏈": ["🎾"]
    } 
    queue = PriorityQueue() 
    queue.put((h["⚽"], ["⚽"])) 
    while not queue.empty(): 
        cost, path = queue.get() 
        current = path[-1] 
        if current == goal_logo: 
            return path 
        for neighbor in graph.get(current, []): 
            queue.put((cost - h[current] + h[neighbor] + 1, path + [neighbor])) 

print("Path to:", astar_logo())

output:
Path to: ['⚽', '🏈', '🎾']
