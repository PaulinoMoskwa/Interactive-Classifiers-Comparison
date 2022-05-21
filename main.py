###-----------------------------------------------------------------------------------------
### Libraries
###-----------------------------------------------------------------------------------------
import streamlit as st
import webbrowser
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Scikit-Learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.datasets        import make_moons, make_circles, make_blobs
from sklearn.inspection      import DecisionBoundaryDisplay

# Classifiers
from sklearn.neural_network           import MLPClassifier
from sklearn.neighbors                import KNeighborsClassifier
from sklearn.svm                      import SVC
from sklearn.gaussian_process         import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree                     import DecisionTreeClassifier
from sklearn.ensemble                 import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes              import GaussianNB
from sklearn.discriminant_analysis    import QuadraticDiscriminantAnalysis
from sklearn.linear_model             import LogisticRegression

###-----------------------------------------------------------------------------------------
### Title
###-----------------------------------------------------------------------------------------
#st.set_page_config(layout="wide")
if st.button('Hello 👋'):
        webbrowser.open_new_tab('https://paulinomoskwa.github.io/Hello/')
st.markdown('# Scikit-Learn Classifiers 🕵️‍♂️')
st.markdown('''Choose the dataset you want to use on the left sidebar and adjust the settings.\\
    Check out how Scikit-Learn classifiers perform! 🤓''')
st.markdown('')

###-----------------------------------------------------------------------------------------
### Classifiers
###-----------------------------------------------------------------------------------------
names = [
    "Logistic L1",
    "Logistic L2",
    "K-Nearest Neighbors",
    "Linear SVM",    
    "RBF SVM",
    "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
]

classifiers = [
    LogisticRegression(penalty='l1', max_iter=300, solver='liblinear'),
    LogisticRegression(penalty='l2', C=100, max_iter=300, solver='liblinear'),
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
]

###-----------------------------------------------------------------------------------------
### Dataset
###-----------------------------------------------------------------------------------------
col1, col2, col3 = st.sidebar.columns([1, 15, 1])
with col2:
    st.markdown('## Make your choices! ✍️')
    dataset_choice = st.selectbox('Select the dataset:', ('Blobs', 'Moons', 'Circles', 'Close Circles'))
    noise          = st.slider('Select the noise:', 0.05, 0.4)
    n_samples      = st.slider('Select the number of samples:', 50, 300)

# Dataset choice
if dataset_choice == 'Moons':
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
elif dataset_choice == 'Circles':
    X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=42)
elif dataset_choice == 'Blobs':
    X, y = make_blobs(n_samples=n_samples, centers=2, cluster_std=1.0 + noise *10, shuffle=False, random_state=42)
elif dataset_choice == 'Close Circles':
    X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.7, random_state=42)

# Dataset fit
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Visualization purposes
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

# Colors
cm = plt.cm.RdYlGn
cm_bright = ListedColormap(["#c13639", "#449e48"])

# Plot the training points
fig_1 = plt.figure()
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="white", linewidth=.4, s=80);

# Plot the testing points (lighter than the training)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.4, edgecolors="white", linewidth=.4, s=80);
plt.xlim(x_min, x_max);
plt.ylim(y_min, y_max);
plt.xticks([])
plt.yticks([])
plt.axis('off')

# Visualize the plot
col4, col5, col6 = st.sidebar.columns([1, 8, 1])
with col5:
    st.pyplot(fig_1)

###-----------------------------------------------------------------------------------------
### Classifiers and Boundaries
###-----------------------------------------------------------------------------------------
i = 1

n_rows = 3
n_cols = 4

fig_2, ax = plt.subplots(figsize=(12,8))

for name, clf in zip(names, classifiers):
    ax = plt.subplot(n_rows, n_cols, i)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    DecisionBoundaryDisplay.from_estimator(clf, X, cmap=cm, alpha=0.8, ax=ax, eps=0.5)

    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="white", linewidth=.4, s=20)
    
    # Plot the testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, edgecolors="white", linewidth=.4, alpha=0.6, s=20)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(name)
    ax.text(x_max - 0.3, y_min + 0.3, ("Score: %.2f" % score).lstrip("0"), size=9, horizontalalignment="right")
    i += 1

plt.subplots_adjust(hspace=.3, wspace=0.2)
st.pyplot(fig_2)