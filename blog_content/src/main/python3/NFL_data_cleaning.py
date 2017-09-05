# general necessities
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# cleaning the data
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# linear classifier models
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# nonlinear classification models
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB

# ensemble classifiers
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

# Grid Search for evaluating multiple parameters of the models while testing them
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# misc
from collections import OrderedDict

class mlOptimizer():
    def __init__(self, df, y_col, global_random_state=0):
        self.X = df.drop(y_col, 1)
        self.y = df[y_col]
        feature_number = len(df.columns) - 1
        self.global_random_state = global_random_state
        self.data = OrderedDict([('Model', []),
                                 ('Training Score', []),
                                 ('Test Score', []),
                                 ('Fit Time', []),
                                 ('Best Model Params', [])])
        self.classification_dict = dict(LogisticRegression={
            'name': 'Logistic Regression',
            'model': LogisticRegression(),
            'params': {
                'model__warm_start': [True, False],
                'model__C': [0.01, 0.1, 1, 10, 100]
            }
        }, LDA={
            'name': 'LDA',
            'model': LDA(),
            'params': {
                'model__solver': ['svd', 'lsqr', 'eigen']
            }
        }, GaussianNB={
            'name': 'Gaussian Naive Bayes',
            'model': GaussianNB(),
            'params': {}
        }, MultinomialNB={
            'name': 'Multinomial Naive Bayes',
            'model': MultinomialNB(),
            'params': {}
        }, QDA={
            'name': 'QDA',
            'model': QDA(),
            'params': {}
        }, DecisionTreeClassifier={
            'name': 'Decision Tree',
            'model': DecisionTreeClassifier(),
            'params': {
                'model__min_samples_split': [2, 5, 10],
                'model__min_samples_leaf': [2, 5, 10]
            }
        }, SVC={
            'name': 'RBF SVC',
            'model': SVC(),
            'params': {
                'model__C': [0.1, 1, 10],
                'model__degree': [2, 3, 4, 5],
                'model__gamma': [0.1, 1, 10],
                'model__kernel': ['rbf']
            }
        }, KNeighborsClassifier={
            'name': 'K Neighbors Classifier',
            'model': KNeighborsClassifier(),
            'params': {
                'model__n_neighbors': list(range(1, feature_number + 1))
            }
        }, RandomForest={
            'name': 'Random Forest',
            'model': RandomForestClassifier(),
            'params': {
                "model__max_depth": [3, None],
                "model__max_features": list(range(1, feature_number, 2)),
                "model__min_samples_split": [5, 10, 15],
                "model__min_samples_leaf": list(range(1, feature_number, 2)),
                "model__bootstrap": [True, False],
                "model__criterion": ["gini", "entropy"]
            }
        }, GradientBoostingClassifier={
            'name': 'Gradient Boosting Classifier',
            'model': GradientBoostingClassifier(),
            'params': {
                'model__loss': ['deviance', 'exponential'],
                'model__n_estimators': [50],
                'model__min_samples_leaf': [10, 15],
                'model__min_samples_split': [5, 10, 15],
                'model__max_features': [feature_number, feature_number - 1, feature_number - 2],
                'model__subsample': [0.5, 0.6, 0.7]
            }
        }, AdaBoostClassifier={
            'name': 'Ada Boost Classifier',
            'model': AdaBoostClassifier(),
            'params': {
                'model__n_estimators': list(range(100, 300, 50)),
                'model__learning_rate': list(np.arange(0.7, 1.1, 0.1))
            }
        })

    def create_data(self, df, y_col, test_percentage, bootstrap=False):
        """
        Splits a dataframe into x and y columns, and train and test sets
        :param bootstrap: whether to use bootstrap or not. default=False
        :param df: dataframe to split
        :param y_col: columns of Y values, string
        :param test_percentage: percentage of the data to use for testing
        :return: list of X_train, y_train, X_test, y_test
        """

        # maybe put in a check for consistent values in here in the future, but not right now, I'm tired
        x = df.drop(y_col, 1)
        y = df[y_col]

        return train_test_split(x, y, test_size=test_percentage,
                                random_state=self.global_random_state)
        # return data_suite

    def import_model(self, clf, clf_name):
        """
        Imports the model parameters to the classification dictionary
        :param clf: classification model
        :param clf_name: model name
        """

        if 'pca' in clf.best_estimator_.named_steps:
            clf_name = str(clf.best_estimator_.named_steps['pca'].n_components_) + " Component PCA + " + clf_name
        clf_df = pd.DataFrame(clf.cv_results_).sort_values('rank_test_score')
        test_score = clf_df['mean_test_score'].values[0]
        train_score = clf_df['mean_train_score'].values[0]
        fit_time = clf_df['mean_fit_time'].values[0]
        self.data['Model'].append(clf_name)
        self.data['Test Score'].append(test_score)
        self.data['Training Score'].append(train_score)
        self.data['Fit Time'].append(fit_time)
        self.data['Best Model Params'].append(clf.best_estimator_.named_steps['model'].get_params())

    def view(self):
        return pd.DataFrame(self.data).sort_values('Test Score', ascending=False)

    def clear(self):
        self.data = OrderedDict([('Model', []),
                                 ('Training Score', []),
                                 ('Test Score', []),
                                 ('Fit Time', []),
                                 ('Best Model Params', [])])
        print("Ranking Cleared.")

    def plot(self):
        """
        Creates and displays a plot of the Classification Models' Test and Training Accuracies, as well as Fit Time.
        """

        index = [v for v in self.data['Model']]
        plotting_df = pd.DataFrame(self.data, index=index).sort_values('Test Score', ascending=False)
#         mms = MinMaxScaler()
#         plotting_df['Normalized Fit Time'] = mms.fit_transform(plotting_df['Fit Time'].values.reshape(-1, 1))
#         plotting_df.drop(['Model', 'Fit Time'], 1, inplace=True)

        # plotting the data
        ax = plotting_df[['Test Score', 'Training Score']].plot(figsize=(15, 8),
                                                                fontsize=15, cmap='coolwarm', rot=90)
        # setting the text value parameters
        text_x = [-0.20, 0, 0.15]
        model_num = 0
        for y_values in plotting_df[['Test Score', 'Training Score']].values:
            for index, int_val in enumerate(y_values):
                # y_position = int_val - 0.02
                # if int_val < 0.1:
                y_position = int_val + 0.08
                ax.text(text_x[index] + model_num, y_position, str(round(int_val, 4)), color='black', fontweight='bold',
                        rotation=90)
            model_num += 1

        plt.title('Classification Model Comparison', size=20)
        plt.ylabel('Accuracy', size=15)
        plt.show()

    def optimize_model(self, model, pca=False):
        """
        Runs all the linear or nonlinear models, with PCA preprocessing on or off, and writes them to the
        best classification model dictionary
        :param model: classification model to use as sklearn method, string
        :param pca: whether to use pca or not, boolean
        """

        model_name = self.classification_dict[model]['name']
        model_instance = self.classification_dict[model]['model']
        model_params = self.classification_dict[model]['params'].copy()
        pipeline_steps = [('model', model_instance)]

        if pca:
            pca_clf = PCA()
            pipeline_steps.insert(0, ('pca', pca_clf))
            model_params['pca__n_components'] = [3, 4, 5, 6]
        pipe = Pipeline(steps=pipeline_steps)
        grid = GridSearchCV(pipe, model_params, verbose=1)
        grid.fit(self.X, self.y)
        self.import_model(grid, model_name)
