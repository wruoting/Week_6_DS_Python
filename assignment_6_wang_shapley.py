import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

def split_by_feature_and_flower(df_iris, flower, marginal_contribution_feature=None):
    '''
    df_iris: Data frame of iris dataset
    returns: A dataframe with classification data, and a dataframe with output data
    '''
    # Split by flower
    df_iris['Binary Classification'] = [1 if row == flower else 0 for row in df_iris['Classification']]
    binary_classification_df = df_iris['Binary Classification'].copy()
    df_new_iris = df_iris.copy()
    if marginal_contribution_feature is not None:
        df_new_iris.drop(columns=[marginal_contribution_feature, 'Classification', 'Binary Classification'], inplace=True)
    else:
        df_new_iris.drop(columns=['Classification', 'Binary Classification'], inplace=True)

    return df_new_iris, binary_classification_df


def main():
    ticker='WMT'
    file_name = '{}_weekly_return_volatility.csv'.format(ticker)
    file_name_self_labels = 'WMT_Labeled_Weeks_Self.csv'
    iris_dataset = 'iris.data'

    # Read from that file for answering our questions
    df = pd.read_csv(file_name, encoding='ISO-8859-1')
    df_2018 = df[df['Year'] == 2018]
    df_2019 = df[df['Year'] == 2019]
    scaler = StandardScaler()
    X_2018 = df_2018[['mean_return', 'volatility']].values
    # Need to scale the training data
    X_2018_Scaled = scaler.fit_transform(X_2018)
    Y_2018 = df_2018[['Classification']].values
    split_2018 = np.split(X_2018_Scaled, 2, axis=1)

    X_2019 = df_2019[['mean_return', 'volatility']].values
    X_2019_Scaled = scaler.fit_transform(X_2019)
    Y_2019 = df_2019[['Classification']].values
    split_2019 = np.split(X_2019_Scaled, 2, axis=1)

    # For Question 1, in lecture, the professor asked us to just do the contributions of mu and sigma on 
    # logistic regression and KNN without the linear model
    print('Question 1')
    print('Contributions KNN Classifier')
    knn_classifier = KNeighborsClassifier(n_neighbors=5, p=2)
    knn_classifier.fit(X_2018_Scaled, Y_2018.ravel())
    prediction = knn_classifier.predict(X_2019_Scaled)
    accuracy = np.subtract(100, np.round(np.multiply(np.mean(prediction != Y_2019.T), 100), 2))

    # Fit 2018 data remove sigma
    knn_classifier.fit(split_2018[0], Y_2018.ravel())
    prediction_remove_sigma = knn_classifier.predict(split_2019[0])
    accuracy_remove_sigma = np.subtract(100, np.round(np.multiply(np.mean(prediction_remove_sigma != Y_2019.T), 100), 2))

    # Fit 2018 data remove mean
    knn_classifier.fit(split_2018[1], Y_2018.ravel())
    prediction_remove_mean = knn_classifier.predict(split_2019[1])
    accuracy_remove_mean = np.subtract(100, np.round(np.multiply(np.mean(prediction_remove_mean != Y_2019.T), 100), 2))

    # Calculate contributions
    contribution_mean = accuracy - accuracy_remove_mean
    contribution_sigma = accuracy - accuracy_remove_sigma
    
    print('Sigma\tMean')
    print('{}%\t{}%'.format(contribution_sigma, contribution_mean))

    print('Contributions Logistic Regression')
    logisticRegression = LogisticRegression()
    logisticRegression.fit(X_2018_Scaled, Y_2018.ravel())
    predict_2019 = logisticRegression.predict(X_2019_Scaled)
    accuracy = np.subtract(100, np.round(np.multiply(np.mean(predict_2019 != Y_2019.T), 100), 2))

    # Fit 2018 data remove sigma
    logisticRegression.fit(split_2018[0], Y_2018.ravel())
    prediction_remove_sigma = logisticRegression.predict(split_2019[0])
    accuracy_remove_sigma = np.subtract(100, np.round(np.multiply(np.mean(prediction_remove_sigma != Y_2019.T), 100), 2))

    # Fit 2018 data remove mean
    logisticRegression.fit(split_2018[1], Y_2018.ravel())
    prediction_remove_mean = logisticRegression.predict(split_2019[1])
    accuracy_remove_mean = np.subtract(100, np.round(np.multiply(np.mean(prediction_remove_mean != Y_2019.T), 100), 2))
    # Calculate contributions
    contribution_mean = accuracy - accuracy_remove_mean
    contribution_sigma = accuracy - accuracy_remove_sigma

    print('Sigma\tMean')
    print('{}%\t{}%'.format(contribution_sigma, contribution_mean))

    print('Contributions for KNN classifier shows that sigma as a feature contributes more than mean towards accuracy.')
    print('Contributes for Logistic Regression show that mean contributes negatively towards accuracy, but sigma contributes nothing towards accuracy.')

    print('\nQuestion 2')
    df_iris = pd.read_csv(iris_dataset, encoding='ISO-8859-1')
    df_iris.columns = ['Sepal Length', 'Sepal Width' , 'Petal Length' , 'Petal Width', 'Classification']
    flower_colors = df_iris['Classification'].unique()
    iris_classifications = np.array(['Sepal Length', 'Sepal Width' , 'Petal Length' , 'Petal Width'])
    df_table = pd.DataFrame(index=range(4), columns=flower_colors)
    df_table.index = iris_classifications
    df_table_overall_accuracy = pd.DataFrame(index=range(4), columns=flower_colors)
    df_table_overall_accuracy.index = iris_classifications

    for flower in flower_colors:
        X, Y = split_by_feature_and_flower(df_iris, flower)
        X_as_array = np.array(X)
        Y_as_array = np.array(Y)
        X_train, X_test, Y_train, Y_test = train_test_split(X_as_array, Y_as_array, test_size=0.5, random_state=3)
        logisticRegression.fit(X_train, Y_train.ravel())
        prediction = logisticRegression.predict(X_test)
        accuracy_overall = np.subtract(100, np.round(np.multiply(np.mean(prediction != Y_test.T), 100), 2))

        for classification in iris_classifications:
            X, Y = split_by_feature_and_flower(df_iris, flower, classification)
            X_as_array = np.array(X)
            Y_as_array = np.array(Y)
            X_train, X_test, Y_train, Y_test = train_test_split(X_as_array, Y_as_array, test_size=0.5, random_state=3)
            # Split into values
            logisticRegression.fit(X_train, Y_train.ravel())
            prediction = logisticRegression.predict(X_test)
            accuracy = np.subtract(100, np.round(np.multiply(np.mean(prediction != Y_test.T), 100), 2))
            df_table[flower][classification] = accuracy_overall - accuracy
            df_table_overall_accuracy[flower][classification] = accuracy
    print('Accuracy')
    print(df_table_overall_accuracy)
    print('Accuracy Shapley')
    print(df_table)
    print('Versicolor had the best performance of classification that was feature agnostic. Binary classifications with versicolor and the other flowers ')
    print('showed that there is separability between versicolor and those flowers. Iris-virginica has the second best separability, with petal length having the highest ')
    print('contribution towards accuracy at 5.33%. For Iris-versicolor, sepal width has the highest contribution with 9.33%, but it has the lowest separability overall.')

if __name__ == "__main__":
    main()