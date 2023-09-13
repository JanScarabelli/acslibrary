import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.cross_decomposition import PLSRegression


def feature_selection_pls(
    X, y, max_components: int, max_variables: int, n_splits: int, early_stop=True
) -> None:
    """
    This function receives as inputs a data frame containing a set of explanatory variables, dataframe containing
    the response variable, the maximum number of components used in the PLS, the maximum number of variables used in the PLS
    and the number of folds used in cross validation. It starts by creating a PLS model for every variable in X, it performs
    shuffle cross validation and no shuffle cross validation and takes the mean r2 of both cases. Once the best variable at
    cross validation is obtained, a PLS model is created using the best variable and every other variable from X, if one of
    the PLS models created performs better than the one using one variable the new best variable is included. This process is
    repeated until adding a new variable does not improve the r2 score in cross validation or until max_variables is reached
    (if early_stop = False, it keeps adding variables only until max_variables reached).
    In every iteration of variables all possible number of components are tried until max_components or the number of
    variables is reached.


    Parameters:
    ------------
    X: dataframe
        dataframe containing a set of explanatory variables

    y: dataframe
        dataframe containing the response variable

    max_component: int
        maximum number of components used for every iteration of PLS

    max_variables: int
        maximum number of variables to be used in all the PLS iterations

    n_splits: int
        number of splits used in the internal cross validation method

    early_stop: bool
        if early_stop = True the function will stop adding new variables if there is no improvement in r2
        if early_stop = True the function will continue to add variables until max_variables is reached

    Output:
    ------------
    None

    """
    X = X.copy()
    y = y.copy()

    best_r2 = float('-inf')
    best_var_index = 0
    best_var = []
    best_score = []

    pls = PLSRegression(n_components=1, scale=True)

    for i in range(0, len(X.columns)):

        X_aux = X[[X.columns[i]]]

        # Perform shuffle cross-validation

        cv = KFold(n_splits=n_splits, shuffle=True, random_state=1)
        scores = cross_validate(pls, X_aux, y, scoring='r2', cv=cv, n_jobs=-1)
        r2 = np.mean(scores['test_score'])

        # Perform no shuffle cross-validation

        cv = KFold(n_splits=n_splits, shuffle=False)
        scores = cross_validate(pls, X_aux, y, scoring='r2', cv=cv, n_jobs=-1)
        r2 = r2 + np.mean(scores['test_score'])

        r2 = r2 / 2

        if r2 > best_r2:
            best_r2 = r2
            best_var_index = i

    best_var.append(X.columns[best_var_index])
    best_score.append(best_r2)
    X_aux = X[[X.columns[best_var_index]]]
    X.drop(columns=X.columns[best_var_index], inplace=True)

    best_component = 0
    best_component_list = [1]
    loop_break = False

    for i in range(
        1, min(len(X.columns), max_variables)
    ):  # Iterates over number of variables used in PLS

        best_subset_r2 = float('-inf')

        for j in range(0, len(X.columns)):  # Iterates over combinations of variables

            best_component_r2 = float('-inf')
            X_aux['iter_' + str(i)] = X[[X.columns[j]]]

            for k in range(
                1, min(i, max_components) + 1
            ):  # Iterates over number of components for a subset of variables

                pls = PLSRegression(n_components=k, scale=True)

                # Perform shuffle cross-validation

                cv = KFold(n_splits=n_splits, shuffle=True, random_state=1)
                scores = cross_validate(pls, X_aux, y, scoring='r2', cv=cv, n_jobs=-1)
                r2 = np.mean(scores['test_score'])

                # Perform shuffle cross-validation

                cv = KFold(n_splits=n_splits, shuffle=False)
                scores = cross_validate(pls, X_aux, y, scoring='r2', cv=cv, n_jobs=-1)
                r2 = r2 + np.mean(scores['test_score'])

                r2 = r2 / 2

                if r2 > best_component_r2:
                    best_component_r2 = r2
                    best_component = k

            if best_component_r2 > best_subset_r2:
                best_subset_r2 = best_component_r2
                best_subset = X.columns[j]

        best_component_list.append(best_component)

        if best_subset_r2 <= best_score[i - 1] and early_stop == True:

            print("Set of variables used:", best_var)
            print("Number of components:", best_component_list)
            print("Best r2 score:", best_score)
            print("No improvements in adding new variables")
            loop_break = True
            break

        else:
            best_score.append(best_subset_r2)
            X_aux['iter_' + str(i)] = X[[str(best_subset)]]
            X_aux.rename(columns={'iter_' + str(i): str(best_subset)}, inplace=True)
            best_var.append(best_subset)
            X.drop(columns=str(best_subset), inplace=True)

    if loop_break != True:

        print("Set of variables used:", best_var)
        print("Number of components:", best_component_list)
        print("Best r2 score:", best_score)
        print("Maximum number of variables reached")


def feature_selection_any(model, X, y, max_variables: int, n_splits: int, early_stop=True) -> None:
    """
    This function receives as inputs a model, data frame containing a set of explanatory variables, dataframe containing
    the response variable, the maximum number of variables used in the model and the number of folds used in cross validation.
    It starts by fitting the model for every variable in X, it performs shuffle cross validation and no shuffle cross validation
    and takes the mean r2 of both cases.
    Once the best variable at cross validation is obtained, a model is fitted using the best variable and every other variable from X, if one of
    the new models created performs better than the one using one variable the new best variable is included. This process is
    repeated until adding a new variable does not improve the r2 score in cross validation or until max_variables is reached
    (if early_stop = False, it keeps adding variables only until max_variables reached).


    Parameters:
    ------------
    model: scikit learn model

    X: dataframe
        dataframe containing a set of explanatory variables

    y: dataframe
        dataframe containing the response variable

    max_variables: int
        maximum number of variables to be used in all the PLS iterations

    n_splits: int
        number of splits used in the internal cross validation method

    early_stop: bool
        if early_stop = True the function will stop adding new variables if there is no improvement in r2
        if early_stop = True the function will continue to add variables until max_variables is reached

    Output:
    ------------
    None

    """

    X = X.copy()
    y = y.copy()

    best_r2 = float('-inf')
    best_var_index = 0
    best_var = []
    best_score = []

    for i in range(0, len(X.columns)):
        X_aux = X[[X.columns[i]]]

        # Perform shuffle cross-validation

        cv = KFold(n_splits=n_splits, shuffle=True, random_state=1)
        scores = cross_validate(model, X_aux, y, scoring='r2', cv=cv, n_jobs=-1)
        r2 = np.mean(scores['test_score'])

        # Perform no shuffle cross-validation

        cv = KFold(n_splits=n_splits, shuffle=False)
        scores = cross_validate(model, X_aux, y, scoring='r2', cv=cv, n_jobs=-1)
        r2 = r2 + np.mean(scores['test_score'])

        r2 = r2 / 2

        if r2 > best_r2:
            best_r2 = r2
            best_var_index = i

    best_var.append(X.columns[best_var_index])
    best_score.append(best_r2)
    X_aux = X[[X.columns[best_var_index]]]
    X.drop(columns=X.columns[best_var_index], inplace=True)

    loop_break = False

    for i in range(
        1, min(len(X.columns), max_variables)
    ):  # Iterates over number of variables used in PLS
        best_subset_r2 = float('-inf')

        for j in range(0, len(X.columns)):  # Iterates over combinations of variables
            best_component_r2 = float('-inf')
            X_aux['iter_' + str(i)] = X[[X.columns[j]]]

            # Perform shuffle cross-validation

            cv = KFold(n_splits=n_splits, shuffle=True, random_state=1)
            scores = cross_validate(model, X_aux, y, scoring='r2', cv=cv, n_jobs=-1)
            r2 = np.mean(scores['test_score'])

            # Perform shuffle cross-validation

            cv = KFold(n_splits=n_splits, shuffle=False)
            scores = cross_validate(model, X_aux, y, scoring='r2', cv=cv, n_jobs=-1)
            r2 = r2 + np.mean(scores['test_score'])

            r2 = r2 / 2

            if r2 > best_subset_r2:
                best_subset_r2 = r2
                best_subset = X.columns[j]

        if best_subset_r2 <= best_score[i - 1] and early_stop == True:
            print("Set of variables used:", best_var)
            print("Best r2 score:", best_score)
            print("No improvements in adding new variables")
            loop_break = True
            break

        else:
            best_score.append(best_subset_r2)
            X_aux['iter_' + str(i)] = X[[str(best_subset)]]
            X_aux.rename(columns={'iter_' + str(i): str(best_subset)}, inplace=True)
            best_var.append(best_subset)
            X.drop(columns=str(best_subset), inplace=True)

    if loop_break != True:
        print("Set of variables used:", best_var)
        print("Best r2 score:", best_score)
        print("Maximum number of variables reached")
