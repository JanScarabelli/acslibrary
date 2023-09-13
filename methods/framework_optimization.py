import time

import matplotlib.pyplot as plt
import numpy as np


def IQR(df, x_name):  # Definition of IQR Outlier removal

    for label in x_name:

        Q1 = df[label].quantile(0.25)
        Q3 = df[label].quantile(0.75)
        IQR = Q3 - Q1
        i = 0
        for x in df[label]:

            if (x < (Q1 - 1.5 * IQR)) | (x > (Q3 + 1.5 * IQR)) == True:
                df = df.drop(df.index[i])
                i = i - 1
            i = i + 1
    return df


def optimize(df, x_name, y_name, model, opt_method: dict, maximize, remove_outliers):

    """
    This function receives as inputs the data frame, explanatory variables, target variable, predicting model and the specifications for the
    desired optimization process and prints, the success of the optimizing function, the maximum (or minimum),
    evaluation of the predicting model in the maximum (or minimum), execution time and a visualisation of the normalised process variables.

    By default the function uses as bounds the minimum and maximum of each of the process variables in the data frame, if removal_outliers
    is True it removes outliers using the IQR method before defining the bounds.
    When using predefined bounds the IQR method will never be used.


    Parameters:
    ------------
    df: dataframe
        dataframe containing at least the explanatory and objective variables.

    x_name: list of strings
        names of the columns from the dataset that want to be used as explanatory variables.

    y_name: string
        name of the column from the dataset that want to be used as response variable.

    model:
        created model to be optimized.

    opt_method: dictionary
        dictionary containing the function used for optimization and a dictionary containing it's hyperparameters.

    maximize: boolean
        If maximize = True : the function will search a maximum of the objective variable
        If maximize = False : the function will search a minimum of the objective variable

    remove_outliers: boolean
        If remove_outliers = True : the function removes outliers from the dataset using the IQR method before defining
         default bounds and prints how many samples have been removed

    """
    dim = len(x_name)  # Counts the number of explanatory variables
    control = 0

    if (
        opt_method['params'].get('bounds', 0) != 0
    ):  # Checks if bounds is a parameter of the chosen optimizing function

        if (
            opt_method['params']['bounds'] == None
        ):  # Checks if bounds has ben passed as an argument before using default bounds

            control = 1

            if remove_outliers == True:  # Performs IQR outlier removal
                df = IQR(df, x_name)

            opt_method['params']['bounds'] = []

            for (
                label
            ) in x_name:  # Creates the bound list using the min and max value of each variable

                var_bound = [[df.min()[label], df.max()[label]]]
                opt_method['params']['bounds'] = opt_method['params']['bounds'] + var_bound

    X = df[x_name]

    if maximize == True:

        def pred(
            x,
        ):  # Function made to shape the output of the model to the shape required by the optimization function
            x = x.reshape(1, dim)
            Y_pred = model.predict(x)
            return (-1) * Y_pred[0]  # Sign change because the objective is a maximum

        start_time = time.time()  # Computing running time
        result = opt_method["method"](pred, **opt_method['params'])
        finish_time = time.time()

    else:

        def pred(
            x,
        ):  # Function made to shape the output of the model to the shape required by the optimization function
            x = x.reshape(1, dim)
            Y_pred = model.predict(x)
            return Y_pred[0]  # No sign change

        start_time = time.time()  # Computing running time
        result = opt_method["method"](pred, **opt_method['params'])
        finish_time = time.time()

    print("Success:", result.success)
    print("Execution time:", finish_time - start_time)

    if maximize == True:
        print("Function maximum is:", -1 * result.fun)

    else:
        print("Function minimum is:", result.fun)

    print("Achieved at the point:", result.x)

    print("Visualisation of the normalized process variables:")

    if (
        control == 1
    ):  # Normalization of the result obtained by the optimization function when no bounds defined

        point = np.subtract(result.x, X.min())
        rng = X.max() - X.min()
        point = np.divide(point, rng)

    else:  # Normalization of the result obtained by the optimization function when bounds defined

        X_min = []
        X_max = []

        for i in range(0, dim):

            X_min.append(opt_method['params']['bounds'][i][0])
            X_max.append(opt_method['params']['bounds'][i][1])

        X_min_np = np.array(X_min)
        X_max_np = np.array(X_max)
        rng = X_max_np - X_min_np

        point = np.subtract(result.x, X_min_np)
        point = np.divide(point, rng)

    plt.bar(x_name, point)  # Plot of the normalized point obtained
    ax = plt.gca()
    ax.set_ylim([0, 1.1])
    plt.show()
