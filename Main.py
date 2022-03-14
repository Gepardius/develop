"""
Gal Bordelius, 92120668
This is a Python-program that uses training data to choose the
four ideal functions which are the best fit out of the fifty provided (C) *.

    i)  Afterwards, the program uses the test data provided (B) to determine
        for each and every x-y-pair of values whether or not they can be assigned to the
        four chosen ideal functions**; if so, the programs to executes the mapping
        and saves it together with the deviation at hand
    ii) All data is visualized logically
    iii) Where possible, suitable unit-test were compiled
"""

import sys     # Standard library imports
import pandas as pd   # related third party imports
import numpy as np
from matplotlib import pyplot as plt
from sqlalchemy import create_engine


class FindFunctions:
    def __init__(self):
        pass

    def find_ideal_matches(self, train_fun, ideal_fun):
        """
        function finds matches between training functions and ideal functions based on min(MSE)
        :param train_fun: define training functions
        :param ideal_fun: define ideal functions set
        :return: ideal functions dataframe and their deviations
        """

        # find last parameters of both fucntions
        if isinstance(train_fun, pd.DataFrame) and isinstance(ideal_fun, pd.DataFrame):
            ideal_lcol = len(ideal_fun.columns)
            train_lrow = train_fun.index[-1] + 1
            train_col = len(train_fun.columns)

            # Loop and find perfect four functions
            index_list = []  # here 4 ideal indexes will be strored
            least_square = []  # here 4 ideal MSEs will be stores
            for j in range(1, train_col):  # loop through 4 train functions
                least_square1 = []
                for k in range(1, ideal_lcol):  # loop through 50 ideal functions
                    MSE_sum = 0  # Sum MSE
                    for i in range(train_lrow):  # calculate MSE Y value of train and Y value of ideal function
                        z1 = train_fun.iloc[i, j]  # Train y value
                        z2 = ideal_fun.iloc[i, k]  # Ideal y value
                        MSE_sum = MSE_sum + ((z1 - z2) ** 2)
                    least_square1.append(MSE_sum / train_lrow)
                min_least = min(least_square1)
                index = least_square1.index(min_least)  # find index of the ideal function
                index_list.append(index + 1)
                least_square.append(min_least)

            per_frame = pd.DataFrame(list(zip(index_list, least_square)), columns=["Index", "least_square_value"])

            return per_frame
        else:
            raise TypeError("Given arguments are not of Dataframe type.")

    def find_ideal_via_row(self, test_fun):
        """
        determine for each and every x-y-pair of values whether they can be assigned to the four chosen ideal functions
        :param test_fun: Dataframe with x and y values
        :return: test function paired with values from the four ideal functions
        """
        if isinstance(test_fun, pd.DataFrame):
            test_lrow = test_fun.index[-1] + 1  # last row of the test df (used for loop)
            test_lcol = len(test_fun.columns)  # last columns of the test df (used for loop)
            # print(test)

            ideal_index = []  # list to store index of ideal function
            deviation = []  # list to store Deviation
            for j in range(test_lrow):  # loop through rows
                MSE_l = []  # list to store all four deviations
                for i in range(2, test_lcol):  # loop through colums 2, 3, 4, 5
                    z1 = test_fun.iloc[j, 1]
                    z2 = test_fun.iloc[j, i]
                    MSE_sum = ((z2 - z1) ** 2)  # calculate MSE
                    MSE_l.append(MSE_sum)  # append MSE to the MSE_l list
                min_least = min(MSE_l)  # select min deviation in MSE_l
                if min_least < (np.sqrt(2)):
                    deviation.append(min_least)  # append min_least to the deviation list
                    index = MSE_l.index(min_least)  # select index of the min_least to find ideal function
                    ideal_index.append(index)  # append index to the ideal_index list
                else:
                    deviation.append(min_least)
                    ideal_index.append("Miss")  # no criteria match

            # Add two new columns to the test
            test["Deviation"] = deviation
            test["Ideal index"] = ideal_index

            return test

        else:
            raise TypeError("Given argument is not of Dataframe type.")

    def prepare_graphs(self, x_fun, x_par, y1_fun, y1_par, y2_fun, y2_par, show_plots=True):
        """
        function prepares a plot based on given paramaters
        :param x_fun: x function
        :param x_par: x position
        :param y1_fun: y1 function
        :param y1_par: y1 position
        :param y2_fun: y2 function
        :param y2_par: y2 position
        :param show_plots: True/False to display plot
        :return: graph of x and y
        """

        x = x_fun.iloc[:, x_par]    # x
        y1 = y1_fun.iloc[:, y1_par]     # y1 (training function)
        y2 = y2_fun.iloc[:, y2_par]     # y2 (ideal function)

        # print(y1, y2)

        plt.plot(x, y1, c="r", label="Train function")  # plot both axis
        plt.plot(x, y2, c="b", label="Ideal function")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend(loc=3)

        if show_plots is True:
            plt.show()  # show current plot
            plt.clf()   # clear plots
        elif show_plots is False:
            pass
        else:
            pass  # no paramater show_plots or wrong paramater show_plots was given


class SqliteDb(FindFunctions):
    """
    Load data into Sqlite database
    """

    def db_and_table_creation(self, dataframe, db_name, table_name):
        """
        function creates a database from a dataframe input
        :param dataframe: dataframe
        :param db_name: database name
        :param table_name: table name
        :return: database file into the same folder as the project
        """
        try:
            if isinstance(dataframe, pd.DataFrame):
                engine = create_engine(f"sqlite:///{db_name}.db", echo=True)
                sqlite_connection = engine.connect()
                sqlite_table = table_name
                dataframe.to_sql(sqlite_table, sqlite_connection, if_exists='fail')
                sqlite_connection.close()
        except Exception:
            exception_type, exception_value, exception_traceback = sys.exc_info()
            print(exception_type, exception_value, exception_traceback)


# read CSV files and load them into Dataframes
train = pd.read_csv("train.csv")
ideal = pd.read_csv("ideal.csv")
test = pd.read_csv("test.csv")

# Check data formats
# print(train.head)
# print(ideal.head)
# print(test.head)

# get ideal functions based on train data
df = FindFunctions().find_ideal_matches(train, ideal)
# print(df)

# plot graph of all 4 pair functions together
graph = FindFunctions()
for i in range(1, len(train.columns)):
    graph.prepare_graphs(train, 0, train, i, ideal, df.iloc[i-1, 0], False)

# Clean test df
test = test.sort_values(by=["x"], ascending=True)   # sort by x
test = test.reset_index()   # reset index
test = test.drop(columns=["index"])     # drop old index column

# Get x, y values of each of the 4 ideal functions
ideals = []
for i in range(0, 4):
    ideals.append(ideal[["x", f"y{str(df.iloc[i, 0])}"]])

# merge test and 4 ideal functions
for i in ideals:
    test = test.merge(i, on="x", how="left")

# determine for each and every x-y-pair of values whether or not they can be assigned to the four chosen ideal functions
test = FindFunctions().find_ideal_via_row(test)

# Replace values with ideal function names
for i in range(0, 4):
    test["Ideal index"] = test["Ideal index"].replace([i], str(f"y{df.iloc[i, 0]}"))

# add y values to another test_fun (used later for scatter plot)
test_scat = test
test_scat["ideal y value"] = ""
for i in range(0, 100):
    k = test_scat.iloc[i, 7]
    if k == "y18":
        test_scat.iloc[i, 8] = test_scat.iloc[i, 2]
    elif k == "y3":
        test_scat.iloc[i, 8] = test_scat.iloc[i, 3]
    elif k == "y30":
        test_scat.iloc[i, 8] = test_scat.iloc[i, 4]
    elif k == "y23":
        test_scat.iloc[i, 8] = test_scat.iloc[i, 5]
    elif k == "Miss":
        test_scat.iloc[i, 8] = test_scat.iloc[i, 1]
# print(test_scat)

# Drop other columns that are not used
test = test.drop(columns=["y18", "y3", "y30", "y23"])

# rename columns for the train table
train = train.rename(columns={"y1": "Y1 (training func)", "y2": "Y2 (training func)",
                              "y3": "Y3 (training func)", "y4": "Y4 (training func)"})
# print(train)

# rename columns for the ideal table
for col in ideal.columns:       # rename columns in ideal to fit criteria
    if len(col) > 1:    # if column name is not x, therefore > 1
        ideal = ideal.rename(columns={col: f"{col} (ideal func)"})

# print(ideal)

# clean column names for the test table
test = test.rename(columns={"x": "X (test func)",
                            "y": "Y (test func)",
                            "Deviation": "Delta Y (test func)",
                            "Ideal index": "No. of ideal func"})

# Load data to sqlite
train_db = SqliteDb()
train_db.db_and_table_creation(train, "train_database", "train_table")

ideal_db = SqliteDb()
ideal_db.db_and_table_creation(ideal, "ideal_database", "ideal_table")

test_db = SqliteDb()
test_db.db_and_table_creation(test, "test_database", "test_table")

# Visualization
# train functions
plt.clf()
x = train.iloc[:, 0]
for i in range(1, len(train.columns)):
    plt.plot(x, train.iloc[:, i], c="g", label=f"Train function y{i}")
    plt.legend(loc=3)
    # plt.show()
    plt.clf()

# ideal functions (all 50)
plt.clf()
x = ideal.iloc[:, 0]
for i in range(1, len(ideal.columns)):
    plt.plot(x, ideal.iloc[:, i], c="#FF4500", label=f"Ideal function y{i}")
    plt.legend(loc=3)
    # plt.show()
    # plt.clf()

# ideal functions (4 chosen)
plt.clf()
x = train.iloc[:, 0]
for i in range(0, df.index[-1] + 1):
    y = df.iloc[i, 0]  # get ideal y column number (18, 3, 30, 23)
    plt.plot(x, ideal.iloc[:, y], c="#FF4500", label=f"Ideal function y{y}")
    plt.legend(loc=3)
    # plt.show()
    # plt.clf()

# test scatter (show points of test.csv)
# plt.clf()  # clear previous plots
plt.scatter(test.iloc[:, 0], test.iloc[:, 1])  # select x and y values
# plt.show()

plt.clf()  # clear previous plots
# create lists to visualize test_scat dataframe
x1 = []
x2 = []
x3 = []
x4 = []
xm = []
y1 = []
y2 = []
y3 = []
y4 = []
ym = []

# append x and y values to the upper lists
for i in range(0, 100):
    k = test_scat.iloc[i, 7]
    if k == "y18":
        x1.append(test_scat.iloc[i, 0])  # append x value of y18 to the x1 list
        y1.append(test_scat.iloc[i, 8])  # append y value of y18 to the y1 list
    elif k == "y3":
        x2.append(test_scat.iloc[i, 0])  # append x value of y3 to the x2 list
        y2.append(test_scat.iloc[i, 8])  # append y value of y3 to the y2 list
    elif k == "y30":
        x3.append(test_scat.iloc[i, 0])  # append x value of y30 to the x3 list
        y3.append(test_scat.iloc[i, 8])  # append y value of y30 to the y3 list
    elif k == "y23":
        x4.append(test_scat.iloc[i, 0])  # append x value of y23 to the x4 list
        y4.append(test_scat.iloc[i, 8])  # append y value of y23 to the y4 list
    elif k == "Miss":
        xm.append(test_scat.iloc[i, 0])  # append x value of "Miss" values to the xm list
        ym.append(test_scat.iloc[i, 8])  # append y value of "Miss" values to the ym list

# plot ideal functions and test y-values on the same scatter plot
plt.scatter(x1, y1, marker="o", label="Test - y18", color="r")
plt.scatter(x2, y2, marker="s", label="Test - y3", color="b")
plt.scatter(x3, y3, marker="^", label="Test - y30", color="g")
plt.scatter(x4, y4, marker="d", label="Test - y23", color="#FFD700")
plt.scatter(xm, ym, marker="x", label="Test - Miss", color="#000000")
plt.plot(ideal.iloc[:, 0], ideal.iloc[:, 18], label="Ideal - Y18", color="#FA8072")
plt.plot(ideal.iloc[:, 0], ideal.iloc[:, 3], label="Ideal - Y3", color="#1E90FF")
plt.plot(ideal.iloc[:, 0], ideal.iloc[:, 30], label="Ideal - Y30", color="#7CFC00")
plt.plot(ideal.iloc[:, 0], ideal.iloc[:, 23], label="Ideal - Y23", color="#FFA500")
plt.xlabel("x")
plt.xlabel("y")
plt.legend()
plt.show()

# print(train.head)
# print(ideal.head)
# print(test.head)
