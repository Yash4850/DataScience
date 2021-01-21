"""
Name: Yaswanth Reddy Maram Reddy
ID: 2003287
Username: ym20741
"""
# Importing libraries/modules

from collections import Counter
from csv import reader
from timeit import default_timer
import random
import math
import statistics

def load_from_csv(file_name):

    """
    The csv module from the standard library is used to easily import the data from the csv file provided.
    A context manager is used to handle opening and closing the file.
    The values are converted to floating points before they are added to the matrix.
    """
    matrix = []
    try:
        with open(file_name, "r") as csvfile:
            csvreader = reader(csvfile, delimiter=",")
            for row in csvreader:
                newline = [float(x) for x in row]
                matrix.append(newline)
        return matrix
    except FileNotFoundError:
        import sys
        print("File not found in current directory")
        sys.exit()

def get_distance(a, b):

    """
    The Euclidean distance is used to calculate the distance between two lists, a and b.
    The two lists are zipped together to easily access elements from both of them.
    """
    euclidean_distance = 0
    euclidean_distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(a, b)]))
    return euclidean_distance

def get_column(matrix, col_number):

    """
    Loops over the matrix selects elements of the same column and returns all of them in a list.
    """
    col_list = []
    for row in matrix:
        col_list.append(row[col_number])
    return col_list

def get_standard_deviation(matrix, col_number):

    """
    Uses get_column to retrieve a specified column from the matrix and then returns the standard deviation of that column.
    """
    return statistics.stdev(get_column(matrix, col_number))

def get_standardised_matrix(matrix):

    """
    The standardisation formula is used to create a new matrix of range scaled data
    For reference,
    𝑚𝑎𝑡𝑟𝑖𝑥(𝑥, 𝑦) = 𝑚𝑎𝑡𝑟𝑖𝑥(𝑥, 𝑦) − 𝑎𝑣𝑔(𝑚𝑎𝑡𝑟𝑖𝑥(: , 𝑦)) / 𝑠𝑡𝑑(𝑚𝑎𝑡𝑟𝑖𝑥(: , 𝑦))
    """
    standardised_matrix = []
    avg_of_col, std_of_col = [], []
    cols = range(len(matrix[0]))
    for i in cols:
        col_list = get_column(matrix, i)
        avg_of_col.append(sum(col_list, 0.0) / len(col_list))
        std_of_col.append(get_standard_deviation(matrix, i))
    for lst in matrix:
        standardised_list = []
        for i, element in enumerate(lst):
            standardised_element = (element - avg_of_col[i]) / std_of_col[i]
            standardised_list.append(standardised_element)
        standardised_matrix.append(standardised_list)
    return standardised_matrix

def get_k_nearest_labels(List,Learning_Data,Learning_Data_Labels,k):

    """
    Function finds the k rows of the Learning_Data that are the closest to the list passed as a parameter.
    It uses the get_distance function to do it. After finding these k rows, it finds and return the
    related rows in the Learning_Data_Labels
    """

    distances = list()
    for row in Learning_Data:
        dist = get_distance(List, row)
        distances.append((row, dist))
    # sorting the distances
    distances.sort(key=lambda lis: lis[1])
    indexes = []
    for i in range(k):
        for index,value in enumerate(Learning_Data):
            if (value == distances[i][0]) & (len(indexes)<k):
                indexes.append(index)
    #Getting Learning_Data_Labels k rows
    neighbors = list()
    for i in indexes:
        neighbors.append(Learning_Data_Labels[i])
    return neighbors

def flatten(L):
    """
    Flattens the matrix
    """
    for item in L:
        try:
            yield from flatten(item)
        except TypeError:
            yield item

# takes input from k nearest labels
def get_mode(k_nearest_labels):
    """
    the mode of the matrix is returned in this function
    """
    # flattening the matrix
    n_num = list(flatten(k_nearest_labels))
    n = len(n_num)
    data = Counter(n_num)
    get_mode = dict(data)
    # finding mode
    mode = [k for k, v in get_mode.items() if v == max(list(data.values()))]
    try:
        if len(mode) == 1:
            mode_knl = mode
        else:
            # chooses mode randomly
            mode_knl = random.choice(mode)
    except:
        print("Empty list, cannot find mode")
    return mode_knl

def classify(data,Learning_Data,Learning_Data_Labels,k):
    """
    This function follows the algorithm given in the assignment,
    classifies the data and gives the matrix_data_labels as output
    """
    matrix_data_labels = list()
    for row_x in data:
        # step 2a & 2b
        k_nearest_labels = get_k_nearest_labels(row_x,Learning_Data ,Learning_Data_Labels,k)
        # step 2c
        mode = get_mode(k_nearest_labels)
        # step 2d
        matrix_data_labels.append(mode)
    return matrix_data_labels

# matrix_data_labels is output of clf
def get_accuracy(Correct_Data_Labels, matrix_Data_Labels):
    """
    This function calculates and returns the percentage of accuracy
    """
    correct = 0
    for i in range(len(Correct_Data_Labels)):
        if Correct_Data_Labels[i] == matrix_Data_Labels[i]:
            correct += 1
    return correct / float(len(Correct_Data_Labels)) * 100.0

def run_test():
    """
    Running a series of tests of k ranging from 3 to 15
    """
    from collections import Counter
    # Loading csv files
    data = load_from_csv('Data.csv')
    Learning_Data = load_from_csv('Learning_Data.csv')
    Learning_Data_Labels = load_from_csv('Learning_Data_Labels.csv')
    Correct_Data_Labels = load_from_csv('Correct_Data_Labels.csv')
    # Standardizing the data
    standardised_data = get_standardised_matrix(data)
    standardised_Learning_Data = get_standardised_matrix(Learning_Data)
    for k in range(3, 16, 1):
        # Classifying the data
        clf = classify(standardised_data,standardised_Learning_Data,Learning_Data_Labels,k)
        # Finding accuracy
        accuracy = get_accuracy(Correct_Data_Labels, clf)
        print('k = ',k,'Accuracy = ',accuracy )

if __name__ == "__main__":
    start = default_timer()
    run_test()
    end = default_timer()
    print(f"\nTime taken: {end-start}")