# Automatic Cancer Diagnostic
It is quite expensive to determine whether a particular patient has cancer or not.This software will read CSV files with measurements taken from blood tests of patients and produce a diagnostic.

The aim of this software is to classify each row of the data matrix (representing the measurements of one patient). The algorithm you should use is explained in the appendix.

# Appendix
Data standardization
where, x is a non-negative integer representing a row number, y is a non-negative integer representing a column number, avg(matrix(:,y)) is the average of column y over all the rows, and std(matrix(:,y)) is the corrected sample standard deviation of column y over all the rows.
The corrected sample standard deviation of the column y over all rows, matrix(:,y), is given by:
where N is the number of rows in the matrix.
Classification algorithm
1. Set a positive value for k.
2. For each row x in the matrix data
 a. Find the k rows in the matrix learning_data that are the closest to the row x in the matrix data
according to the Euclidean distance (see below).
 b. Given the rows found on Step 2(a), find the values of the same row numbers in the matrix
learning_data_labels.
 c. Given the values found in Step 2(b), find the most common value (that with the highest
frequency). If two (or more) values have the same frequency, choose one of them at random.
 d. Set the row x of the matrix data_labels to the value found in Step 2(c).
Euclidean distance:
The Euclidean distance between a list_a and a list_b (both of size M) is given by:

