k-nearest neighbor classifier
=====================

The code written in Matlab.

Input
--------------
train_data: N*D matrix, each row as a sample and each column as a feature

train_label: N*1 vector, each row as a label

new_data: M*D matrix, each row as a sample and each column as a feature

new_label: M*1 vector, each row as a label

k: number of nearest neighbors

Output
-------------
new_accu: accuracy of classifying new_data

train_accu: accuracy of classifying train_data (using leave-one-out strategy)

Usage
-----------------
    [test_accu, train_accu] = knn_classify(train_data, train_label, test_data, test_label, k);
    
