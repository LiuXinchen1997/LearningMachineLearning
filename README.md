# LearningMachineLearning

> Coding for machine learning practice!

## 1 variable naming rule

variable|explanation
-|-
feat| feature vector of single sample
label| label of single sample, often just a integer
sample| vector concatenate feature vector and label of single sample
feats| feature vectors of all samples (often for train dataset or test dataset)
labels| labels of all samples
samples| vectors consisted of all samples
data| same as samples(above)
dataset| an object of class Dataset
attr| index of single feature
attrs| indexes of multi features

## 2 dataset

name| quantity| application| linear separable
-|-|-|-
watermelon3| small | classification| no
demodata| medium | classification| no
salary_data| small |regression | -

> we can also apply for dataset from sklearn

## 3 to do list \& issues

### 3.0 General

index|name|finished
-|-|-
1| optimize dataset interface | √
2| use python package pandas |

### 3.1 Linear Model

index|name|finished
-|-|-
1| Linear Regression| √
2| Logistic Regression|√
3| Linear Discriminant Analysis| √
4| Kernel LDA| 

### 3.2 Decision Tree

index|name|finished
-|-|-
1| process sequential attribution | √
2| pre-pruning, post-pruning |
3| process blank / missing value |
4| visualize decision tree |

### 3.3 Support Vector Machine

index|name|finished
-|-|-
1| improve classification performance | √
2| implement multi kernel methods | √
3| visualize data | √
4| case of eta >= 0 |

### 3.4 Neural Network

index|name|finished
-|-|-
1| perceptron| √
2| visualize with turtle| √
3| fully connected nn| √
4| improve classification performance| √
5| visualize classification result of fcnn| √
6| visualize training loss| √
7| visualize network architecture with turtle|
8| save and load parameter interface| √