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
1| 模型类不包括数据集成员对象，而应直接包括数据| √
2| 使用pandas改写|

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
1| 连续属性处理| √
2| 预剪枝、后剪枝|
3| 缺失值处理|
4| 使用pandas提升程序效率|
5| 决策树可视化|

### 3.3 Support Vector Machine

index|name|finished
-|-|-
1| 分类效果差| √
2| 多种核函数实现| √
3| 数据可视化| √
4| eta >= 0的情况|

### 3.4 Neural Network

index|name|finished
-|-|-
1| perceptron| √
2| visualize with turtle| √
3| fully connected nn|