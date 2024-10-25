# Data Mining and Data Warehousing Lab

This repository contains practical implementations of various data mining and warehousing tasks. The projects utilize machine learning models, data preprocessing techniques, and clustering algorithms on different datasets like medical records, fuel consumption, customer segmentation, and academic performance.

## Table of Contents
1. [Diabetes Dataset](#1-diabetes-dataset)
2. [Petrol Consumption Dataset](#2-petrol-consumption-dataset)
3. [Mall Customers Dataset](#3-mall-customers-dataset)
4. [Marks Dataset](#4-marks-dataset)

---

### 1. Diabetes Dataset
Read `diabetes.csv` for diabetes that datasets consist of several medical predictor variables and one target variable, Outcome. Predictor variables include the number of pregnancies the
patient has had, their BMI, insulin level, age, and so on. Experiment with the following issues with python programming language-

#### Tasks:
- **a)** Show the number of patients information using a pie chart.
- **b)** Handle missing values using mean value for one column, median for another and mode for 3rd one if (any).
- **c)** Plot the boxplot of the pre-processed dataset.
- **d)** Compare the performance results of the ML model like LR, SVM and DT.
- **e)** Show the confusion matrix of your results.

[View the Jupyter Notebook for this task](https://github.com/nishatrhythm/Data-Mining-and-Data-Warehousing-Lab/blob/main/1.ipynb)

---

### 2. Petrol Consumption Dataset
Read `petrol_consumption.csv` Apply and Experiment with the following issues with python programming language:

#### Tasks:
- **a)** Predict the fuel consumption using multiple linear regression.
- **b)** Show and compare the results using 70:30, and 80:20 distribution during the training of the dataset.
- **c)** Show the actual and predicted value in a scatter plot for 80:20 distribution.
- **d)** Find the Mean Absolute Error.

[View the Jupyter Notebook for this task](https://github.com/nishatrhythm/Data-Mining-and-Data-Warehousing-Lab/blob/main/2.ipynb)

---

### 3. Mall Customers Dataset
Load the `Mall_Customers.csv`

#### Tasks:
- **a)** Visualize male and female customer spending scores.
- **b)** Find the ideal number of k using the elbow method.
- **c)** Apply k-means clustering using 4 clusters and 5 clusters.
- **d)** Draw the graph.

[View the Jupyter Notebook for this task](https://github.com/nishatrhythm/Data-Mining-and-Data-Warehousing-Lab/blob/main/3.ipynb)

---

### 4. Marks Dataset
Load the `Marks.csv` file. Then do the following:

#### Tasks:
- **a)** Write the statement to display the first and third quartiles of all subjects
- **b)** Find the standard deviation and variance of each subject
- **c)** Find the summary of the data

[View the Jupyter Notebook for this task](https://github.com/nishatrhythm/Data-Mining-and-Data-Warehousing-Lab/blob/main/4.ipynb)

---

## Getting Started

Clone the repository:
```bash
git clone https://github.com/nishatrhythm/Data-Mining-and-Data-Warehousing-Lab.git
```

### Prerequisites

Ensure that you have Python installed, along with all necessary dependencies. You can install the dependencies using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Usage

Navigate to the respective dataset directory and run the corresponding Python scripts or open the Jupyter notebooks to experiment with the code.

---

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
