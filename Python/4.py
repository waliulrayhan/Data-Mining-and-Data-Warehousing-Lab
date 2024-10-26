import pandas as pd

# Loading the marks dataset
marks_data = pd.read_csv('Marks.csv')

# a) First and third quartiles of all subjects
quartiles = marks_data[['Maths', 'Physics', 'Chemistry', 'English', 'Biology']].quantile([0.25, 0.75])
print("First and Third Quartiles:\n", quartiles)

# b) Standard deviation and variance of each subject
std_dev = marks_data[['Maths', 'Physics', 'Chemistry', 'English', 'Biology']].std()
variance = marks_data[['Maths', 'Physics', 'Chemistry', 'English', 'Biology']].var()

print("Standard Deviation:\n", std_dev)
print("Variance:\n", variance)

# c) Summary of the data
summary = marks_data.describe()
print("Summary of the data:\n", summary)