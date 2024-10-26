import pandas as pd
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

# Loading the mall customers dataset
mall_data = pd.read_csv('Mall_Customers.csv')

# a) Visualizing Male and Female Spending Scores
sns.boxplot(x='Gender', y='Spending Score (1-100)', data=mall_data)
plt.title('Spending Scores by Gender')
plt.show()

# b) Elbow method to find optimal k
sse = []
X_mall = mall_data[['Annual Income (k$)', 'Spending Score (1-100)']]
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_mall)
    sse.append(kmeans.inertia_)

plt.plot(range(1, 11), sse)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.show()

# c) Applying K-Means with 4 and 5 clusters
kmeans_4 = KMeans(n_clusters=4, random_state=42).fit(X_mall)
mall_data['Cluster_4'] = kmeans_4.labels_

kmeans_5 = KMeans(n_clusters=5, random_state=42).fit(X_mall)
mall_data['Cluster_5'] = kmeans_5.labels_

# d) Scatter plot of the clusters
plt.scatter(X_mall['Annual Income (k$)'], X_mall['Spending Score (1-100)'], c=mall_data['Cluster_5'], cmap='viridis')
plt.title('K-Means Clustering with 5 Clusters')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()