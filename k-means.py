import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the data
print("Loading customer data...")
data = pd.read_csv('Mall_Customers.csv')  # Adjust the filename if needed
print("Data loaded successfully.")

# Print the first few rows of the dataset
print("Customer data preview:")
print(data.head())

# Data preprocessing: select relevant features for clustering
# Ensure these columns exist in your dataset
features = data[['Annual Income (k$)', 'Spending Score (1-100)']]  # Adjust the feature names based on your dataset

# Check if the selected features exist
print("Selected features for clustering:")
print(features.describe())

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
print("Features standardized.")

# Determine the optimal number of clusters using the elbow method
inertia = []
k_values = range(1, 11)

print("Calculating inertia for different cluster values...")
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(features_scaled)
    inertia.append(kmeans.inertia_)
    print(f"Calculated inertia for k={k}: {kmeans.inertia_}")

# Plotting the elbow method
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.grid()
plt.show()

# After examining the plot, choose the optimal number of clusters
optimal_k = 5  # Change this based on the elbow method results
print(f"Using {optimal_k} clusters for K-means.")

# Apply K-means clustering
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
data['Cluster'] = kmeans.fit_predict(features_scaled)

# Print cluster information
print("Cluster centroids (original scale):")
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
print(centroids)

# Save the clustered data to a new CSV file
data.to_csv('clustered_customers.csv', index=False)
print("Clustered data saved to 'clustered_customers.csv'.")
