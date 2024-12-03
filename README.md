# HCSAlgorithm
Here’s a clear explanation of how our code and the k-Means algorithm work:


---

Explanation of Our Code

1. Data Preparation

Sampling:

To reduce computational complexity, the dataset was sampled using Simple Random Sampling (SRS) to 200 observations.# HCSAlgorithm
Here’s a clear explanation of how our code and the k-Means algorithm work:


---

Explanation of Our Code

1. Data Preparation

Sampling:

To reduce computational complexity, the dataset was sampled using Simple Random Sampling (SRS) to 200 observations.

The sampled data included latitude and longitude as features.


sample_size = 200
housing_data_sampled = housing_data.sample(n=sample_size, random_state=42)
coordinates_sampled = housing_data_sampled[['longitude', 'latitude']].to_numpy()


2. Similarity Graph Construction

A graph was constructed using the networkx library:

Each observation was represented as a node.

Edges were added between nodes if the Euclidean distance between their coordinates was below a specified cutoff distance (e.g., 0.02).


for i, j in combinations(range(len(coordinates_sampled)), 2):
    if euclidean(coordinates_sampled[i], coordinates_sampled[j]) <= cutoff_distance:
        graph.add_edge(i, j)


3. Highly Connected Subgraphs (HCS) Algorithm

Key Steps:

1. High Connectivity Check: A subgraph was considered highly connected if its edge density exceeded 0.5.


2. Recursive Partitioning: If a subgraph was not highly connected, it was split into two subgraphs using the Stoer-Wagner minimum cut algorithm. This process repeated until all subgraphs met the high-connectivity criterion.


3. Singleton Handling: Nodes that could not form clusters were treated as singletons.



def hcs(graph):
    if is_highly_connected(graph):
        clusters.append(set(graph.nodes))
    else:
        cut_value, partition = nx.stoer_wagner(graph)
        left = graph.subgraph(partition[0]).copy()
        right = graph.subgraph(partition[1]).copy()
        hcs(left)
        hcs(right)


4. Visualization

The clusters were visualized using matplotlib or overlaid on a geographic map using folium. Each cluster was represented with a unique color.



---

How the k-Means Algorithm Works

1. Overview

k-Means is a centroid-based clustering algorithm that partitions the data into  clusters by minimizing the variance within each cluster. It assumes clusters are spherical and evenly distributed.

2. Key Steps

1. Initialize Centroids:

Randomly select  initial centroids from the data.



2. Assign Points to Clusters:

Each data point is assigned to the cluster of the nearest centroid (based on Euclidean distance).



3. Update Centroids:

The centroid of each cluster is recalculated as the mean of all points in that cluster.



4. Repeat:

Steps 2 and 3 are repeated until the centroids stabilize or the algorithm reaches a maximum number of iterations.




3. Code for k-Means

In our implementation, we used the KMeans class from the scikit-learn library:

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(coordinates_sampled)

# Get cluster labels and centroids
cluster_labels = kmeans.labels_
centroids = kmeans.cluster_centers_

4. Evaluation

The elbow method was used to determine the optimal , balancing cluster compactness (inertia) and computational efficiency.

Summary statistics (e.g., mean coordinates, cluster size) were calculated for each cluster.



---

Comparison of Both Algorithms


---

Summary

Our code applied the HCS algorithm to dynamically identify highly connected clusters using graph-theoretic principles, leveraging proximity and connectivity.

The k-Means algorithm was also applied to the same dataset, clustering points based on their distance to centroids for a predefined .

Both methods provide complementary insights:

HCS excels in irregular, connectivity-driven clusters.

k-Means is efficient and effective for compact, spherical clusters.



This explanation combines the theoretical and practical aspects of both algorithms in the context of the dataset. Let me know if you’d like to format this in LaTeX!



The sampled data included latitude and longitude as features.


sample_size = 200
housing_data_sampled = housing_data.sample(n=sample_size, random_state=42)
coordinates_sampled = housing_data_sampled[['longitude', 'latitude']].to_numpy()


2. Similarity Graph Construction

A graph was constructed using the networkx library:

Each observation was represented as a node.

Edges were added between nodes if the Euclidean distance between their coordinates was below a specified cutoff distance (e.g., 0.02).


for i, j in combinations(range(len(coordinates_sampled)), 2):
    if euclidean(coordinates_sampled[i], coordinates_sampled[j]) <= cutoff_distance:
        graph.add_edge(i, j)


3. Highly Connected Subgraphs (HCS) Algorithm

Key Steps:

1. High Connectivity Check: A subgraph was considered highly connected if its edge density exceeded 0.5.


2. Recursive Partitioning: If a subgraph was not highly connected, it was split into two subgraphs using the Stoer-Wagner minimum cut algorithm. This process repeated until all subgraphs met the high-connectivity criterion.


3. Singleton Handling: Nodes that could not form clusters were treated as singletons.



def hcs(graph):
    if is_highly_connected(graph):
        clusters.append(set(graph.nodes))
    else:
        cut_value, partition = nx.stoer_wagner(graph)
        left = graph.subgraph(partition[0]).copy()
        right = graph.subgraph(partition[1]).copy()
        hcs(left)
        hcs(right)


4. Visualization

The clusters were visualized using matplotlib or overlaid on a geographic map using folium. Each cluster was represented with a unique color.



---

How the k-Means Algorithm Works

1. Overview

k-Means is a centroid-based clustering algorithm that partitions the data into  clusters by minimizing the variance within each cluster. It assumes clusters are spherical and evenly distributed.

2. Key Steps

1. Initialize Centroids:

Randomly select  initial centroids from the data.



2. Assign Points to Clusters:

Each data point is assigned to the cluster of the nearest centroid (based on Euclidean distance).



3. Update Centroids:

The centroid of each cluster is recalculated as the mean of all points in that cluster.



4. Repeat:

Steps 2 and 3 are repeated until the centroids stabilize or the algorithm reaches a maximum number of iterations.




3. Code for k-Means

In our implementation, we used the KMeans class from the scikit-learn library:

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(coordinates_sampled)

# Get cluster labels and centroids
cluster_labels = kmeans.labels_
centroids = kmeans.cluster_centers_

4. Evaluation

The elbow method was used to determine the optimal , balancing cluster compactness (inertia) and computational efficiency.

Summary statistics (e.g., mean coordinates, cluster size) were calculated for each cluster.



---

Comparison of Both Algorithms


---

Summary

Our code applied the HCS algorithm to dynamically identify highly connected clusters using graph-theoretic principles, leveraging proximity and connectivity.

The k-Means algorithm was also applied to the same dataset, clustering points based on their distance to centroids for a predefined .

Both methods provide complementary insights:

HCS excels in irregular, connectivity-driven clusters.

k-Means is efficient and effective for compact, spherical clusters.



This explanation combines the theoretical and practical aspects of both algorithms in the context of the dataset. Let me know if you’d like to format this in LaTeX!

