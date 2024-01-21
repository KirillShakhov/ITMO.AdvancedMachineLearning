import pandas as pd
from scipy.spatial.distance import euclidean, cityblock

# Create a DataFrame with the provided data
data = {
    'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'X': [42,47,46,33,10,40,76,81,32,82],
    'Y': [53,50,69,67,37,22,37,96,90,70],
    'Class': [0,0,1,0,0,0,0,1,0,0]
}

df = pd.DataFrame(data)

# Coordinates of the new object
new_object = (67, 95)

# Calculate distances using Euclidean metric
df['Euclidean_Distance'] = df.apply(lambda row: euclidean((row['X'], row['Y']), new_object), axis=1)

# Identify the three nearest neighbors using Euclidean distance
nearest_neighbors_euclidean = df.nsmallest(3, 'Euclidean_Distance')['id'].tolist()

# Classify the new object using Euclidean distance with k=3
class_euclidean = df.loc[df['id'].isin(nearest_neighbors_euclidean), 'Class'].mode()[0]

# Calculate distances using Manhattan (city block) metric
df['Manhattan_Distance'] = df.apply(lambda row: cityblock((row['X'], row['Y']), new_object), axis=1)

# Identify the three nearest neighbors using Manhattan distance
nearest_neighbors_manhattan = df.nsmallest(3, 'Manhattan_Distance')['id'].tolist()

# Classify the new object using Manhattan distance with k=3
class_manhattan = df.loc[df['id'].isin(nearest_neighbors_manhattan), 'Class'].mode()[0]

# Print the results
print(f"Euclidean Distance to nearest neighbor: {df['Euclidean_Distance'].min()}")
print(f"Identifiers of three nearest neighbors (Euclidean): {nearest_neighbors_euclidean}")
print(f"Class of new object (k=3, Euclidean): {class_euclidean}")
print(f"Manhattan (City Block) Distance to nearest neighbor: {df['Manhattan_Distance'].min()}")
print(f"Identifiers of three nearest neighbors (Manhattan): {nearest_neighbors_manhattan}")
print(f"Class of new object (k=3, Manhattan): {class_manhattan}")
