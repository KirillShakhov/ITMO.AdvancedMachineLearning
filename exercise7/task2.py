import numpy as np
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
from PIL import Image

def plot_pixels(data, colors=None, N=10000):
    if colors is None:
        colors = data

    # Choose a random subset
    rng = np.random.RandomState(0)
    i = rng.permutation(data.shape[0])[:N]
    colors = colors[i]
    pixel = data[i].T
    R, G, B = pixel[0], pixel[1], pixel[2]

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    ax[0].scatter(R, G, color=colors, marker='.')
    ax[0].set(xlabel='Red', ylabel='Green', xlim=(0, 1), ylim=(0, 1))

    ax[1].scatter(R, B, color=colors, marker='.')
    ax[1].set(xlabel='Red', ylabel='Blue', xlim=(0, 1), ylim=(0, 1))

    fig.suptitle('Color Plot of RGB')
    plt.show()

# Load the image
img_path = 'spb_bridge.jpg'  # <-- Make sure to have the image in your local directory
img = Image.open(img_path)
img = img.convert('RGB')

# Convert the image to numpy array and check its shape
img_array = np.array(img)

# Calculate the mean intensity of the pixels across all channels
mean_intensity_original = np.mean(img_array)

# Normalize the pixel intensity values
img_array_norm = img_array / 255.0

# Calculate the mean intensity of the normalized pixel values across all channels
mean_intensity_normalized = np.mean(img_array_norm)

# Reshape the image array
img_array_reshaped = img_array_norm.reshape(-1, 3)

# Training a MiniBatchKMeans model
kmeans = MiniBatchKMeans(n_clusters=16, random_state=16)
kmeans.fit(img_array_reshaped)

# Getting the cluster centers and labels
cluster_centers = kmeans.cluster_centers_
cluster_labels = kmeans.labels_

# Replacing pixel values with cluster centers
img_array_compressed = cluster_centers[cluster_labels].reshape(img_array.shape[:-1] + (-1,))

# Calculate the mean intensity of the pixels of the compressed image
mean_intensity_compressed = np.mean(img_array_compressed)

# Plot the color plots of the compressed image
plot_pixels(img_array_reshaped, colors=cluster_centers[cluster_labels])

# Show the 4x4 image of 16 colors
img_4x4 = Image.fromarray((cluster_centers * 255).astype('uint8')).resize((4, 4), Image.NEAREST)
img_4x4.show()

# Outputs
print(f"Mean intensity original: {mean_intensity_original:.3f}")
print(f"Mean intensity normalized: {mean_intensity_normalized:.3f}")
print(f"Mean intensity compressed: {mean_intensity_compressed:.3f}")
