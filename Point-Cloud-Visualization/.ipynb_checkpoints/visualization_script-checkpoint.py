from platform import python_version

# Print the Python version
print("Python Version:", python_version())

# Data read & write
import numpy as np
# import laspy  # Uncomment if needed for LAS files
import h5py  # Uncomment if you're working with HDF5 data
# Visualization
import open3d as o3d
import pptk  # If using pptk, for Python 3.6 (not used in this script)



# Load the 3D volume and remove the batch dimension
volume = np.load('alpha_14000.npy')[0]

# Threshold the volume to extract points with high intensity (e.g., > 0.5)
threshold = 0.5
points = np.argwhere(volume > threshold).astype(np.float32)  # Extract (z, y, x) indices of voxels above threshold

# Ensure the points array has the shape (N, 3) - discard the fourth column if it exists
print(f"Original points shape: {points.shape}")

# If the points array has 4 columns, remove the fourth column (intensity or other data)
points = points[:, :3]  # Keep only (z, y, x) coordinates

# Ensure the points array has the correct shape (N, 3)
print(f"Updated points shape: {points.shape}")

# Create a PointCloud object in Open3D
point_cloud = o3d.geometry.PointCloud()

# Convert the numpy array of points to an Open3D point cloud
point_cloud.points = o3d.utility.Vector3dVector(points)

# Normalize intensity values for coloring
intensities = volume[volume > threshold]  # Extract the intensities
intensities = intensities / intensities.max()  # Normalize to [0, 1]

# Map intensities to RGB colors (grayscale for simplicity)
colors = np.stack([intensities, intensities, intensities], axis=-1)

# Assign the colors to the point cloud
point_cloud.colors = o3d.utility.Vector3dVector(colors)

# Save the point cloud to a PLY file
ply_file_path = "output_point_cloud.ply"
o3d.io.write_point_cloud(ply_file_path, point_cloud)

print(f"Point cloud saved as {ply_file_path}")


# # Optional: Visualize the point cloud in Open3D (this will open a window with the 3D visualization)
# o3d.visualization.draw_geometries([point_cloud])


print("Visualization complete.")
