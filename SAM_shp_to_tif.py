import geopandas as gpd
import os
import glob
import rasterio
from rasterio.plot import show
from rasterio.features import rasterize
# input_path = "/home/phillip-4090/Documents/globalmapper_output"
input_path = "/home/phillip-4090/Music"
output_path = "/home/phillip-4090/Documents/shp_to_tif_output"

def find_files(directory, pattern):
    # List to store paths of matching files
    matching_files = []

    # Walk through the directory
    for root, dirs, files in os.walk(directory):
        for name in files:
            # Check if file matches the pattern
            if pattern in name:
                # Construct full file path and add to the list
                full_path = os.path.join(root, name)
                matching_files.append(full_path)

    return matching_files




print(find_files(input_path, ".tif"))
train_path_files = [x.split("/")[-1].split(".tif")[0] for x in find_files(input_path, ".tif")]

print(train_path_files)
# exit()
for file in train_path_files:
    shapefile = gpd.read_file(os.path.join(input_path, file.replace("img", "mask") + ".shp"))
    raster = rasterio.open(os.path.join(input_path, file + ".tif"))

# shapefile = gpd.read_file(os.path.join(train_path, "o_35803013.shp"))
# raster = rasterio.open(os.path.join(train_path, "o_35803013.tif"))

# Define the transformation from pixel coordinates to geographic coordinates
    transform = raster.transform

    # Rasterize the shapefile to the same pixel space as the raster image
    mask = rasterize(
        [(geom, 1) for geom in shapefile.geometry],
        out_shape=(raster.height, raster.width),
        transform=transform
    )


# plt.imshow(mask, cmap='gray')
# plt.show()

# To save the mask
    with rasterio.open(os.path.join(output_path, 'mask_' +  file + ".tif"), 'w', driver='GTiff', height=mask.shape[0], width=mask.shape[1], count=1, dtype=mask.dtype, crs=raster.crs, transform=transform) as dst:
        dst.write(mask, 1)

