import numpy as np
import zarr
import dask.array as da
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy import ndimage
from collections import defaultdict

from aind_hcr_data_loader.tile_data import TileData

#from ng_link import parsers
import aind_hcr_qc.parsers as parsers

def load_tile_data(tile_name: str, 
                  bucket_name: str, 
                  dataset_path: str, 
                  pyramid_level:int = 0,
                  dims_order:tuple = (1,2,0)) -> np.ndarray:
    """
    Load tile data from zarr
    (1,2,0) is the order of the dimensions in the zarr file
    """
    tile_array_loc = f"{dataset_path}{tile_name}/{pyramid_level}"
    zarr_path = f"s3://{bucket_name}/{tile_array_loc}"
    tile_data = da.from_zarr(url=zarr_path, storage_options={'anon': False}).squeeze()
    return tile_data.compute().transpose(dims_order)

# import boto3
# def download_stitch_xmls(dataset_list,
#                         save_dir=None,
#                         overwrite=False):
#     s3 = boto3.client("s3")
#     bucket_name = "aind-open-data"
#     xml_list = ["stitching_single_channel.xml", "stitching_spot_channels.xml"]

#     xmls = {}
#     for i, round_n in enumerate(dataset_list):
#         round_dict = {}
#         for xml in xml_list:
#             xml_path = f"{round_n}/{xml}"
#             round_dict[xml.split("_")[1]] = xml_path

#             fn = save_dir / xml_path
#             fn.parent.mkdir(parents=True, exist_ok=True)
#             if fn.exists() and not overwrite:
#                 print(f"Skipping {xml_path} because it already exists")
#                 continue

#             s3.download_file(
#                 Bucket=bucket_name,
#                 Key=xml_path,
#                 Filename=fn
#             )
#             print(f"Downloaded {xml_path} from S3")

#         xmls[i+1] = round_dict

#     return xmls

def get_thyme_xmls():
    r1 = "HCR_736963_2024-12-07_13-00-00"
    r2 = "HCR_736963_2024-12-13_13-00-00"
    r3 = "HCR_736963_2024-12-19_13-00-00"
    r4 = "HCR_736963_2025-01-09_13-00-00"
    r5 = "HCR_736963_2025-01-22_13-00-00"

    round_names = [r1, r2, r3, r4, r5]

    xmls = download_stitch_xmls(round_names, save_dir=Path(f'/home/matt.davis/code/hcr-stich/xml_data/'))
    return xmls


def map_channels_to_keys(tile_dict):
    """
    Create a mapping from channel names to lists of tile IDs.
    
    Args:
        tile_dict: Dictionary mapping IDs to tile names
                  Example: {0: 'Tile_X_0000_Y_0000_Z_0000_ch_405.zarr', ...}
    
    Returns:
        Dictionary mapping channel names to lists of tile IDs
        Example: {'405': [0, 1, 2, 3], '488': [4, 5, 6, 7]}
    """
    # Initialize dictionary to hold keys by channel
    channel_to_keys = {}
    
    # Process each tile
    for tile_id, tile_name in tile_dict.items():
        # Extract channel from tile name
        try:
            if '_ch_' in tile_name:
                channel = tile_name.split('_ch_')[1].split('.')[0]
            elif '.ch' in tile_name:
                channel = tile_name.split('.ch')[1].split('.')[0]
            else:
                parts = tile_name.split('_')
                for i, part in enumerate(parts):
                    if part == 'ch' and i+1 < len(parts):
                        channel = parts[i+1]
                        break
                else:
                    channel = 'unknown'
        except:
            channel = 'unknown'
        
        # Add key to channel list
        if channel not in channel_to_keys:
            channel_to_keys[channel] = []
        
        channel_to_keys[channel].append(tile_id)
    
    return channel_to_keys


def parse_bigstitcher_xml(xml_path):
    """
    Parse the XML file and return a dictionary of tile names, transforms, and other information.

    Works for both local and s3 paths.
    Works for both single and spot xml files from bigstitcher.
    """

    dataset_path = parsers.XmlParser.extract_dataset_path(xml_path=xml_path)
    # if start with /data, remove it (needed for s3 data)
    if dataset_path.startswith('/data/'):
        dataset_path = dataset_path[len('/data/'):]
    # make sure it ends with /
    if not dataset_path.endswith('/'):
        dataset_path += '/'

    tile_names = parsers.XmlParser.extract_tile_paths(xml_path=xml_path)
    tile_transforms = parsers.XmlParser.extract_tile_transforms(xml_path=xml_path)
    tile_info = parsers.XmlParser.extract_info(xml_path=xml_path)
    net_transforms = calculate_net_transforms(tile_transforms)

    print(f"Dataset path: {dataset_path}")

    channel_keys = map_channels_to_keys(tile_names)

    channels = list(channel_keys.keys())
    for channel in channels:
        print(f"{channel}: n={len(channel_keys[channel])}")

    # put all in dict
    data = {
        "tile_names": tile_names,
        "tile_transforms": tile_transforms,
        "tile_info": tile_info,
        "net_transforms": net_transforms,
        "channel_keys_map": channel_keys,
        "channels": channels,
        "dataset_path": dataset_path,
        'pixel_resolution': tile_info[0]
    }

    return data


def channel_data_from_parsed_xml(data, channel):
    """ArithmeticError

    Spot xml example:
    HCR_736963_2024-12-07_13-00-00/radial_correction.ome.zarr/
    N tiles: 278
    488: 68
    514: 68
    561: 68
    594: 68
    405: 6

    """

    channel_keys = data["channel_keys_map"]
    # assert channel in channel_keys
    assert channel in data["channels"]
    dataset_path = data["dataset_path"]
    tile_names = {key: data["tile_names"][key] for key in channel_keys[channel]}
    tile_transforms = {key: data["tile_transforms"][key] for key in channel_keys[channel]}
    #tile_info = {key: tile_info[key] for key in channel_keys[channel]}
    net_transforms = {key: data["net_transforms"][key] for key in channel_keys[channel]}

    channel_data = {
        "channel": channel,
        "channel_keys": channel_keys[channel],
        "tile_names": tile_names,
        "tile_transforms": tile_transforms,
        "net_transforms": net_transforms,
        "dataset_path": dataset_path
    }
    return channel_data


def load_tile_data(tile_name: str, 
                  bucket_name: str, 
                  dataset_path: str, 
                  pyramid_level:int = 0,
                  dims_order:tuple = (1,2,0)) -> np.ndarray:
    """
    Load tile data from zarr
    (1,2,0) is the order of the dimensions in the zarr file
    """
    tile_array_loc = f"{dataset_path}{tile_name}/{pyramid_level}"
    zarr_path = f"s3://{bucket_name}/{tile_array_loc}"
    tile_data = da.from_zarr(url=zarr_path, storage_options={'anon': False}).squeeze()
    return tile_data.compute().transpose(dims_order)


# def load_slice_data(tile_name: str, 
#                   bucket_name: str, 
#                   dataset_path: str, 
#                   slice_r
#                   pyramid_level:int = 0,
#                   dims_order:tuple = (1,2,0)) -> np.ndarray:
#     """
#     Load tile data from zarr
#     Zarrs are stored in (z,y,x) order
#     (1,2,0) is the order of the dimensions in the zarr file
#     """
#     tile_array_loc = f"{dataset_path}{tile_name}/{pyramid_level}"
#     zarr_path = f"s3://{bucket_name}/{tile_array_loc}"
#     tile_data = da.from_zarr(url=zarr_path, storage_options={'anon': False}).squeeze()
#     # get the slice at z_slice
#     slice_data = tile_data[z_slice, :, :]
#     return slice_data.compute().transpose(dims_order)


def calculate_net_transforms(
    view_transforms: dict[int, list[dict]]
) -> dict[int, np.ndarray]:
    """
    Accumulate net transform and net translation for each matrix stack.
    Net translation =
        Sum of translation vectors converted into original nominal basis
    Net transform =
        Product of 3x3 matrices
    NOTE: Translational component (last column) is defined
          wrt to the DOMAIN, not codomain.
          Implementation is informed by this given.

    NOTE: Carsons version 2/21
    Parameters
    ------------------------
    view_transforms: dict[int, list[dict]]
        Dictionary of tile ids to transforms associated with each tile.

    Returns
    ------------------------
    dict[int, np.ndarray]:
        Dictionary of tile ids to net transform.

    """

    identity_transform = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    )
    net_transforms: dict[int, np.ndarray] = defaultdict(
        lambda: np.copy(identity_transform)
    )

    for view, tfs in view_transforms.items():
        net_translation = np.zeros(3)
        net_matrix_3x3 = np.eye(3)
        curr_inverse = np.eye(3)

        for (tf) in (tfs):  # Tfs is a list of dicts containing transform under 'affine' key
            nums = [float(val) for val in tf["affine"].split(" ")]
            matrix_3x3 = np.array([nums[0::4], nums[1::4], nums[2::4]])
            translation = np.array(nums[3::4])
            
            # print(translation)
            nums = np.array(nums).reshape(3,4)
            matrix_3x3 = np.array([nums[:,0], nums[:,1], nums[:,2]]).T
            translation = np.array(nums[:,3])
            
            #old way
            net_translation = net_translation + (curr_inverse @ translation)
            net_matrix_3x3 = matrix_3x3 @ net_matrix_3x3  
            curr_inverse = np.linalg.inv(net_matrix_3x3)  # Update curr_inverse

            #new way
            #net_translation = net_translation + (translation)
            #net_matrix_3x3 = net_matrix_3x3 @ matrix_3x3 

        net_transforms[view] = np.hstack(
            (net_matrix_3x3, net_translation.reshape(3, 1))
        )

    return net_transforms

# class TileData:
#     """
#     A class for lazily loading and manipulating tile data with flexible slicing and projection options.
    
#     This class maintains the original dask array for memory efficiency and only computes data when needed.
#     It provides methods to access data in different orientations (XY, ZY, ZX) and to perform projections.
#     """
    
#     def __init__(self, tile_name, bucket_name, dataset_path, pyramid_level=0):
#         """
#         Initialize the TileData object.
        
#         Args:
#             tile_name: Name of the tile
#             bucket_name: S3 bucket name
#             dataset_path: Path to dataset in bucket
#             pyramid_level: Pyramid level to load (default 0)
#         """
#         self.tile_name = tile_name
#         self.bucket_name = bucket_name
#         self.dataset_path = dataset_path
#         self.pyramid_level = pyramid_level
#         self._data = None
#         self._loaded = False
        
#     def _load_lazy(self):
#         """Lazily load the data as a dask array without computing"""
#         if not self._loaded:
#             tile_array_loc = f"{self.dataset_path}{self.tile_name}/{self.pyramid_level}"
#             zarr_path = f"s3://{self.bucket_name}/{tile_array_loc}"
#             self._data = da.from_zarr(url=zarr_path, storage_options={'anon': False}).squeeze()
#             self._loaded = True
            
#             # Store shape information
#             self.shape = self._data.shape
#             # Assuming zarr is stored in (z,y,x) order
#             self.z_dim, self.y_dim, self.x_dim = self.shape
    
#     @property
#     def data(self):
#         """Get the full computed data in (x,y,z) order"""
#         self._load_lazy()
#         return self._data.compute().transpose(2,1,0)
    
#     @property
#     def data_raw(self):
#         """Get the full computed data in original (z,y,x) order"""
#         self._load_lazy()
#         return self._data.compute()
    
#     @property
#     def dask_array(self):
#         """Get the underlying dask array without computing"""
#         self._load_lazy()
#         return self._data

#     def connect(self):
#         """Establish connection to the data source without computing"""
#         self._load_lazy()
#         return self
    
#     def get_slice(self, index, orientation='xy', compute=True):
#         """
#         Get a 2D slice through the data in the specified orientation.
        
#         Args:
#             index: Index of the slice
#             orientation: One of 'xy', 'zy', 'zx' (default 'xy')
#             compute: Whether to compute the dask array (default True)
            
#         Returns:
#             2D numpy array or dask array
#         """
#         self._load_lazy()
        
#         if orientation == 'xy':
#             # XY slice at specific Z
#             if index >= self.z_dim:
#                 raise IndexError(f"Z index {index} out of bounds (max {self.z_dim-1})")
#             slice_data = self._data[index, :, :]
#         elif orientation == 'zy':
#             # ZY slice at specific X
#             if index >= self.x_dim:
#                 raise IndexError(f"X index {index} out of bounds (max {self.x_dim-1})")
#             slice_data = self._data[:, :, index]
#         elif orientation == 'zx':
#             # ZX slice at specific Y
#             if index >= self.y_dim:
#                 raise IndexError(f"Y index {index} out of bounds (max {self.y_dim-1})")
#             slice_data = self._data[:, index, :]
#         else:
#             raise ValueError(f"Unknown orientation: {orientation}. Use 'xy', 'zy', or 'zx'")
        
#         if compute:
#             return slice_data.compute()
#         return slice_data
    
#     def get_slice_range(self, start, end, axis='z', compute=True):
#         """
#         Get a range of slices along the specified axis.
        
#         Args:
#             start: Start index (inclusive)
#             end: End index (exclusive)
#             axis: One of 'z', 'y', 'x' (default 'z')
#             compute: Whether to compute the dask array (default True)
            
#         Returns:
#             3D numpy array or dask array
#         """
#         self._load_lazy()
        
#         if axis == 'z':
#             if end > self.z_dim:
#                 raise IndexError(f"Z end index {end} out of bounds (max {self.z_dim})")
#             slice_data = self._data[start:end, :, :]
#         elif axis == 'y':
#             if end > self.y_dim:
#                 raise IndexError(f"Y end index {end} out of bounds (max {self.y_dim})")
#             slice_data = self._data[:, start:end, :]
#         elif axis == 'x':
#             if end > self.x_dim:
#                 raise IndexError(f"X end index {end} out of bounds (max {self.x_dim})")
#             slice_data = self._data[:, :, start:end]
#         else:
#             raise ValueError(f"Unknown axis: {axis}. Use 'z', 'y', or 'x'")
        
#         if compute:
#             return slice_data.compute()
#         return slice_data
    
#     def project(self, axis='z', method='max', start=None, end=None, compute=True):
#         """
#         Project data along the specified axis using the specified method.
        
#         Args:
#             axis: One of 'z', 'y', 'x' (default 'z')
#             method: One of 'max', 'mean', 'min', 'sum' (default 'max')
#             start: Start index for projection range (default None = 0)
#             end: End index for projection range (default None = full dimension)
#             compute: Whether to compute the dask array (default True)
            
#         Returns:
#             2D numpy array or dask array
#         """
#         self._load_lazy()
        
#         # Set default range
#         if start is None:
#             start = 0
#         if end is None:
#             if axis == 'z':
#                 end = self.z_dim
#             elif axis == 'y':
#                 end = self.y_dim
#             else:
#                 end = self.x_dim
        
#         # Get the slice range
#         range_data = self.get_slice_range(start, end, axis, compute=False)
        
#         # Apply projection method
#         if method == 'max':
#             if axis == 'z':
#                 result = range_data.max(axis=0)
#             elif axis == 'y':
#                 result = range_data.max(axis=1)
#             else:  # axis == 'x'
#                 result = range_data.max(axis=2)
#         elif method == 'mean':
#             if axis == 'z':
#                 result = range_data.mean(axis=0)
#             elif axis == 'y':
#                 result = range_data.mean(axis=1)
#             else:  # axis == 'x'
#                 result = range_data.mean(axis=2)
#         elif method == 'min':
#             if axis == 'z':
#                 result = range_data.min(axis=0)
#             elif axis == 'y':
#                 result = range_data.min(axis=1)
#             else:  # axis == 'x'
#                 result = range_data.min(axis=2)
#         elif method == 'sum':
#             if axis == 'z':
#                 result = range_data.sum(axis=0)
#             elif axis == 'y':
#                 result = range_data.sum(axis=1)
#             else:  # axis == 'x'
#                 result = range_data.sum(axis=2)
#         else:
#             raise ValueError(f"Unknown method: {method}. Use 'max', 'mean', 'min', or 'sum'")
        
#         if compute:
#             return result.compute()
#         return result
    
#     def get_orthogonal_views(self, z_index=None, y_index=None, x_index=None, compute=True):
#         """
#         Get orthogonal views (XY, ZY, ZX) at the specified indices.
        
#         Args:
#             z_index: Z index for XY view (default None = middle slice)
#             y_index: Y index for ZX view (default None = middle slice)
#             x_index: X index for ZY view (default None = middle slice)
#             compute: Whether to compute the dask arrays (default True)
            
#         Returns:
#             dict with keys 'xy', 'zy', 'zx' containing the respective views
#         """
#         self._load_lazy()
        
#         # Use middle slices by default
#         if z_index is None:
#             z_index = self.z_dim // 2
#         if y_index is None:
#             y_index = self.y_dim // 2
#         if x_index is None:
#             x_index = self.x_dim // 2
        
#         # Get the three orthogonal views
#         xy_view = self.get_slice(z_index, 'xy', compute)
#         zy_view = self.get_slice(x_index, 'zy', compute)
#         zx_view = self.get_slice(y_index, 'zx', compute)
        
#         return {
#             'xy': xy_view,
#             'zy': zy_view,
#             'zx': zx_view
#         }

#     def set_pyramid_level(self, level: int):
#         """
#         Set the pyramid level and clear any loaded data.
        
#         Args:
#             level: New pyramid level to use
            
#         Returns:
#             self (for method chaining)
#         """
#         if level != self.pyramid_level:
#             self.pyramid_level = level
#             # Clear loaded data so it will be reloaded at new pyramid level
#             self._data = None
#             self._loaded = False
#         return self


#     def calculate_max_slice(self, level_to_use=2):
#         """

#         Use pyramidal level 3 and calulate the mean of the slices in all 3 dimensions,
#         report back using the index for all pyramid levels.

#         scale = int(2**pyramid_level)

#         Help to get estimates of where lots of signal is in the tile.

#         """
#         level_to_use = level_to_use
#         self.set_pyramid_level(level_to_use)

#         # first load the data
#         data = self.data

#         max_slices = {}
#         # find index of max slice in z
#         max_slice_z = data.mean(axis=0)
#         max_slice_z_index = np.unravel_index(max_slice_z.argmax(), max_slice_z.shape)
#         max_slice_y = data.mean(axis=1)
#         max_slice_y_index = np.unravel_index(max_slice_y.argmax(), max_slice_y.shape)
#         max_slice_x = data.mean(axis=2)
#         max_slice_x_index = np.unravel_index(max_slice_x.argmax(), max_slice_x.shape)

#         pyramid_levels = [0,1,2,3]

#         max_slices[level_to_use] = {
#             "z": int(max_slice_z_index[0]),
#             "y": int(max_slice_y_index[0]),
#             "x": int(max_slice_x_index[0])
#         }

#         # remove level_to_use from pyramid_levels
#         pyramid_levels.remove(level_to_use)

#         for level in pyramid_levels:
#             if level_to_use >= level:
#                 scale_factor = 2**(level_to_use - level)
#             else:
#                 print(f"level_to_use: {level_to_use}, level: {level}")
#                 scale_factor = 1/(2**(level - level_to_use))
#             max_slices[level] = {
#                 "z": int(max_slice_z_index[0] * scale_factor),
#                 "y": int(max_slice_y_index[0] * scale_factor),
#                 "x": int(max_slice_x_index[0] * scale_factor)
#             }

#         # sort keys by int value
#         max_slices = dict(sorted(max_slices.items(), key=lambda item: int(item[0])))

#         return max_slices


class PairedTiles:
    """
    Class to hold a pair of adjacent tiles and visualize their overlap in 3D.
    
    This class handles translation-only registration between tiles and provides 
    methods to visualize slices through the composite volume in any orientation.
    """
    
    def __init__(self, tile1, tile2, transform1, transform2, names=None, clip_percentiles=(1, 99)):
        """
        Initialize with two TileData objects and their transformation matrices.
        
        Args:
            tile1: First TileData object
            tile2: Second TileData object
            transform1: 4x4 transformation matrix for tile1
            transform2: 4x4 transformation matrix for tile2
            names: Optional tuple of (name1, name2) for the tiles
            clip_percentiles: Tuple of (min_percentile, max_percentile) for intensity clipping
        """
        self.tile1 = tile1
        self.tile2 = tile2
        self.transform1 = transform1.copy()
        self.transform2 = transform2.copy()
        
        self.pyramid_level1 = tile1.pyramid_level
        self.pyramid_level2 = tile2.pyramid_level
        
        self.shape1 = tile1.shape
        self.shape2 = tile2.shape
        
        # Store tile names
        if names is None:
            self.name1, self.channel1 = tile1.tile_name
            self.name2, self.channel2 = tile2.tile_name
        else:
            self.name1, self.name2 = names

        self.clip_percentiles = clip_percentiles
        
        self._scale_transforms()
        self._calculate_bounds()
        self.load_data()
    
    def _scale_transforms(self):
        """Scale the translation components of transforms based on pyramid level."""
        scale_factor1 = 2**self.pyramid_level1
        self.scaled_transform1 = self.transform1.copy()
        self.scaled_transform1[:3, 3] = self.scaled_transform1[:3, 3] / scale_factor1
        
        scale_factor2 = 2**self.pyramid_level2
        self.scaled_transform2 = self.transform2.copy()
        self.scaled_transform2[:3, 3] = self.scaled_transform2[:3, 3] / scale_factor2
        
        self.scale_factor1 = scale_factor1
        self.scale_factor2 = scale_factor2
    
    def _calculate_bounds(self):
        """Calculate the global bounds for the composite volume."""
        # For translation-only transforms, we need to find:
        # 1. The minimum coordinates (for the origin of the composite array)
        # 2. The maximum coordinates (to determine the size of the composite array)
        
        # Get corners of tile1 in global space
        shape1_zyx = np.array(self.shape1)  # (z, y, x)
        corners1 = np.array([
            [0, 0, 0],  # origin
            [shape1_zyx[0], 0, 0],  # max z
            [0, shape1_zyx[1], 0],  # max y
            [0, 0, shape1_zyx[2]],  # max x
            [shape1_zyx[0], shape1_zyx[1], 0],  # max z, y
            [shape1_zyx[0], 0, shape1_zyx[2]],  # max z, x
            [0, shape1_zyx[1], shape1_zyx[2]],  # max y, x
            [shape1_zyx[0], shape1_zyx[1], shape1_zyx[2]]  # max z, y, x
        ])
        
        # Transform corners to global space (for translation only)
        global_corners1 = corners1 + self.scaled_transform1[:3, 3]
        
        # Repeat for tile2
        shape2_zyx = np.array(self.shape2)
        corners2 = np.array([
            [0, 0, 0],
            [shape2_zyx[0], 0, 0],
            [0, shape2_zyx[1], 0],
            [0, 0, shape2_zyx[2]],
            [shape2_zyx[0], shape2_zyx[1], 0],
            [shape2_zyx[0], 0, shape2_zyx[2]],
            [0, shape2_zyx[1], shape2_zyx[2]],
            [shape2_zyx[0], shape2_zyx[1], shape2_zyx[2]]
        ])
        
        global_corners2 = corners2 + self.scaled_transform2[:3, 3]
        
        # Combine all corners and find min/max
        all_corners = np.vstack([global_corners1, global_corners2])
        self.min_corner = np.floor(np.min(all_corners, axis=0)).astype(int)
        self.max_corner = np.ceil(np.max(all_corners, axis=0)).astype(int)
        
        # Calculate composite shape
        self.composite_shape = self.max_corner - self.min_corner
        
        # Calculate offsets for each tile in the composite array
        self.offset1 = (self.scaled_transform1[:3, 3] - self.min_corner).astype(int)
        self.offset2 = (self.scaled_transform2[:3, 3] - self.min_corner).astype(int)
        
        # Print some debug info
        print(f"Composite shape: {self.composite_shape}")
        print(f"Tile1 offset: {self.offset1}")
        print(f"Tile2 offset: {self.offset2}")
    
    def load_data(self):
        """Load and transform tile data into composite space with percentile clipping."""
        composite_shape = tuple(self.composite_shape) + (3,)
        self.composite = np.zeros(composite_shape, dtype=np.float32)
        
        
        
        # data1 = self.tile1.data.copy().transpose(2,1,0)
        # data2 = self.tile2.data.copy().transpose(2,1,0)

        data1 = self.tile1.data.copy()
        data2 = self.tile2.data.copy()
        
        print(f"Tile1 shape: {data1.shape}, non-zero pixels: {np.count_nonzero(data1)}")
        print(f"Tile2 shape: {data2.shape}, non-zero pixels: {np.count_nonzero(data2)}")
        
        min_percentile, max_percentile = self.clip_percentiles
        
        # Clip and normalize tile1 data
        if np.any(data1 > 0):
            # non zero for min
            p_min1 = np.percentile(data1[data1 > 0], min_percentile)
            p_max1 = np.percentile(data1[data1 > 0], max_percentile)
            print(f"Tile1 percentiles: {min_percentile}% = {p_min1}, {max_percentile}% = {p_max1}")
            
            data1_clipped = np.clip(data1, p_min1, p_max1)
            
            # normalize to [0, 1]
            data1_norm = (data1_clipped - p_min1) / (p_max1 - p_min1) if p_max1 > p_min1 else np.zeros_like(data1_clipped)
            print(f"Tile1 normalized range: {data1_norm.min()} to {data1_norm.max()}")
        else:
            data1_norm = np.zeros_like(data1, dtype=np.float32)
            p_min1, p_max1 = 0, 0
        
        # Clip and normalize tile2 data
        if np.any(data2 > 0):
            p_min2 = np.percentile(data2[data2 > 0], min_percentile)
            p_max2 = np.percentile(data2[data2 > 0], max_percentile)
            print(f"Tile2 percentiles: {min_percentile}% = {p_min2}, {max_percentile}% = {p_max2}")
            
            data2_clipped = np.clip(data2, p_min2, p_max2)
            
            # Normalize to [0, 1]
            data2_norm = (data2_clipped - p_min2) / (p_max2 - p_min2) if p_max2 > p_min2 else np.zeros_like(data2_clipped)
            print(f"Tile2 normalized range: {data2_norm.min()} to {data2_norm.max()}")
        else:
            data2_norm = np.zeros_like(data2, dtype=np.float32)
            p_min2, p_max2 = 0, 0
        
        self.percentile_values = {
            'tile1': (p_min1, p_max1),
            'tile2': (p_min2, p_max2)
        }
        
        # put data into composite
        z1, y1, x1 = data1.shape
        oz1, oy1, ox1 = self.offset1
        
        print(f"Tile1 offset in composite: {self.offset1}")
        print(f"Tile2 offset in composite: {self.offset2}")
        
        # Calculate the actual space available in the composite array
        z1_space = min(z1, self.composite_shape[0] - oz1)
        y1_space = min(y1, self.composite_shape[1] - oy1)
        x1_space = min(x1, self.composite_shape[2] - ox1)
        
        if z1_space < z1 or y1_space < y1 or x1_space < x1:
            print(f"Warning: Tile1 extends beyond composite bounds. Clipping tile data.")
            print(f"Available space: {z1_space}, {y1_space}, {x1_space}")
        
        # Place tile1 data, clipping if necessary
        self.composite[oz1:oz1+z1_space, oy1:oy1+y1_space, ox1:ox1+x1_space, 0] = \
            data1_norm[:z1_space, :y1_space, :x1_space]
        
        # Tile2 goes into green channel
        z2, y2, x2 = data2.shape
        oz2, oy2, ox2 = self.offset2
        
        # Calculate the actual space available in the composite array
        z2_space = min(z2, self.composite_shape[0] - oz2)
        y2_space = min(y2, self.composite_shape[1] - oy2)
        x2_space = min(x2, self.composite_shape[2] - ox2)
        
        if z2_space < z2 or y2_space < y2 or x2_space < x2:
            print(f"Warning: Tile2 extends beyond composite bounds. Clipping tile data.")
            print(f"Available space: {z2_space}, {y2_space}, {x2_space}")
        
        # Place tile2 data, clipping if necessary
        self.composite[oz2:oz2+z2_space, oy2:oy2+y2_space, ox2:ox2+x2_space, 1] = \
            data2_norm[:z2_space, :y2_space, :x2_space]

        # print(f"Composite shape: {self.composite_shape}")
        # # Rotate composite 90 degrees if height < width
        # if self.composite_shape[1] < self.composite_shape[0]:
        #     self.composite = np.rot90(self.composite, k=-1, axes=(0,1))
        #     self.should_rotate = True
        # else:
        #     self.should_rotate = False
        # Create overlap mask (where both tiles have data)
        self.overlap_mask = (self.composite[..., 0] > 0) & (self.composite[..., 1] > 0)
        # if self.should_rotate:
        #     self.overlap_mask = np.rot90(self.overlap_mask, k=-1, axes=(0,1))
        
        overlap_volume = np.sum(self.overlap_mask)
        total_volume = np.prod(self.composite_shape)
        print(f"Overlap volume: {overlap_volume} voxels ({overlap_volume/total_volume:.2%} of composite)")
        
        print(f"Red channel (Tile1) max value in composite: {self.composite[..., 0].max()}")
        print(f"Green channel (Tile2) max value in composite: {self.composite[..., 1].max()}")
    
    def get_slice(self, index, orientation='xy', overlap_only=False, padding=20):
        """
        Get a slice from the composite volume.
        
        Args:
            index: Index along the slicing dimension
            orientation: One of 'xy', 'zy', 'zx'
            overlap_only: If True, show only the overlap region
            padding: Number of pixels to pad around overlap region
            
        Returns:
            RGB slice data
        """
        if orientation == 'xy':
            if index >= self.composite_shape[2]:
                raise IndexError(f"Z index {index} out of bounds (max {self.composite_shape[2]-1})")
            slice_data = self.composite[:, :, index, :]
            slice_mask = self.overlap_mask[:, :, index] if overlap_only else None
            
        elif orientation == 'zy':
            if index >= self.composite_shape[0]:
                raise IndexError(f"X index {index} out of bounds (max {self.composite_shape[0]-1})")
            slice_data = self.composite[index, :, :, :]
            slice_mask = self.overlap_mask[index, :, :] if overlap_only else None
            
        elif orientation == 'zx':
            if index >= self.composite_shape[1]:
                raise IndexError(f"Y index {index} out of bounds (max {self.composite_shape[1]-1})")
            slice_data = self.composite[:, index, :, :]
            slice_mask = self.overlap_mask[:, index, :] if overlap_only else None
        
        else:
            raise ValueError(f"Unknown orientation: {orientation}. Use 'xy', 'zy', or 'zx'")
        
        # If overlap_only is True, find the overlap region and add padding
        if overlap_only and slice_mask is not None:
            # Find the bounds of overlap region
            rows, cols = np.where(slice_mask)
            if len(rows) > 0 and len(cols) > 0:
                rmin, rmax = rows.min(), rows.max()
                cmin, cmax = cols.min(), cols.max()
                
                # Add padding
                rmin = max(0, rmin - padding)
                rmax = min(slice_data.shape[0], rmax + padding)
                cmin = max(0, cmin - padding)
                cmax = min(slice_data.shape[1], cmax + padding)
                
                # Crop to padded overlap region
                slice_data = slice_data[rmin:rmax+1, cmin:cmax+1]
            else:
                # No overlap found
                slice_data = np.zeros((1, 1, 3))
            
        return slice_data

    def visualize_slice(self, index, orientation='xy', overlap_only=False, ax=None, padding=20, rotate_z=True):
        """
        Visualize a slice from the composite volume.
        Shows the overlap region with padding if overlap_only is True.
        
        Args:
            index: Index along the slicing dimension
            orientation: One of 'xy', 'zy', 'zx' (default 'xy')
            overlap_only: If True, show only the overlap region
            ax: Matplotlib axis to plot on
            padding: Number of pixels to pad around overlap region
            rotate_z: If True, display Z as the vertical axis in XZ and YZ views
        """
        slice_data = self.get_slice(index, orientation, overlap_only, padding=padding)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        else:
            fig = ax.figure
        
        # Rotate the slice if needed
        if rotate_z and orientation in ['zx', 'zy']:
            slice_data = np.rot90(slice_data,3)
        
        ax.imshow(slice_data)
        
        # Adjust axis labels based on rotation
        if rotate_z:
            axis_labels = {
                'xy': ('Y', 'X', 'Z'),  # XY view unchanged
                'zy': ('Z', 'Y', 'X'),  # YZ view becomes ZY
                'zx': ('Z', 'X', 'Y')   # XZ view becomes ZX
            }
        else:
            axis_labels = {
                'xy': ('Y', 'X', 'Z'),
                'zy': ('Y', 'Z', 'X'),
                'zx': ('X', 'Z', 'Y')
            }
        
        ylabel, xlabel, slice_dim = axis_labels[orientation]
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{orientation.upper()} slice at {slice_dim}={index}")
        
        return fig, ax

    def visualize_orthogonal_views(self, z_slice=None, y_slice=None, x_slice=None, 
                                 overlap_only=False, padding=20, rotate_z=True):
        """
        Visualize orthogonal views of the composite volume.
        
        Args:
            z_slice, y_slice, x_slice: Slice indices
            overlap_only: If True, show only the overlap region
            padding: Number of pixels to pad around overlap region
            rotate_z: If True, display Z as the vertical axis in XZ and YZ views
        """
        # Use middle slices by default
        if x_slice is None:
            x_slice = self.composite_shape[0] // 2
        if y_slice is None:
            y_slice = self.composite_shape[1] // 2 
        if z_slice is None:
            z_slice = self.composite_shape[2] // 2
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot views with padding
        self.visualize_slice(z_slice, 'xy', overlap_only, axes[0], padding=padding, rotate_z=rotate_z)
        self.visualize_slice(y_slice, 'zx', overlap_only, axes[1], padding=padding, rotate_z=rotate_z)
        self.visualize_slice(x_slice, 'zy', overlap_only, axes[2], padding=padding, rotate_z=rotate_z)
        
        plt.suptitle(f"Orthogonal Views of Paired Tiles\n{self.name1} (red) and {self.name2} (green)", fontsize=16)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        
        return fig, axes


def visualize_multichannel_paired_tiles(tile1_name, tile2_name, data, 
                                      channels=['405', '488', '514', '561', '594', '638'],
                                      pyramid_level=2, overlap_only=False, padding='auto', rotate_z=True):
    """
    Visualize orthogonal views for all channels of a pair of tiles.
    
    Args:
        tile1_name: Name of first tile
        tile2_name: Name of second tile
        data: Parsed XML data
        channels: List of channels to visualize
        pyramid_level: Pyramid level to load
        overlap_only: Whether to show only overlap regions
        padding: Padding around overlap regions
        rotate_z: Whether to rotate Z axis to vertical
    """
    sns.set_context("talk")
    # Create figure with n_channels rows and 3 columns
    n_channels = len(channels)
    fig, axes = plt.subplots(n_channels, 3, figsize=(18, 10*n_channels))
    
    # Get tile IDs
    tile1_id = list(data["tile_names"].keys())[list(data["tile_names"].values()).index(tile1_name)]
    tile2_id = list(data["tile_names"].keys())[list(data["tile_names"].values()).index(tile2_name)]

    # Get transforms
    transform1 = data["net_transforms"][tile1_id]
    transform2 = data["net_transforms"][tile2_id]

    
    parsed_name1, ch1 = parse_tile_name(tile1_name)
    parsed_name2, ch2 = parse_tile_name(tile2_name)


    if padding == 'auto':
        padding = 16 * 2**(3-pyramid_level)
    
    # Function to replace channel in tile name
    def replace_channel(tile_name, new_channel):
        parts = tile_name.split('_ch_')
        return f"{parts[0]}_ch_{new_channel}.zarr"
    
    # Process each channel
    for i, channel in enumerate(channels):
        # Replace channel in tile names
        ch_tile1 = replace_channel(tile1_name, channel)
        ch_tile2 = replace_channel(tile2_name, channel)

        print(f"\nChannel {channel}\n----------")
        print(f"Tile1: {ch_tile1}")
        print(f"Tile2: {ch_tile2}")
        
        try:
            # Create paired tiles for this channel
            paired_tiles = PairedTiles(
                tile1=TileData(ch_tile1, "aind-open-data", data["dataset_path"], pyramid_level).connect(),
                tile2=TileData(ch_tile2, "aind-open-data", data["dataset_path"], pyramid_level).connect(),
                transform1=transform1,
                transform2=transform2,
                names=(f"{ch_tile1} ({channel})", f"{ch_tile2} ({channel})")
            )
            
            # Load the data
            paired_tiles.load_data()
            
            # Get middle slices
            z_slice = paired_tiles.composite_shape[2] // 2
            y_slice = paired_tiles.composite_shape[1] // 2
            x_slice = paired_tiles.composite_shape[0] // 2
            
            # Plot the three views
            paired_tiles.visualize_slice(z_slice, 'xy', overlap_only, axes[i,0], padding=padding, rotate_z=rotate_z)
            paired_tiles.visualize_slice(y_slice, 'zx', overlap_only, axes[i,1], padding=padding, rotate_z=rotate_z)
            paired_tiles.visualize_slice(x_slice, 'zy', overlap_only, axes[i,2], padding=padding, rotate_z=rotate_z)
            
            # Make titles more compact
            axes[i,0].set_title(f"XY@Z={z_slice}", pad=2)
            
            # Add clims above middle plot
            red_min, red_max = paired_tiles.percentile_values['tile1']
            green_min, green_max = paired_tiles.percentile_values['tile2']
            clim_text = f"Ch {channel}\nRed min/max: {int(red_min)}-{int(red_max)}\nGreen min/max: {int(green_min)}-{int(green_max)}"
            axes[i,1].set_title(f"{clim_text}\nZX@Y={y_slice}", pad=2)
            
            axes[i,2].set_title(f"ZY@X={x_slice}", pad=2)
            # add channel name to left of XY plot
            

        except Exception as e:
            print(f"Error processing channel {channel}: {str(e)}")
            # show traceback
            # Create empty plots for this row
            # for j in range(3):
            #     axes[i,j].imshow(np.zeros((100,100,3)))
            #     axes[i,j].set_title(f"Channel {channel} - Error")
    
    #plt.suptitle(f"Orthogonal Views of Paired Tiles Across Channels\n{data['dataset_path']}\n{parsed_name1} and {parsed_name2}", fontsize=20)
    plt.suptitle(f"{data['dataset_path']}\n{parsed_name1} and {parsed_name2}", fontsize=20)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)

    return fig, axes


def visualize_paired_tiles(tile1_name, tile2_name, data, pyramid_level=1, 
                           bucket_name='aind-open-data', overlap_only=False, padding='auto'):

    # 20 is good for pyramid level 3, scale up for lower levels by factor of 4
    if padding == 'auto':
        padding = 16 * 2**(3-pyramid_level)
    # Create TileData objects
    tile1 = TileData(tile1_name, bucket_name, data["dataset_path"], pyramid_level=pyramid_level).connect()
    tile2 = TileData(tile2_name, bucket_name, data["dataset_path"], pyramid_level=pyramid_level).connect()

    # Get tile IDs
    tile1_id = list(data["tile_names"].keys())[list(data["tile_names"].values()).index(tile1_name)]
    tile2_id = list(data["tile_names"].keys())[list(data["tile_names"].values()).index(tile2_name)]

    # Get transforms
    transform1 = data["net_transforms"][tile1_id]
    transform2 = data["net_transforms"][tile2_id]
    
    # Parse tile names for display
    parsed_name1 = parse_tile_name(tile1_name)
    parsed_name2 = parse_tile_name(tile2_name)
    
    # Create PairedTiles object
    paired = PairedTiles(tile1, tile2, transform1, transform2, names=(parsed_name1, parsed_name2))
    
    # Visualize orthogonal views
    fig, axes = paired.visualize_orthogonal_views(overlap_only=overlap_only, padding=padding)
    
    return paired, fig, axes

def get_net_transforms(xml, tile_name):
    """
    Get the net transforms for a pair of tiles from the data dictionary.
    
    Parameters:
    -----------
    xml : dict
        Dictionary containing dataset information including tile_names and net_transforms
    tile_name : str
        Name of the first tile

        
    Returns:
    --------
    transform - The net transform for the tile
    """
    print(f"Getting net transforms for tile: {tile_name}")
    print(f"Available tile names: {list(xml['tile_names'].values())}")
    match_ind = list(xml["tile_names"].values()).index(tile_name)
    if match_ind < 0 or match_ind >= len(xml["tile_names"]):
        raise ValueError(f"Tile name '{tile_name}' not found in data['tile_names']")
    match_ind = int(match_ind)

    # Get tile IDs
    tile_id = list(xml["tile_names"].keys())[match_ind]

    # Get transforms
    transform = xml["net_transforms"][tile_id]
    
    return transform

def create_paired_tiles(data, tile1_name, tile2_name, bucket_name, pyramid_level=2):
    """
    Create a PairedTiles object from two tile names and a data dictionary.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing dataset information including tile_names, net_transforms, and dataset_path
        From BigSticher parsed XML
    tile1_name : str
        Name of the first tile
    tile2_name : str
        Name of the second tile
    bucket_name : str
        Name of the S3 bucket
    pyramid_level : int, optional
        Pyramid level to use (default: 0)
        
    Returns:
    --------
    paired : io_utils.PairedTiles
        PairedTiles object containing the two connected tiles with transforms
    """
    # Create TileData objects
    tile1 = TileData(tile1_name, bucket_name, data["dataset_path"], pyramid_level=pyramid_level).transpose((2,1,0))
    tile2 = TileData(tile2_name, bucket_name, data["dataset_path"], pyramid_level=pyramid_level).transpose((2,1,0))

    # Get tile IDs and transforms
    transform1 = get_net_transforms(data, tile1_name)
    transform2 = get_net_transforms(data, tile2_name)
    # Parse tile names for display
    parsed_name1, ch1 = parse_tile_name(tile1_name)
    parsed_name2, ch2 = parse_tile_name(tile2_name)

    # Create PairedTiles object
    paired = PairedTiles(tile1, tile2, transform1, transform2, names=(parsed_name1, parsed_name2))
    
    return paired

def fig_tile_overlap_4_slices(tile1_name, 
                                 tile2_name, 
                                 data,
                                 channel='405', # or "spots"
                                 pyramid_level=1, 
                                 bucket_name='aind-open-data',
                                 save=False,
                                 output_dir=None,
                                 verbose=False):
    
    # Create TileData objects
    if channel == 'spots-avg':
        spots_channels = ["488", "514", "561", "594", "638"]
        
        # Pre-create all TileData objects
        tile_pairs = []
        
        for ch in spots_channels:
            tile1_name_ch = tile1_name.replace('_ch_405.zarr', f'_ch_{ch}.zarr')
            tile2_name_ch = tile2_name.replace('_ch_405.zarr', f'_ch_{ch}.zarr')
            
            try:
                tile1_obj = TileData(tile1_name_ch, bucket_name, data["dataset_path"], pyramid_level=pyramid_level)
                tile2_obj = TileData(tile2_name_ch, bucket_name, data["dataset_path"], pyramid_level=pyramid_level)
                tile_pairs.append((tile1_obj, tile2_obj))
                if verbose:
                    print(f"Successfully loaded channel {ch}")
            except Exception as e:
                print(f"Error loading tile {tile1_name_ch} or {tile2_name_ch}: {e}")
                continue
        if not tile_pairs:
            raise ValueError("No valid tile pairs loaded for spots averaging")
        # Start with first pair
        tile1, tile2 = tile_pairs[0]
        
        # Average remaining pairs
        for i in range(1, len(tile_pairs)):
            tile1_next, tile2_next = tile_pairs[i]
            tile1 = tile1.average(tile1_next)
            tile2 = tile2.average(tile2_next)
            
            # Explicit cleanup
            del tile1_next, tile2_next
            
            # Force garbage collection every few iterations
            if i % 2 == 0:
                import gc
                gc.collect()
        
        # Final cleanup
        del tile_pairs
        import gc
        gc.collect()

    elif channel == '405':
        tile1 = TileData(tile1_name, bucket_name, data["dataset_path"], pyramid_level=pyramid_level)
        tile2 = TileData(tile2_name, bucket_name, data["dataset_path"], pyramid_level=pyramid_level)

    padding_dict = {0: 75, 1: 50, 2: 30, 3: 10}
    padding = padding_dict.get(pyramid_level, 50)  # Default to 50 if not found

    # look up the values in data["tile_names"] to get the ids (which is the key)
    tile1_id = list(data["tile_names"].keys())[list(data["tile_names"].values()).index(tile1_name)]
    tile2_id = list(data["tile_names"].keys())[list(data["tile_names"].values()).index(tile2_name)]

    # Get transforms for the tiles
    transform1 = data["net_transforms"][tile1_id]
    transform2 = data["net_transforms"][tile2_id]

    n_cols = 4
    size = 6
    fig, axes = plt.subplots(1, n_cols, figsize=(size,size*1.25), sharey=True, constrained_layout=True)
    axes = axes.flatten()

    # Calculate z-slices at 20%, 40%, 60%, and 80% through the z dimension
    z_min = max(0, min(tile1.shape[0], tile2.shape[0]))
    z_slices = [int(z_min * p) for p in [0.2, 0.4, 0.6, 0.8]]

    for i, z_slice in enumerate(z_slices):
        result = visualize_tile_overlap(tile1, tile2, transform1, transform2, 
                                        z_slice=z_slice, padding=padding, verbose=verbose)
        composite = result['composite']
        overlap_shape = composite.shape
        # transpose if a vertically adjacent tile
        if overlap_shape[1] > overlap_shape[0]:
            composite = composite.transpose(1, 0, 2)
        
        axes[i].imshow(composite, aspect='auto')
        axes[i].set_title(f'Z={z_slice}')
        axes[i].axis('on')
    tile1_name, ch1 = parse_tile_name(tile1.tile_name)
    tile2_name, ch2 = parse_tile_name(tile2.tile_name)

    plt.suptitle(f'Tile Overlap - Red={tile1_name}, Green={tile2_name} Ch={channel} Pyr={pyramid_level}', fontsize=16)
    plt.subplots_adjust(top=0.15)
    if save and output_dir:
        from pathlib import Path
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filename = f'tile_overlap_4slices_{tile1_name}_{tile2_name}_{channel}_pyr{pyramid_level}.png'
        filepath = output_path / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {filepath}")
        plt.close()
    else:
        plt.show()

# def fig_tile_overlap_4_slices(tile1_name, 
#                                  tile2_name, 
#                                  data,
#                                  channel='405', # or "spots"
#                                  pyramid_level=1, 
#                                  bucket_name='aind-open-data',
#                                  save=False,
#                                  output_dir=None,
#                                  verbose=False):
    
#     # Create TileData objects
#     if channel == 'spots-avg':
#         spots_channels = ["488", "514", "561", "594", "638"]
#         # get tile data for each channel and average them
#         for ch in spots_channels:
#             tile1_name_ch = tile1_name.replace('_ch_405.zarr', f'_ch_{ch}.zarr')
#             tile2_name_ch = tile2_name.replace('_ch_405.zarr', f'_ch_{ch}.zarr')
            
#             try:
#                 if ch == spots_channels[0]:
#                     tile1 = TileData(tile1_name_ch, bucket_name, data["dataset_path"], pyramid_level=pyramid_level)
#                     tile2 = TileData(tile2_name_ch, bucket_name, data["dataset_path"], pyramid_level=pyramid_level)
#                     previous_tile_data = None
#                 else:
#                     tile1 = tile1.average(TileData(tile1_name_ch, bucket_name, data["dataset_path"], pyramid_level=pyramid_level))
#                     tile2 = tile2.average(TileData(tile2_name_ch, bucket_name, data["dataset_path"], pyramid_level=pyramid_level))
#             except Exception as e:
#                 print(f"Error loading tile {tile1_name_ch} or {tile2_name_ch}: {e}")
#                 continue
            
#             # check if tile data changed values, max should be diff
#             print(np.max(previous_tile_data), np.max(tile1.data))
#             previous_tile_data = tile1.data.copy()
#             if np.max(previous_tile_data) != np.max(tile1.data):
#                 print(f"Tile data changed for {tile1_name_ch} or {tile2_name_ch}, max value: {np.max(tile1.data)}")
#             previous_tile_data = tile1.data.copy()

#     elif channel == '405':
#         tile1 = TileData(tile1_name, bucket_name, data["dataset_path"], pyramid_level=pyramid_level)
#         tile2 = TileData(tile2_name, bucket_name, data["dataset_path"], pyramid_level=pyramid_level)

#     padding_dict = {0: 75, 1: 50, 2: 30, 3: 10}
#     padding = padding_dict.get(pyramid_level, 50)  # Default to 50 if not found

#     # look up the values in data["tile_names"] to get the ids (which is the key)
#     tile1_id = list(data["tile_names"].keys())[list(data["tile_names"].values()).index(tile1_name)]
#     tile2_id = list(data["tile_names"].keys())[list(data["tile_names"].values()).index(tile2_name)]

#     # Get transforms for the tiles
#     transform1 = data["net_transforms"][tile1_id]
#     transform2 = data["net_transforms"][tile2_id]

#     n_cols = 4
#     size = 6
#     fig, axes = plt.subplots(1, n_cols, figsize=(size,size*1.25), sharey=True, constrained_layout=True)
#     axes = axes.flatten()

#     # Calculate z-slices at 20%, 40%, 60%, and 80% through the z dimension
#     z_min = max(0, min(tile1.shape[0], tile2.shape[0]))
#     z_slices = [int(z_min * p) for p in [0.2, 0.4, 0.6, 0.8]]

#     for i, z_slice in enumerate(z_slices):
#         result = visualize_tile_overlap(tile1, tile2, transform1, transform2, 
#                                         z_slice=z_slice, padding=padding, verbose=verbose)
#         composite = result['composite']
#         overlap_shape = composite.shape
#         # transpose if a vertically adjacent tile
#         if overlap_shape[1] > overlap_shape[0]:
#             composite = composite.transpose(1, 0, 2)
        
#         axes[i].imshow(composite, aspect='auto')
#         axes[i].set_title(f'Z={z_slice}')
#         axes[i].axis('on')
#     tile1_name, ch1 = parse_tile_name(tile1.tile_name)
#     tile2_name, ch2 = parse_tile_name(tile2.tile_name)

#     plt.suptitle(f'Tile Overlap - Red={tile1_name}, Green={tile2_name} Ch={channel} Pyr={pyramid_level}', fontsize=16)
#     plt.subplots_adjust(top=0.15)
#     if save and output_dir:
#         from pathlib import Path
#         output_path = Path(output_dir)
#         output_path.mkdir(parents=True, exist_ok=True)
#         filename = f'tile_overlap_4slices_{tile1_name}_{tile2_name}_{channel}_pyr{pyramid_level}.png'
#         filepath = output_path / filename
#         plt.savefig(filepath, dpi=150, bbox_inches='tight')
#         print(f"Figure saved to: {filepath}")
#         plt.close()
#     else:
#         plt.show()


# ------------------------------------------------------------------------------------------------
# From: sticht_utils.py
# ------------------------------------------------------------------------------------------------




def analyze_tile_grid(tile_dict, plot=True):
    """
    Analyze the tile grid structure and show coverage with visualization
    Args:
        tile_dict: Dictionary of tile names key: tile_id, value: tile_name
        Example. {0: 'Tile_X_0000_Y_0000_Z_0000_ch_405.zarr', 1: 'Tile_X_0001_Y_0000_Z_0000_ch_405.zarr'}
        plot: Whether to show the coverage plot
    Returns:
        dict: Detailed information about the tile grid
    """
    # Extract coordinates
    coords = []
    for _, tile in tile_dict.items():
        base_name = tile.split('_ch_')[0]
        parts = base_name.split('_')
        x = int(parts[2])
        y = int(parts[4])
        z = int(parts[6])
        coords.append((x, y, z))
    
    # Find dimensions
    x_coords = {x for x, _, _ in coords}
    y_coords = {y for _, y, _ in coords}
    z_coords = {z for _, _, z in coords}
    
    x_dim = max(x_coords) + 1
    y_dim = max(y_coords) + 1
    z_dim = max(z_coords) + 1
    
    # Create coverage map
    coverage = np.zeros((y_dim, x_dim))  # Note: y_dim first for correct plotting
    for x, y, _ in coords:
        coverage[y, x] = 1
    
    if plot:
        plt.figure(figsize=(12, 8))
        plt.imshow(coverage, cmap='RdYlGn', interpolation='nearest')
        plt.colorbar(label='Tile Present')
        plt.title(f'Tile Coverage Map ({len(tile_dict)} tiles)')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        
        # Add grid lines
        plt.grid(True, which='major', color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # Add coordinate labels
        for i in range(x_dim):
            for j in range(y_dim):
                color = 'white' if coverage[j, i] == 0 else 'black'
                plt.text(i, j, f'({i},{j})', ha='center', va='center', color=color)
        
        plt.show()
    
    # Calculate statistics
    theoretical_tiles = x_dim * y_dim * z_dim
    actual_tiles = len(tile_dict)
    
    # Find missing coordinates
    all_coords = {(x, y, z) for x in range(x_dim) 
                           for y in range(y_dim) 
                           for z in range(z_dim)}
    present_coords = set(coords)
    missing_coords = all_coords - present_coords
    
    return {
        'dimensions': (x_dim, y_dim, z_dim),
        'theoretical_tiles': theoretical_tiles,
        'actual_tiles': actual_tiles,
        'coverage_percentage': (actual_tiles / theoretical_tiles) * 100,
        'x_range': (min(x_coords), max(x_coords)),
        'y_range': (min(y_coords), max(y_coords)),
        'z_range': (min(z_coords), max(z_coords)),
        'missing_coords': sorted(missing_coords),
        'present_coords': sorted(present_coords),
        'coverage_map': coverage
    }



def plot_tile_transforms(tile_dict, transforms, coverage_map):
    """
    Plot tiles with arrows showing their transformation vectors
    Args:
        tile_dict: Dictionary mapping IDs to tile names
        transforms: defaultdict of transformation matrices as numpy arrays
        coverage_map: 2D numpy array showing tile presence
    """
    y_dim, x_dim = coverage_map.shape
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot the tile coverage base
    im = ax.imshow(coverage_map, cmap='RdYlGn', interpolation='nearest', alpha=0.3)
    
    # Keep track of magnitudes for color scaling
    magnitudes = []
    
    # First pass to get magnitude range
    for tile_id, tile_name in tile_dict.items():
        transform = transforms[tile_id]
        dx = transform[0, 3]
        dy = transform[1, 3]
        magnitudes.append(np.sqrt(dx**2 + dy**2))
    
    max_magnitude = max(magnitudes)
    
    # Plot transformation vectors for each tile
    for tile_id, tile_name in tile_dict.items():
        # Get tile coordinates
        parts = tile_name.split('_ch_')[0].split('_')
        tile_x = int(parts[2])
        tile_y = int(parts[4])
        
        # Get transformation
        transform = transforms[tile_id]
        dx = transform[0, 3]
        dy = transform[1, 3]
        
        # Calculate magnitude
        magnitude = np.sqrt(dx**2 + dy**2)
        
        # Color based on magnitude
        color = plt.cm.viridis(magnitude / max_magnitude)
        
        # Scale factor for visualization (adjust as needed)
        scale = 1/10000  # This might need adjustment based on your transform magnitudes
        
        # Plot arrow from tile center
        ax.arrow(tile_x, tile_y,           # Start at tile position
                dx * scale, dy * scale,     # Scaled displacement
                head_width=0.1,
                head_length=0.1,
                fc=color, ec=color,
                alpha=0.7)
        
        # Add magnitude text
        ax.text(tile_x, tile_y, f'{magnitude:.0f}', 
               ha='center', va='bottom', color='black')
    
    ax.set_title('Tile Transformation Vectors')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.grid(True, alpha=0.3)
    
    # Add scale reference
    scale_length = 1000 * scale  # Length in plot units
    ax.arrow(x_dim-1, y_dim-1, scale_length, 0,
            head_width=0.1, head_length=0.1,
            fc='red', ec='red',
            label='Scale')
    ax.text(x_dim-1, y_dim-1.3, f'1000 pixels', ha='center')
    
    # Add colorbar
    norm = plt.Normalize(vmin=0, vmax=max_magnitude)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label='Transform magnitude (pixels)')
    
    plt.tight_layout()
    plt.show()



def plot_transform_heatmaps(tile_dict, 
                            transforms, 
                            coverage_map, 
                            remove_nominal_transform=False, 
                            tile_dims=None,
                            overlap=0.0):
    """
    Plot three heatmaps showing X, Y, and Z transformations with colormaps centered on zero
    Args:
        tile_dict: Dictionary mapping IDs to tile names
        transforms: defaultdict of transformation matrices as numpy arrays
        coverage_map: 2D numpy array showing tile presence
        remove_nominal_transform: Whether to remove the nominal transform from the heatmaps

    The nominal transform is the transform that maps the tile to the nominal coordinate system
    tile_dims: tuple of (x_dim, y_dim, z_dim) representing the dimensions of tile in pixels
    We use the tile_dims to remove the nominal transform from the heatmaps. 
    For X transforms, we remove the x_dim from the transform. We need to get the X coordinate index, starting from the middle.
    So if we have 7 tiles in the X direction, the middle tile is index 3. The scale index vector in [-3,-2,-1,0,1,2,3] * tile_dims[0]
   
   For Y transforms, we remove the y_dim from the transform. 
    """
    y_dim, x_dim = coverage_map.shape
    
    # assert that tile_dims is not None if remove_nominal_transform is True
    if remove_nominal_transform and tile_dims is None:
        raise ValueError("tile_dims must be provided if remove_nominal_transform is True")
    
    # Create arrays to store the transform values
    x_transforms = np.full_like(coverage_map, np.nan, dtype=float)
    y_transforms = np.full_like(coverage_map, np.nan, dtype=float)
    z_transforms = np.full_like(coverage_map, np.nan, dtype=float)

    x_scale_index = np.arange(x_dim) - (x_dim - 1) / 2
    y_scale_index = np.arange(y_dim) - (y_dim - 1) / 2
    x_scale_index = x_scale_index * tile_dims[0] * (1 - overlap)
    y_scale_index = y_scale_index * tile_dims[1] * (1 - overlap)
    
    # Fill in the transform values
    for tile_id, tile_name in tile_dict.items():
        parts = tile_name.split('_ch_')[0].split('_')
        tile_x = int(parts[2])
        tile_y = int(parts[4])
        
        transform = transforms[tile_id]
        x_transforms[tile_y, tile_x] = transform[0, 3]
        y_transforms[tile_y, tile_x] = transform[1, 3]
        z_transforms[tile_y, tile_x] = transform[2, 3]

        if remove_nominal_transform:
            x_transforms[tile_y, tile_x] = x_transforms[tile_y, tile_x] - x_scale_index[tile_x]
            y_transforms[tile_y, tile_x] = y_transforms[tile_y, tile_x] - y_scale_index[tile_y]
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Function to get symmetric vmin/vmax for centered colormap
    def get_symmetric_limits(data):
        valid_data = data[~np.isnan(data)]
        abs_max = np.max(np.abs(valid_data))
        return -abs_max, abs_max
    
    # Plot X transforms
    vmin_x, vmax_x = get_symmetric_limits(x_transforms)
    im1 = ax1.imshow(x_transforms, cmap='RdBu', vmin=vmin_x, vmax=vmax_x)
    ax1.set_title('X Transforms')
    ax1.set_xlabel('X coordinate')
    ax1.set_ylabel('Y coordinate')
    fig.colorbar(im1, ax=ax1, label='X translation (pixels)')
    
    # Plot Y transforms
    vmin_y, vmax_y = get_symmetric_limits(y_transforms)
    im2 = ax2.imshow(y_transforms, cmap='RdBu', vmin=vmin_y, vmax=vmax_y)
    ax2.set_title('Y Transforms')
    ax2.set_xlabel('X coordinate')
    ax2.set_ylabel('Y coordinate')
    fig.colorbar(im2, ax=ax2, label='Y translation (pixels)')
    
    # Plot Z transforms
    vmin_z, vmax_z = get_symmetric_limits(z_transforms)
    im3 = ax3.imshow(z_transforms, cmap='RdBu', vmin=vmin_z, vmax=vmax_z)
    ax3.set_title('Z Transforms')
    ax3.set_xlabel('X coordinate')
    ax3.set_ylabel('Y coordinate')
    fig.colorbar(im3, ax=ax3, label='Z translation (pixels)')
    
    # Add grid lines
    for ax in [ax1, ax2, ax3]:
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return x_transforms, y_transforms, z_transforms


def get_tile_grid_dimensions(tile_names):
    """
    Determine the dimensions of the tile grid (max X, Y, Z coordinates)
    Args:
        tile_names: List of tile names in format 'Tile_X_0000_Y_0000_Z_0000_ch_405.zarr'
    Returns:
        tuple: (max_x + 1, max_y + 1, max_z + 1) representing grid dimensions
    """
    x_coords = set()
    y_coords = set()
    z_coords = set()
    
    for tile in tile_names:
        # Split by '_ch_' to remove suffix and then split remaining parts
        base_name = tile.split('_ch_')[0]
        parts = base_name.split('_')
        
        x_coords.add(int(parts[2]))  # X coordinate
        y_coords.add(int(parts[4]))  # Y coordinate
        z_coords.add(int(parts[6]))  # Z coordinate
    
    # Add 1 to each max coordinate to get dimensions
    dimensions = (
        max(x_coords) + 1,
        max(y_coords) + 1,
        max(z_coords) + 1
    )
    
    return dimensions


def parse_tile_name(tile_name):
    """Extract X, Y, Z coordinates from a tile name like 'Tile_X_0000_Y_0000_Z_0000_ch_405.zarr'"""
    # Remove the suffix and split by underscore
    base_name = get_base_tile_name(tile_name)
    parts = base_name.split('_')
    x = int(parts[2])
    y = int(parts[4])
    z = int(parts[6])
    ch = int(tile_name.split('_ch_')[1].split('.zarr')[0])
    return (x, y, z), ch

def get_base_tile_name(tile_name):
    """Get the base tile name without the channel and extension"""
    return tile_name.split('_ch_')[0]


def get_adjacent_tiles(tile_name, existing_tiles, include_diagonals=True):
    """
    Get the names of adjacent tiles that exist in the provided list
    Args:
        tile_name: Name of the tile in format 'Tile_X_0000_Y_0000_Z_0000_ch_405.zarr'
        existing_tiles: List of valid tile names to check against
        include_diagonals: If True, includes diagonal neighbors
                          If False, only cardinal directions
    """
    (x, y, z), ch = parse_tile_name(tile_name)
    
    # Convert existing_tiles to base names for comparison
    existing_base_tiles = {get_base_tile_name(t) for t in existing_tiles}
    
    adjacent_tiles = []
    
    # Choose which directions to check
    if include_diagonals:
        directions = [(dx, dy) for dx in [-1, 0, 1] 
                             for dy in [-1, 0, 1] 
                             if not (dx == 0 and dy == 0)]
    else:
        directions = [(0, 1),   # North
                     (1, 0),    # East
                     (0, -1),   # South
                     (-1, 0)]   # West
    
    # Generate and filter adjacent tile names
    for dx, dy in directions:
        adj_x = str(x + dx).zfill(4)
        adj_y = str(y + dy).zfill(4)
        adj_z = str(z).zfill(4)
        
        adjacent_base_name = f"Tile_X_{adj_x}_Y_{adj_y}_Z_{adj_z}"
        if adjacent_base_name in existing_base_tiles:
            # Find the full tile name from the original list
            full_name = [t for t in existing_tiles 
                        if get_base_tile_name(t) == adjacent_base_name][0]
            adjacent_tiles.append(full_name)
    
    return adjacent_tiles

def get_all_adjacent_pairs(tile_names, include_diagonals=False):
    """
    Generate all pairs of adjacent tiles in the dataset
    
    Args:
        tile_names: List of tile names in format 'Tile_X_0000_Y_0000_Z_0000_ch_405.zarr'
        include_diagonals: If True, includes diagonal neighbors
                          If False, only cardinal directions
    
    Returns:
        list: List of tuples containing pairs of adjacent tile names
              Each pair is ordered (tile1, tile2) where tile1 has a lower 
              index in the original tile_names list than tile2
    """
    pairs = []
    
    # Convert to list if dictionary is provided
    if isinstance(tile_names, dict):
        tile_names = list(tile_names.values())
    
    # For each tile, find its adjacent tiles
    for i, tile in enumerate(tile_names):
        adjacent_tiles = get_adjacent_tiles(tile, tile_names, include_diagonals)
        
        # Only keep pairs where the adjacent tile has a higher index
        # This prevents duplicate pairs and ensures consistent ordering
        for adj_tile in adjacent_tiles:
            j = tile_names.index(adj_tile)
            if i < j:  # Only add if current tile index is lower
                pairs.append((tile, adj_tile))
    
    return pairs

def analyze_adjacent_pairs(pairs, tile_names, transforms):
    """
    Analyze the transformation differences between adjacent tile pairs
    
    Args:
        pairs: List of tuples containing pairs of adjacent tile names
        transforms: Dictionary mapping tile names or IDs to transformation matrices
        tile_names: Dictionary mapping tile IDs to tile names
    
    Returns:
        dict: Statistics about the transformations between adjacent tiles
              Including mean, std, min, max of translation differences
    """
    # Store differences for each dimension
    x_diffs = []
    y_diffs = []
    z_diffs = []
    
    for tile1, tile2 in pairs:
        # Get transforms for both tiles
        if isinstance(transforms, dict):
            # get index of tile1 and tile2 in tile_names
            idx1 = list(tile_names.keys())[list(tile_names.values()).index(tile1)]
            idx2 = list(tile_names.keys())[list(tile_names.values()).index(tile2)]
            t1 = transforms[idx1]
            t2 = transforms[idx2]
        
        # Calculate differences in translation components
        x_diff = abs(t1[0, 3] - t2[0, 3])
        y_diff = abs(t1[1, 3] - t2[1, 3])
        z_diff = abs(t1[2, 3] - t2[2, 3])
        
        x_diffs.append(x_diff)
        y_diffs.append(y_diff)
        z_diffs.append(z_diff)
    
    # Calculate statistics
    stats = {
        'x_translation': {
            'mean': np.mean(x_diffs),
            'std': np.std(x_diffs),
            'min': np.min(x_diffs),
            'max': np.max(x_diffs),
            'x_diffs': x_diffs
        },
        'y_translation': {
            'mean': np.mean(y_diffs),
            'std': np.std(y_diffs),
            'min': np.min(y_diffs),
            'max': np.max(y_diffs),
            'y_diffs': y_diffs
        },
        'z_translation': {
            'mean': np.mean(z_diffs),
            'std': np.std(z_diffs),
            'min': np.min(z_diffs),
            'max': np.max(z_diffs),
            'z_diffs': z_diffs
        },
        'n_pairs': len(pairs)
    }
    
    return stats


def extract_x_y_z_transforms(transforms):
    """
    Extract x, y, z transforms from a dictionary of transforms
    """
    transform = list(transforms.values())
    x_transforms = [t[0, 3] for t in transform]
    y_transforms = [t[1, 3] for t in transform]
    z_transforms = [t[2, 3] for t in transform]

    return x_transforms, y_transforms, z_transforms


def analyze_outlier_pairs(pairs, tile_names, transforms, threshold=None):
    """
    Analyze pairs of tiles with unusually large transformation differences
    
    Args:
        pairs: List of tuples containing pairs of adjacent tile names
        tile_names: Dictionary mapping tile IDs to tile names
        transforms: Dictionary mapping tile IDs to transformation matrices
        threshold: Optional float to define outlier threshold. If None,
                  uses mean + 2*std
    
    Returns:
        dict: Information about outlier pairs including:
              - The tile pairs
              - Their locations
              - The magnitude of their differences
    """
    # Get basic stats first
    stats = analyze_adjacent_pairs(pairs, tile_names, transforms)
    
    # Analyze each dimension separately
    outliers = {
        'x': [],
        'y': [],
        'z': []
    }
    
    # If threshold not provided, use statistical threshold
    if threshold is None:
        thresholds = {
            'x': stats['x_translation']['mean'] + 1 * stats['x_translation']['std'],
            'y': stats['y_translation']['mean'] + 1 * stats['y_translation']['std'],
            'z': stats['z_translation']['mean'] + 1 * stats['z_translation']['std']
        }
    else:
        thresholds = {'x': threshold, 'y': threshold, 'z': threshold}
    print(thresholds)
    
    for i, (tile1, tile2) in enumerate(pairs):
        # Get indices and transforms
        idx1 = list(tile_names.keys())[list(tile_names.values()).index(tile1)]
        idx2 = list(tile_names.keys())[list(tile_names.values()).index(tile2)]
        t1 = transforms[idx1]
        t2 = transforms[idx2]
        
        # Calculate differences
        diffs = {
            'x': abs(t1[0, 3] - t2[0, 3]),
            'y': abs(t1[1, 3] - t2[1, 3]),
            'z': abs(t1[2, 3] - t2[2, 3])
        }
        
        # Get tile positions
        pos1 = parse_tile_name(tile1)
        pos2 = parse_tile_name(tile2)
        
        # Check each dimension for outliers
        for dim in ['x', 'y', 'z']:
            if diffs[dim] > thresholds[dim]:
                outliers[dim].append({
                    'tile1': {
                        'name': tile1,
                        'position': pos1,
                        'transform': t1
                    },
                    'tile2': {
                        'name': tile2,
                        'position': pos2,
                        'transform': t2
                    },
                    'difference': diffs[dim]
                })
    
    # Sort outliers by difference magnitude
    for dim in outliers:
        outliers[dim].sort(key=lambda x: x['difference'], reverse=True)
    
    return outliers


def plot_transform_differences_histogram(stats, bins=50):

    """
    Plot histograms of transform differences with additional analysis
    
    Args:
        stats: Statistics dictionary from analyze_adjacent_pairs
        bins: Number of bins for histogram
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot X differences
    ax1.hist(stats['x_translation']['x_diffs'], bins=bins)
    ax1.set_title('X Translation Differences')
    ax1.set_xlabel('Difference (pixels)')
    ax1.set_ylabel('Count')
    
    # Plot Y differences
    ax2.hist(stats['y_translation']['y_diffs'], bins=bins)
    ax2.set_title('Y Translation Differences')
    ax2.set_xlabel('Difference (pixels)')
    
    # Plot Z differences
    ax3.hist(stats['z_translation']['z_diffs'], bins=bins)
    ax3.set_title('Z Translation Differences')
    ax3.set_xlabel('Difference (pixels)')

    # # XLIMIT
    # ax1.set_xlim(0, 100)
    # ax2.set_xlim(0, 100)
    # ax3.set_xlim(0, 100)
    
    plt.tight_layout()
    plt.show()


def get_transformed_tile_pair(tile1_name, tile2_name, 
                              transforms, tile_names, 
                              bucket_name, dataset_path, 
                             z_slice=None, pyramid_level=0):
    """
    Get a z-slice through two adjacent tiles in their transformed positions.
    Transforms are relative to (0,0) at the center of the nominal grid.
    Z-transform is applied before selecting the slice.
    
    Args:
        tile1_name, tile2_name: Names of tiles to compare
        transforms: Dictionary mapping tile IDs to transformation matrices
        tile_names: Dictionary mapping tile IDs to tile names
        bucket_name: S3 bucket name
        dataset_path: Path to dataset in bucket
        z_slice: Z-slice to display
        pyramid_level: Pyramid level to load
        
    Returns:
        combined: Combined image array
        extent: [x_min, x_max, y_min, y_max] for plotting
        z_slice: The z-slice that was used
    """
    # Get indices and transforms
    idx1 = list(tile_names.keys())[list(tile_names.values()).index(tile1_name)]
    idx2 = list(tile_names.keys())[list(tile_names.values()).index(tile2_name)]
    t1 = transforms[idx1]
    t2 = transforms[idx2]
    
    # Load tile data
    tile1_data = load_tile_data(tile1_name, bucket_name, dataset_path, pyramid_level)
    tile2_data = load_tile_data(tile2_name, bucket_name, dataset_path, pyramid_level)
    
    # Apply Z transform to the data
    scale = int(2**pyramid_level)
    z_offset1 = int(round(t1[2, 3] / scale))  # Get Z offset from transform
    z_offset2 = int(round(t2[2, 3] / scale))  # Get Z offset from transform
    
    # Pad or crop the data based on z offsets
    max_z = max(tile1_data.shape[2] + abs(z_offset1), tile2_data.shape[2] + abs(z_offset2))
    min_z = min(0, z_offset1, z_offset2)
    
    # Create padded arrays (initialized to black)
    tile1_padded = np.zeros((tile1_data.shape[0], tile1_data.shape[1], max_z - min_z), dtype=tile1_data.dtype)
    tile2_padded = np.zeros((tile2_data.shape[0], tile2_data.shape[1], max_z - min_z), dtype=tile2_data.dtype)
    
    # Fill padded arrays (rest remains black)
    z1_start = abs(min_z) + z_offset1
    z2_start = abs(min_z) + z_offset2
    tile1_padded[:, :, z1_start:z1_start + tile1_data.shape[2]] = tile1_data
    tile2_padded[:, :, z2_start:z2_start + tile2_data.shape[2]] = tile2_data
    
    if z_slice is None:
        z_slice = (max_z - min_z) // 2
    elif z_slice =="max":
        # determine which z slice had the most signal together
        z_slice = np.argmax(np.sum(tile1_padded, axis=(0,1)) + np.sum(tile2_padded, axis=(0,1)))
    elif z_slice == "center":
        z_slice = (max_z - min_z) // 2
    
    # # Create RGB arrays for visualization
    # def create_rgb_slice(data, z_idx, color):
    #     """Create RGB array with data in specified color channel"""
    #     rgb = np.zeros((*data.shape[:2], 3))
    #     if color == 'red':
    #         rgb[..., 0] = data[:, :, z_idx] / np.percentile(data[:, :, z_idx], 99.99)
    #     elif color == 'green':
    #         rgb[..., 1] = data[:, :, z_idx] / np.percentile(data[:, :, z_idx], 99.99)
    #     return np.clip(rgb, 0, 1)


        # Create RGB arrays for visualization
    def create_rgb_slice(data, z_idx, color, clip_range=None):
        """Create RGB array with data in specified color channel"""
        rgb = np.zeros((*data.shape[:2], 3))

        # clip to percentile 1 and 99 of whole stack
        if clip_range is not None:
            min_val, max_val = clip_range
        else:
            min_val = np.percentile(data, 1)
            max_val = np.percentile(data, 99.9)
        data = np.clip(data, min_val, max_val)
        print(min_val, max_val)
        if color == 'red':
            slice_data = data[:, :, z_idx]
            rgb[..., 0] = slice_data / max_val
        elif color == 'green':
            slice_data = data[:, :, z_idx]
            rgb[..., 1] = slice_data / max_val
        return np.clip(rgb, 0, 1)
    
    # Create RGB arrays using padded data
    rgb1 = create_rgb_slice(tile1_padded, z_slice, 'red')
    rgb2 = create_rgb_slice(tile2_padded, z_slice, 'green')
    
    # Calculate transformed coordinates
    def get_transformed_coords(data_shape, transform):
        """Get transformed corner coordinates relative to center (0,0)"""
        h, w = data_shape[:2]
        # Define corners relative to center of tile
        corners = np.array([
            [-w/2, -h/2, 0, 1],  # top-left
            [w/2, -h/2, 0, 1],   # top-right
            [-w/2, h/2, 0, 1],   # bottom-left
            [w/2, h/2, 0, 1]     # bottom-right
        ])
        transformed = np.dot(transform, corners.T).T
        return transformed[:, [0, 1]]  # Return x,y coordinates
    
    # Scale transforms by pyramid level
    scale = 2**pyramid_level
    t1_scaled = t1.copy()
    t2_scaled = t2.copy()
    t1_scaled[:3, 3] /= scale
    t2_scaled[:3, 3] /= scale
    
    coords1 = get_transformed_coords(tile1_data.shape, t1_scaled)
    coords2 = get_transformed_coords(tile2_data.shape, t2_scaled)
    
    # Create a combined image that covers both tiles
    x_coords = np.concatenate([coords1[:, 0], coords2[:, 0]])
    y_coords = np.concatenate([coords1[:, 1], coords2[:, 1]])
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    
    # Calculate pixel dimensions needed for the combined image
    pixel_size = 1.0  # base pixel size
    width = int((x_max - x_min) / pixel_size)
    height = int((y_max - y_min) / pixel_size)
    
    # Create empty combined image
    combined = np.zeros((height, width, 3))
    
    # Map tile coordinates to combined image coordinates
    def map_to_combined(rgb, coords):
        src_h, src_w = rgb.shape[:2]
        x_start = int((coords[:, 0].min() - x_min) / pixel_size)
        y_start = int((coords[:, 1].min() - y_min) / pixel_size)
        x_end = x_start + src_w
        y_end = y_start + src_h
        return (slice(y_start, y_end), slice(x_start, x_end))
    
    # Add both tiles to the combined image
    slice1 = map_to_combined(rgb1, coords1)
    slice2 = map_to_combined(rgb2, coords2)
    combined[slice1] += rgb1
    combined[slice2] += rgb2
    
    # Clip to ensure we don't exceed 1.0
    combined = np.clip(combined, 0, 1)
    
    # Return the combined image and extent for plotting
    extent = [x_min, x_max, y_min, y_max]
    
    return combined, extent, z_slice

def plot_adjacent_tile_pair(tile1_name, tile2_name, transforms, tile_names, bucket_name, dataset_path, 
                          slice_index=None, pyramid_level=0, save=False, output_dir=None):
    """
    Plot a z-slice through two adjacent tiles in their transformed positions.
    Transforms are relative to (0,0) at the center of the nominal grid.
    Z-transform is applied before selecting the slice.
    
    Args:
        tile1_name, tile2_name: Names of tiles to compare
        transforms: Dictionary mapping tile IDs to transformation matrices
        tile_names: Dictionary mapping tile IDs to tile names
        bucket_name: S3 bucket name
        dataset_path: Path to dataset in bucket
        slice_index: slice index in the zarr file
        pyramid_level: Pyramid level to load
        save: Whether to save the plot
        output_dir: Directory to save plot if save=True
    """
    # Get the transformed and combined tile data
    combined, extent, slice = get_transformed_tile_pair(
        tile1_name, tile2_name, transforms, tile_names,
        bucket_name, dataset_path, slice_index, pyramid_level
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_facecolor('black')
    
    # Plot combined image
    ax.imshow(combined, extent=extent)
    
    # Add tile information
    pos1 = parse_tile_name(tile1_name)
    pos2 = parse_tile_name(tile2_name)
    ax.set_title(f'Tile Pair Comparison\nRed: {pos1} | Green: {pos2} | Yellow: Overlap\nZ-slice: {slice_index}')
    
    # Set axis limits with padding
    padding = 0.05  # 5% padding
    x_range = extent[1] - extent[0]
    y_range = extent[3] - extent[2]
    ax.set_xlim(extent[0] - x_range * padding, extent[1] + x_range * padding)
    ax.set_ylim(extent[2] - y_range * padding, extent[3] + y_range * padding)
    
    if save and output_dir:
        output_path = Path(output_dir) / f'tile_pair_{tile1_name}_{tile2_name}_z{slice_index}.png'
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()
        
    return fig, ax


def plot_adjacent_tile_pair(tile1_name, tile2_name, transforms, tile_names, bucket_name, dataset_path, 
                          slice_index=None, pyramid_level=0, save=False, output_dir=None):
    """
    Plot a z-slice through two adjacent tiles in their transformed positions.
    Transforms are relative to (0,0) at the center of the nominal grid.
    Z-transform is applied before selecting the slice.
    
    Args:
        tile1_name, tile2_name: Names of tiles to compare
        transforms: Dictionary mapping tile IDs to transformation matrices
        tile_names: Dictionary mapping tile IDs to tile names
        bucket_name: S3 bucket name
        dataset_path: Path to dataset in bucket
        slice_index: slice index in the zarr file
        pyramid_level: Pyramid level to load
        save: Whether to save the plot
        output_dir: Directory to save plot if save=True
    """
    # Get the transformed and combined tile data
    combined, extent, slice = get_transformed_tile_pair(
        tile1_name, tile2_name, transforms, tile_names,
        bucket_name, dataset_path, slice_index, pyramid_level
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_facecolor('black')
    
    # Plot combined image
    ax.imshow(combined, extent=extent)
    
    # Add tile information
    pos1 = parse_tile_name(tile1_name)
    pos2 = parse_tile_name(tile2_name)
    ax.set_title(f'Tile Pair Comparison\nRed: {pos1} | Green: {pos2} | Yellow: Overlap\nZ-slice: {slice_index}')
    
    # Set axis limits with padding
    padding = 0.05  # 5% padding
    x_range = extent[1] - extent[0]
    y_range = extent[3] - extent[2]
    ax.set_xlim(extent[0] - x_range * padding, extent[1] + x_range * padding)
    ax.set_ylim(extent[2] - y_range * padding, extent[3] + y_range * padding)
    
    if save and output_dir:
        output_path = Path(output_dir) / f'tile_pair_{tile1_name}_{tile2_name}_z{slice_index}.png'
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()
        
    return fig, ax



def plot_adjacent_tile_pair_zoom(tile1_name, tile2_name, transforms, tile_names, bucket_name, dataset_path, 
                          slice_index=None, pyramid_level=0, save=False, output_dir=None, zoom_padding=50):
    """
    Plot a z-slice through two adjacent tiles in their transformed positions.
    Transforms are relative to (0,0) at the center of the nominal grid.
    Z-transform is applied before selecting the slice.
    
    Args:
        tile1_name, tile2_name: Names of tiles to compare
        transforms: Dictionary mapping tile IDs to transformation matrices
        tile_names: Dictionary mapping tile IDs to tile names
        bucket_name: S3 bucket name
        dataset_path: Path to dataset in bucket
        slice_index: slice index in the zarr file, or str 'center' or 'max'
        pyramid_level: Pyramid level to load
        save: Whether to save the plot
        output_dir: Directory to save plot if save=True
        zoom_padding: Number of pixels to pad around the overlap region
    """
    # Get the transformed and combined tile data
    combined, extent, slice_ind = get_transformed_tile_pair(
        tile1_name, tile2_name, transforms, tile_names,
        bucket_name, dataset_path, slice_index, pyramid_level
    )

    # rotate the combined image 90 degrees if longer in y than x
    if combined.shape[0] > combined.shape[1]:
        combined = np.rot90(combined)
        extent = [extent[2], extent[3], extent[0], extent[1]]
    
    # Create figure with GridSpec for main view, zoom, and top regions
    
    fig = plt.figure(figsize=(20, 8))
    gs = gridspec.GridSpec(1, 4, width_ratios=[3, 1, 1, 1], wspace=0.05, hspace=0.05)
    
    # Main view
    ax_main = plt.subplot(gs[0])
    ax_main.set_facecolor('black')
    
    # Plot combined image
    ax_main.imshow(combined, extent=extent)
    ax_main.set_aspect('equal')
    
    # Add tile information
    pos1, ch = parse_tile_name(tile1_name)
    pos2, ch = parse_tile_name(tile2_name)
    ax_main.set_title(f'Tile Pair Comparison\nRed: {pos1} | Green: {pos2} | Ch: {ch}\nZ-slice: {slice_index} - {slice_ind}')
    
    # Set axis limits with padding
    padding = 0.05  # 5% padding
    x_range = extent[1] - extent[0]
    y_range = extent[3] - extent[2]
    ax_main.set_xlim(extent[0] - x_range * padding, extent[1] + x_range * padding)
    ax_main.set_ylim(extent[2] - y_range * padding, extent[3] + y_range * padding)
    
    # Find overlap region (where both red and green channels are present)
    overlap_mask = np.logical_and(
        combined[:,:,0] > 0.05,  # Red channel threshold
        combined[:,:,1] > 0.05   # Green channel threshold
    )
    
    # Make a brighter version of the combined image for zoomed views
    brightness_factor = 1.5
    brightened = np.clip(combined * brightness_factor, 0, 1)
    
    if np.any(overlap_mask):
        # Get the bounds of the overlap region
        y_indices, x_indices = np.where(overlap_mask)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        
        # Add padding to the zoom region
        y_min = max(0, y_min - zoom_padding)
        y_max = min(combined.shape[0] - 1, y_max + zoom_padding)
        x_min = max(0, x_min - zoom_padding)
        x_max = min(combined.shape[1] - 1, x_max + zoom_padding)
        
        # Calculate the pixel to coordinate mapping
        pixel_width = (extent[1] - extent[0]) / combined.shape[1]
        pixel_height = (extent[3] - extent[2]) / combined.shape[0]
        
        # Calculate the extent of the zoom region
        zoom_extent = [
            extent[0] + x_min * pixel_width,
            extent[0] + x_max * pixel_width,
            extent[2] + y_min * pixel_height,
            extent[2] + y_max * pixel_height
        ]
        print(zoom_extent)
        
        # Create zoom view
        ax_zoom = plt.subplot(gs[1])
        ax_zoom.set_facecolor('black')
        
        # Plot zoomed region
        ax_zoom.imshow(brightened, extent=extent)
        ax_zoom.set_xlim(zoom_extent[0], zoom_extent[1])
        ax_zoom.set_ylim(zoom_extent[2], zoom_extent[3])
        # y off
        ax_zoom.set_yticks([])
        
        ax_zoom.set_title('Overlap')
        ax_zoom.set_aspect('auto')
        
        # get crop of overlap region for top and bottom
        overlap_region = brightened[y_min:y_max, x_min:x_max]
        overlap_mask_cropped = overlap_mask[y_min:y_max, x_min:x_max]

        # get the top and bottom halves of the overlap region
        h_mid = (y_max - y_min) // 2
        top_region = overlap_region[:h_mid, :]
        bottom_region = overlap_region[h_mid:, :]


        overlap_start = np.where(overlap_mask_cropped)[1].min()
        overlap_end = np.where(overlap_mask_cropped)[1].max()
        overlap_start = max(0, overlap_start - 0.8 * (overlap_end - overlap_start))
        overlap_end = min(bottom_region.shape[1], overlap_end + 0.8 * (overlap_end - overlap_start))

        # create a subplot for the top half
        ax_top = plt.subplot(gs[2])
        ax_top.set_facecolor('black')
        ax_top.imshow(top_region)
        ax_top.set_xlim(overlap_start, overlap_end)
        ax_top.set_aspect('auto')
        ax_top.set_title('Overlap Top')
        # y off
        ax_top.set_yticks([])

        # create a subplot for the bottom half
        ax_bottom = plt.subplot(gs[3])
        ax_bottom.set_facecolor('black')
        ax_bottom.imshow(bottom_region)
        ax_bottom.set_xlim(overlap_start, overlap_end)
        ax_bottom.set_aspect('auto')
        ax_bottom.set_title('Overlap Bottom')
        # y off
        ax_bottom.set_yticks([])

    plt.tight_layout()
    
    if save and output_dir:
        output_path = Path(output_dir) / f'tile_pair_{pos1}_{pos2}_ch{ch}_z{slice_index}_pyr{pyramid_level}.png'
        
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()
        
    return fig, ax_main


def plot_adjacent_tile_pair_zoom2(tile1_name, tile2_name, transforms, tile_names, bucket_name, dataset_path, 
                          slice_index=None, pyramid_level=0, save=False, output_dir=None, zoom_padding=50):
    """
    Plot a z-slice through two adjacent tiles in their transformed positions.
    Transforms are relative to (0,0) at the center of the nominal grid.
    Z-transform is applied before selecting the slice.
    
    Args:
        tile1_name, tile2_name: Names of tiles to compare
        transforms: Dictionary mapping tile IDs to transformation matrices
        tile_names: Dictionary mapping tile IDs to tile names
        bucket_name: S3 bucket name
        dataset_path: Path to dataset in bucket
        slice_index: slice index in the zarr file, or str 'center' or 'max'
        pyramid_level: Pyramid level to load
        save: Whether to save the plot
        output_dir: Directory to save plot if save=True
        zoom_padding: Number of pixels to pad around the overlap region
    """
    sns.set_context("talk")
    # Get the transformed and combined tile data
    combined, extent, slice_ind = get_transformed_tile_pair(
        tile1_name, tile2_name, transforms, tile_names,
        bucket_name, dataset_path, slice_index, pyramid_level
    )

    # rotate the combined image 90 degrees if longer in y than x
    if combined.shape[0] > combined.shape[1]:
        combined = np.rot90(combined)
        extent = [extent[2], extent[3], extent[0], extent[1]]
    
    # Create figure with GridSpec for main view on top, zoom and sections below
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 2.5], width_ratios=[1, 1, 1])
    
    # Main view (spans all columns in top row)
    ax_main = plt.subplot(gs[0, :])
    ax_main.set_facecolor('black')
    
    # Plot combined image
    ax_main.imshow(combined, extent=extent)
    ax_main.set_aspect('equal')
    
    # Add tile information
    pos1, ch = parse_tile_name(tile1_name)
    pos2, ch = parse_tile_name(tile2_name)
    ax_main.set_title(f'Tile Pair Comparison\nRed: {pos1} | Green: {pos2} | Ch: {ch}\nZ-slice: {slice_index} - {slice_ind}')
    
    # Set axis limits with padding
    padding = 0.05  # 5% padding
    x_range = extent[1] - extent[0]
    y_range = extent[3] - extent[2]
    ax_main.set_xlim(extent[0] - x_range * padding, extent[1] + x_range * padding)
    ax_main.set_ylim(extent[2] - y_range * padding, extent[3] + y_range * padding)
    
    # Find overlap region (where both red and green channels are present)
    overlap_mask = np.logical_and(
        combined[:,:,0] > 0.05,  # Red channel threshold
        combined[:,:,1] > 0.05   # Green channel threshold
    )
    
    # Make a brighter version of the combined image for zoomed views
    brightness_factor = 1.5
    brightened = np.clip(combined * brightness_factor, 0, 1)
    
    if np.any(overlap_mask):
        # Get the bounds of the overlap region
        y_indices, x_indices = np.where(overlap_mask)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        
        # Add padding to the zoom region
        y_min = max(0, y_min - zoom_padding)
        y_max = min(combined.shape[0] - 1, y_max + zoom_padding)
        x_min = max(0, x_min - zoom_padding)
        x_max = min(combined.shape[1] - 1, x_max + zoom_padding)
        
        # Calculate the pixel to coordinate mapping
        pixel_width = (extent[1] - extent[0]) / combined.shape[1]
        pixel_height = (extent[3] - extent[2]) / combined.shape[0]
        
        # Calculate the extent of the zoom region
        zoom_extent = [
            extent[0] + x_min * pixel_width,
            extent[0] + x_max * pixel_width,
            extent[2] + y_min * pixel_height,
            extent[2] + y_max * pixel_height
        ]
        print(zoom_extent)
        
        # Create zoom view in bottom left
        ax_zoom = plt.subplot(gs[1, 0])
        ax_zoom.set_facecolor('black')
        
        # Plot zoomed region
        ax_zoom.imshow(brightened, extent=extent)
        ax_zoom.set_xlim(zoom_extent[0], zoom_extent[1])
        ax_zoom.set_ylim(zoom_extent[2], zoom_extent[3])
        # y off
        ax_zoom.set_yticks([])
        
        ax_zoom.set_title('Overlap')
        ax_zoom.set_aspect('auto')
        
        # Draw a rectangle on the main plot showing the zoom region
        rect = plt.Rectangle(
            (zoom_extent[0], zoom_extent[2]),
            zoom_extent[1] - zoom_extent[0],
            zoom_extent[3] - zoom_extent[2],
            linewidth=1, edgecolor='white', facecolor='none'
        )
        ax_main.add_patch(rect)
        
        # get crop of overlap region for top and bottom
        overlap_region = brightened[y_min:y_max, x_min:x_max]
        overlap_mask_cropped = overlap_mask[y_min:y_max, x_min:x_max]

        # get the top and bottom halves of the overlap region
        h_mid = (y_max - y_min) // 2
        top_region = overlap_region[:h_mid, :]
        bottom_region = overlap_region[h_mid:, :]

        overlap_start = np.where(overlap_mask_cropped)[1].min()
        overlap_end = np.where(overlap_mask_cropped)[1].max()
        overlap_start = max(0, overlap_start - 0.8 * (overlap_end - overlap_start))
        overlap_end = min(bottom_region.shape[1], overlap_end + 0.8 * (overlap_end - overlap_start))

        # create a subplot for the top half in bottom middle
        ax_top = plt.subplot(gs[1, 1])
        ax_top.set_facecolor('black')
        ax_top.imshow(top_region)
        ax_top.set_xlim(overlap_start, overlap_end)
        ax_top.set_aspect('auto')
        ax_top.set_title('Overlap Top')
        # y off
        ax_top.set_yticks([])

        # create a subplot for the bottom half in bottom right
        ax_bottom = plt.subplot(gs[1, 2])
        ax_bottom.set_facecolor('black')
        ax_bottom.imshow(bottom_region)
        ax_bottom.set_xlim(overlap_start, overlap_end)
        ax_bottom.set_aspect('auto')
        ax_bottom.set_title('Overlap Bottom')
        # y off
        ax_bottom.set_yticks([])

    plt.tight_layout()
    
    if save and output_dir:
        output_path = Path(output_dir) / f'tile_pair_{pos1}_{pos2}_ch{ch}_z{slice_index}_pyr{pyramid_level}.png'
        
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()
        
    return fig, ax_main


def plot_adjacent_tile_pair_zoom3(tile1_name, tile2_name, transforms, tile_names, bucket_name, dataset_path, 
                          slice_index=None, pyramid_level=0, save=False, output_dir=None, zoom_padding=50):
    """
    Plot a z-slice through two adjacent tiles in their transformed positions.
    Transforms are relative to (0,0) at the center of the nominal grid.
    Z-transform is applied before selecting the slice.
    
    Args:
        tile1_name, tile2_name: Names of tiles to compare
        transforms: Dictionary mapping tile IDs to transformation matrices
        tile_names: Dictionary mapping tile IDs to tile names
        bucket_name: S3 bucket name
        dataset_path: Path to dataset in bucket
        slice_index: slice index in the zarr file
        pyramid_level: Pyramid level to load
        save: Whether to save the plot
        output_dir: Directory to save plot if save=True
        zoom_padding: Number of pixels to pad around the overlap region
    """
    # Get the transformed and combined tile data
    combined, extent, slice = get_transformed_tile_pair(
        tile1_name, tile2_name, transforms, tile_names,
        bucket_name, dataset_path, slice_index, pyramid_level
    )
    
    # Create figure with GridSpec for main view, zoom, and vertical sections
    fig = plt.figure(figsize=(18, 8))
    gs = gridspec.GridSpec(1, 3, width_ratios=[3, 1, 1])
    
    # Main view
    ax_main = plt.subplot(gs[0])
    ax_main.set_facecolor('black')
    
    # Plot combined image
    ax_main.imshow(combined, extent=extent)
    
    # Add tile information
    pos1 = parse_tile_name(tile1_name)
    pos2 = parse_tile_name(tile2_name)
    ax_main.set_title(f'Tile Pair Comparison\nRed: {pos1} | Green: {pos2} | Yellow: Overlap\nZ-slice: {slice_index}')
    
    # Set axis limits with padding
    padding = 0.05  # 5% padding
    x_range = extent[1] - extent[0]
    y_range = extent[3] - extent[2]
    ax_main.set_xlim(extent[0] - x_range * padding, extent[1] + x_range * padding)
    ax_main.set_ylim(extent[2] - y_range * padding, extent[3] + y_range * padding)
    
    # Find overlap region (where both red and green channels are present)
    overlap_mask = np.logical_and(
        combined[:,:,0] > 0.05,  # Red channel threshold
        combined[:,:,1] > 0.05   # Green channel threshold
    )
    
    # Make a brighter version of the combined image for zoomed views
    brightness_factor = 1.5
    brightened = np.clip(combined * brightness_factor, 0, 1)
    
    if np.any(overlap_mask):
        # Get the bounds of the overlap region
        y_indices, x_indices = np.where(overlap_mask)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        
        # Add padding to the zoom region
        y_min = max(0, y_min - zoom_padding)
        y_max = min(combined.shape[0] - 1, y_max + zoom_padding)
        x_min = max(0, x_min - zoom_padding)
        x_max = min(combined.shape[1] - 1, x_max + zoom_padding)
        
        # Calculate the pixel to coordinate mapping
        pixel_width = (extent[1] - extent[0]) / combined.shape[1]
        pixel_height = (extent[3] - extent[2]) / combined.shape[0]
        
        # Calculate the extent of the zoom region
        zoom_extent = [
            extent[0] + x_min * pixel_width,
            extent[0] + x_max * pixel_width,
            extent[2] + y_min * pixel_height,
            extent[2] + y_max * pixel_height
        ]
        
        # Create zoom view
        ax_zoom = plt.subplot(gs[1])
        ax_zoom.set_facecolor('black')
        
        # Plot zoomed region
        ax_zoom.imshow(brightened, extent=extent)
        ax_zoom.set_xlim(zoom_extent[0], zoom_extent[1])
        ax_zoom.set_ylim(zoom_extent[2], zoom_extent[3])
        ax_zoom.set_title('Overlap Region (Zoomed)')
        
        # Add a small grid to the zoom view
        ax_zoom.grid(True, color='white', alpha=0.2, linestyle=':')
        
        # Create a nested GridSpec for the 2x2 vertical sections
        gs_sections = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[2])
        
        # Calculate the height of each vertical section
        section_height = (y_max - y_min) // 4
        
        # Create 4 vertical sections
        for i in range(4):
            row = i // 2
            col = i % 2
            
            # Calculate the vertical bounds for this section
            section_y_min = y_min + i * section_height
            section_y_max = section_y_min + section_height
            
            # Ensure the last section goes to the end
            if i == 3:
                section_y_max = y_max
                
            # Calculate the extent of this section
            section_extent = [
                zoom_extent[0],
                zoom_extent[1],
                extent[2] + section_y_min * pixel_height,
                extent[2] + section_y_max * pixel_height
            ]
            
            # Create subplot
            ax_section = plt.subplot(gs_sections[row, col])
            ax_section.set_facecolor('black')
            
            # Plot the section
            ax_section.imshow(brightened, extent=extent)
            ax_section.set_xlim(section_extent[0], section_extent[1])
            ax_section.set_ylim(section_extent[2], section_extent[3])
            
            # Turn off axis
            ax_section.set_xticks([])
            ax_section.set_yticks([])
            
            # Draw a rectangle on the zoom view showing this section
            rect = plt.Rectangle(
                (section_extent[0], section_extent[2]),
                section_extent[1] - section_extent[0],
                section_extent[3] - section_extent[2],
                linewidth=1, edgecolor=['red', 'green', 'blue', 'cyan'][i], facecolor='none'
            )
            ax_zoom.add_patch(rect)
            
            # Add a small colored marker in the corner of each section to match the rectangle
            ax_section.plot(section_extent[0] + 5*pixel_width, section_extent[2] + 5*pixel_height, 
                          'o', color=['red', 'green', 'blue', 'cyan'][i], markersize=8)
        
        # Add a title to the sections subplot area
        plt.figtext(0.83, 0.95, 'Vertical Sections', ha='center', va='center', fontsize=12)
        
    else:
        # If no overlap is found, just display a message
        for ax_idx in [1, 2]:
            ax = plt.subplot(gs[ax_idx])
            ax.set_facecolor('black')
            ax.text(0.5, 0.5, 'No overlap detected', 
                    color='white', ha='center', va='center')
            ax.set_xticks([])
            ax.set_yticks([])
            if ax_idx == 1:
                ax.set_title('Overlap Region')
            else:
                ax.set_title('Vertical Sections')
    
    plt.tight_layout()
    
    if save and output_dir:
        output_path = Path(output_dir) / f'tile_pair_{tile1_name}_{tile2_name}_z{slice_index}.png'
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()
        
    return fig, ax_main


def plot_adjacent_tile_pair_t(tile1_name, tile2_name, 
                              transforms, tile_names, bucket_name, dataset_path, 
                          slice=None, pyramid_level=0, save=False, output_dir=None):
    """
    Plot a z-slice through two adjacent tiles in their transformed positions.
    Transforms are relative to (0,0) at the center of the nominal grid.
    Z-transform is applied before selecting the slice.
    
    Args:
        tile1_name, tile2_name: Names of tiles to compare
        transforms: Dictionary mapping tile IDs to transformation matrices
        tile_names: Dictionary mapping tile IDs to tile names
        bucket_name: S3 bucket name
        dataset_path: Path to dataset in bucket
        slice: slice in 
        pyramid_level: Pyramid level to load
        save: Whether to save the plot
        output_dir: Directory to save plot if save=True
    """
    # Get the transformed and combined tile data
    combined, extent, slice = get_transformed_tile_pair(
        tile1_name, tile2_name, transforms, tile_names,
        bucket_name, dataset_path, slice, pyramid_level
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_facecolor('black')
    
    # Plot combined image
    ax.imshow(combined, extent=extent)
    
    # Add tile information
    pos1 = parse_tile_name(tile1_name)
    pos2 = parse_tile_name(tile2_name)
    ax.set_title(f'Tile Pair Comparison\nRed: {pos1} | Green: {pos2} | Yellow: Overlap\nZ-slice: {z_slice}')
    
    # Set axis limits with padding
    padding = 0.05  # 5% padding
    x_range = extent[1] - extent[0]
    y_range = extent[3] - extent[2]
    ax.set_xlim(extent[0] - x_range * padding, extent[1] + x_range * padding)
    ax.set_ylim(extent[2] - y_range * padding, extent[3] + y_range * padding)
    
    if save and output_dir:
        output_path = Path(output_dir) / f'tile_pair_{tile1_name}_{tile2_name}_z{z_slice}.png'
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()
        
    return fig, ax

def plot_all_adjacent_pairs(pairs, transforms, tile_names, bucket_name, dataset_path,
                           z_slice=None, pyramid_level=0, save=False, output_dir=None,
                           max_pairs=None):
    """
    Plot all adjacent tile pairs
    
    Args:
        pairs: List of (tile1, tile2) tuples from get_all_adjacent_pairs()
        transforms: Dictionary mapping tile IDs to transformation matrices
        tile_names: Dictionary mapping tile IDs to tile names
        bucket_name: S3 bucket name
        dataset_path: Path to dataset in bucket
        z_slice: Optional z-slice index. If None, uses middle slice
        pyramid_level: Pyramid level to load (default 0 = full resolution)
        save: If True, saves figures instead of displaying
        output_dir: Directory to save figures if save=True
        max_pairs: Maximum number of pairs to plot (None for all)
    """
    if max_pairs is not None:
        pairs = pairs[:max_pairs]
    
    for tile1, tile2 in pairs:
        plot_adjacent_tile_pair(
            tile1, tile2, transforms, tile_names,
            bucket_name, dataset_path,
            z_slice=z_slice,
            pyramid_level=pyramid_level,
            save=save,
            output_dir=output_dir
        )



def plot_tile_comparison(tile1_name, tile2_name, bucket_name, dataset_path, 
                        tile_dict, pyramid_level=0, z_slice=100,
                        minmax_percentile=(1,99)):
    """
    Plot two tiles side by side with their position in the grid.
    
    Args:
        tile1_name: name or identifier for first tile
        tile2_name: name or identifier for second tile
        bucket_name: S3 bucket name
        dataset_path: Path to dataset in bucket
        tile_dict: Dictionary of all tiles {tile_id: tile_name}
        pyramid_level: pyramid level to load (default 0 = full resolution)
        z_slice: z-slice index, or indices for projection
        minmax_percentile: tuple of (min, max) percentiles for contrast adjustment
    """
    tile1_data = load_tile_data(tile1_name, bucket_name, dataset_path, pyramid_level)
    tile2_data = load_tile_data(tile2_name, bucket_name, dataset_path, pyramid_level)

    print(tile1_data.shape)
    print(tile2_data.shape)

    # Create a figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Plot first two subplots (individual tiles) as before

    if isinstance(z_slice, tuple):
        tile1_img = np.max(tile1_data[:,:,z_slice], axis=2)
        tile2_img = np.max(tile2_data[:,:,z_slice], axis=2)
    else:
        tile1_img = tile1_data[:,:,z_slice]
        tile2_img = tile2_data[:,:,z_slice]

    # Plot first tile (red)
    vmin = np.percentile(tile1_img, minmax_percentile[0])
    vmax = np.percentile(tile1_img, minmax_percentile[1])
    ax1.imshow(tile1_img, cmap='Reds', vmin=vmin, vmax=vmax)
    ax1.set_title(f'Tile 1 (Red)\n{tile1_name}\nminmax:({vmin}, {vmax})')
    #ax1.grid(True)
    
    # Plot second tile (green)
    vmin = np.percentile(tile2_img, minmax_percentile[0])
    vmax = np.percentile(tile2_img, minmax_percentile[1])
    ax2.imshow(tile2_img, cmap='Greens', vmin=vmin, vmax=vmax)
    ax2.set_title(f'Tile 2 (Green)\n{tile2_name}\nminmax:({vmin}, {vmax})')
    #ax2.grid(True)

    # Create coverage map for third subplot
    coords = []
    for _, tile in tile_dict.items():
        base_name = tile.split('_ch_')[0]
        parts = base_name.split('_')
        x = int(parts[2])
        y = int(parts[4])
        coords.append((x, y))
    
    # Find dimensions
    x_coords = {x for x, _ in coords}
    y_coords = {y for _, y in coords}
    
    x_dim = max(x_coords) + 1
    y_dim = max(y_coords) + 1
    
    # Create coverage map
    coverage = np.zeros((y_dim, x_dim))
    for x, y in coords:
        coverage[y, x] = 1

    # Get coordinates of our two tiles
    pos1 = parse_tile_name(tile1_name)
    pos2 = parse_tile_name(tile2_name)
    
    # Highlight the two tiles we're comparing
    coverage[pos1[1], pos1[0]] = 2  # Mark first tile
    coverage[pos2[1], pos2[0]] = 3  # Mark second tile

    # Plot the coverage map
    im = ax3.imshow(coverage, cmap='RdYlBu', interpolation='nearest')
    ax3.grid(True, which='major', color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    ax3.set_title('Tile Grid Position')
    
    # Add coordinate labels
    for i in range(x_dim):
        for j in range(y_dim):
            color = 'white' if coverage[j, i] == 0 else 'black'
            ax3.text(i, j, f'({i},{j})', ha='center', va='center', color=color)

    plt.colorbar(im, ax=ax3, label='Tile Present')
    # add a bit of space for subtitle
    plt.subplots_adjust(top=0.85)
    plt.suptitle(f'Tile Pair Comparison Z-slice: {z_slice}') 
    plt.tight_layout()
    plt.show()

def plot_tiles_on_grid(tile1_name, tile2_name, transforms, tile_names, bucket_name, dataset_path,
                      n_x_tiles, n_y_tiles, z_slice=None, pyramid_level=0,
                      color1='red', color2='green'):
    """
    Plot two tiles transformed onto their position in the full nominal grid.
    
    Args:
        tile1_name, tile2_name: Names of tiles to compare
        transforms: Dictionary mapping tile IDs to transformation matrices
        tile_names: Dictionary mapping tile IDs to tile names
        bucket_name: S3 bucket name
        dataset_path: Path to dataset in bucket
        n_x_tiles, n_y_tiles: Number of tiles in x and y dimensions
        z_slice: Z-slice to display (default: middle slice)
        pyramid_level: Pyramid level to load (default: 0 = full resolution)
        color1, color2: Colors for the two tiles (default: red and green)
    """
    # Load tile data
    tile1_data = load_tile_data(tile1_name, bucket_name, dataset_path, pyramid_level)
    tile2_data = load_tile_data(tile2_name, bucket_name, dataset_path, pyramid_level)
    
    if z_slice is None:
        z_slice = tile1_data.shape[2] // 2
        
    # Get tile dimensions
    tile_height, tile_width = tile1_data.shape[:2]
    
    # Create full canvas
    full_height = tile_height * n_y_tiles
    full_width = tile_width * n_x_tiles
    canvas = np.full((full_height, full_width, 3), np.nan)
    
    # Get transforms
    idx1 = list(tile_names.keys())[list(tile_names.values()).index(tile1_name)]
    idx2 = list(tile_names.keys())[list(tile_names.values()).index(tile2_name)]
    t1 = transforms[idx1]
    t2 = transforms[idx2]
    
    # Scale transforms by pyramid level
    scale = 2**pyramid_level
    t1_scaled = t1.copy()
    t2_scaled = t2.copy()
    t1_scaled[:3, 3] /= scale
    t2_scaled[:3, 3] /= scale
    
    def transform_coordinates(y, x, transform):
        """Transform pixel coordinates"""
        coords = np.array([0, y, x, 1])
        transformed = np.dot(transform, coords)
        return transformed[2], transformed[1]  # return x, y
    
    # Create color maps for each tile
    def create_color_array(data, color):
        """Create RGB array with data in specified color channel"""
        rgb = np.zeros((*data.shape[:2], 3))
        if color == 'red':
            rgb[..., 0] = data[:, :, z_slice] / np.percentile(data[:, :, z_slice], 99.99)
        elif color == 'green':
            rgb[..., 1] = data[:, :, z_slice] / np.percentile(data[:, :, z_slice], 99.99)
        elif color == 'blue':
            rgb[..., 2] = data[:, :, z_slice] / np.percentile(data[:, :, z_slice], 99.99)
        return np.clip(rgb, 0, 1)
    
    # Create meshgrid for pixel coordinates
    y, x = np.mgrid[0:tile_height, 0:tile_width]
    
    # Transform and place first tile
    rgb1 = create_color_array(tile1_data, color1)
    for i in range(tile_height):
        for j in range(tile_width):
            y_trans, x_trans = transform_coordinates(i, j, t1_scaled)
            y_idx, x_idx = int(y_trans), int(x_trans)
            if 0 <= y_idx < full_height and 0 <= x_idx < full_width:
                canvas[y_idx, x_idx] = rgb1[i, j]
    
    # Transform and place second tile with blending
    rgb2 = create_color_array(tile2_data, color2)
    for i in range(tile_height):
        for j in range(tile_width):
            y_trans, x_trans = transform_coordinates(i, j, t2_scaled)
            y_idx, x_idx = int(y_trans), int(x_trans)
            if 0 <= y_idx < full_height and 0 <= x_idx < full_width:
                if np.all(np.isnan(canvas[y_idx, x_idx])):
                    canvas[y_idx, x_idx] = rgb2[i, j]
                else:
                    # Blend colors in overlap regions
                    canvas[y_idx, x_idx] = np.nanmax([canvas[y_idx, x_idx], rgb2[i, j]], axis=0)
    
    # Plot the result
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.imshow(canvas)
    
    # Add tile information
    pos1 = parse_tile_name(tile1_name)
    pos2 = parse_tile_name(tile2_name)
    ax.set_title(f'Tile Pair on Nominal Grid\n{color1}: {pos1} | {color2}: {pos2}\nZ-slice: {z_slice}')
    
    # Add grid lines at tile boundaries
    for i in range(n_x_tiles + 1):
        ax.axvline(x=i * tile_width, color='white', alpha=0.3)
    for i in range(n_y_tiles + 1):
        ax.axhline(y=i * tile_height, color='white', alpha=0.3)
    
    plt.show()
    return fig, ax



def plot_tile_orthogonal_views(tile_data, tile_name, 
                             center_points=None, save=False, output_dir=None):
    """
    Plot orthogonal views (XY, XZ, YZ) through a tile at specified center points.
    
    Args:
        tile_data: 3D numpy array containing the tile data
        tile_name: Name of tile (for display purposes)
        center_points: Dict with 'x', 'y', 'z' keys specifying slice centers. If None, uses middle of volume
        save: Whether to save the plot
        output_dir: Directory to save plot if save=True
    """
    # Get tile dimensions
    h, w, d = tile_data.shape
    
    # If no center points provided, use middle of volume
    if center_points is None:
        center_points = {
            'x': w // 2,
            'y': h // 2,
            'z': d // 2
        }
    
    # Create figure with three subplots
    fig, (ax_xy, ax_xz, ax_yz) = plt.subplots(1, 3, figsize=(18, 6))
    
    # XY view (top down)
    xy_slice = tile_data[:, :, center_points['z']]
    ax_xy.imshow(xy_slice)
    ax_xy.set_title(f'XY Slice (Z={center_points["z"]})')
    ax_xy.set_xlabel('X')
    ax_xy.set_ylabel('Y')
    
    # XZ view (side view)
    xz_slice = tile_data[:, center_points['y'], :].T
    ax_xz.imshow(xz_slice)
    ax_xz.set_title(f'XZ Slice (Y={center_points["y"]})')
    ax_xz.set_xlabel('X')
    ax_xz.set_ylabel('Z')
    
    # YZ view (front view)
    yz_slice = tile_data[center_points['x'], :, :].T
    ax_yz.imshow(yz_slice)
    ax_yz.set_title(f'YZ Slice (X={center_points["x"]})')
    ax_yz.set_xlabel('Y')
    ax_yz.set_ylabel('Z')
    
    # Add tile information to overall figure
    pos = parse_tile_name(tile_name)
    fig.suptitle(f'Orthogonal Views of Tile {pos}')
    
    # Adjust layout
    plt.tight_layout()
    
    if save and output_dir:
        output_path = Path(output_dir) / f'orthogonal_views_{tile_name}.png'
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()
        
    return fig, (ax_xy, ax_xz, ax_yz)

def calculate_accutance(image_slice, percentile_threshold=99):
    """
    Calculate accutance (edge sharpness) for an image slice.
    Uses Sobel operator to detect edges and measures their strength.
    
    Args:
        image_slice: 2D numpy array containing the image slice
        percentile_threshold: Percentile threshold for edge detection (default 99)
        
    Returns:
        dict containing:
            mean_accutance: Mean edge strength
            max_accutance: Maximum edge strength
            accutance_map: 2D array of edge strengths
            edge_mask: Boolean mask of detected edges
    """
    
    
    # Normalize image to [0,1]
    img_norm = image_slice.astype(float)
    if img_norm.max() > 0:
        img_norm = img_norm / img_norm.max()
    
    # Calculate gradients using Sobel operator
    grad_x = ndimage.sobel(img_norm, axis=1)
    grad_y = ndimage.sobel(img_norm, axis=0)
    
    # Calculate gradient magnitude
    accutance_map = np.sqrt(grad_x**2 + grad_y**2)
    
    # Create edge mask using threshold
    edge_threshold = np.percentile(accutance_map, percentile_threshold)
    edge_mask = accutance_map > edge_threshold
    
    # Calculate statistics for detected edges
    edge_values = accutance_map[edge_mask]
    mean_accutance = edge_values.mean() if edge_values.size > 0 else 0
    max_accutance = edge_values.max() if edge_values.size > 0 else 0
    
    return {
        'mean_accutance': mean_accutance,
        'max_accutance': max_accutance,
        'accutance_map': accutance_map,
        'edge_mask': edge_mask
    }

def plot_accutance_profile(tile_data, axis='z', slice_range=None, step=1):
    """
    Plot accutance profile along specified axis.
    
    Args:
        tile_data: 3D numpy array (z,y,x)
        axis: Axis along which to calculate profile ('z', 'y', or 'x')
        slice_range: Tuple of (start, end) indices. If None, uses full range
        step: Step size for sampling slices (default 1)
        
    Returns:
        fig: Figure object
        ax: Axes object
        profile: Dict containing accutance values and positions
    """
    # Set up axis mapping
    axis_map = {'z': 0, 'y': 1, 'x': 2}
    axis_idx = axis_map[axis]
    
    # Determine slice range
    if slice_range is None:
        slice_range = (0, tile_data.shape[axis_idx])
    
    # Initialize arrays for profile
    positions = range(slice_range[0], slice_range[1], step)
    mean_accutance = []
    max_accutance = []
    
    # Calculate accutance for each slice
    for pos in positions:
        # Take slice along specified axis
        if axis == 'z':
            slice_data = tile_data[pos, :, :]
        elif axis == 'y':
            slice_data = tile_data[:, pos, :]
        else:  # x
            slice_data = tile_data[:, :, pos]
            
        # Calculate accutance
        acc = calculate_accutance(slice_data)
        mean_accutance.append(acc['mean_accutance'])
        max_accutance.append(acc['max_accutance'])
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(positions, mean_accutance, label='Mean Accutance', color='blue')
    ax.plot(positions, max_accutance, label='Max Accutance', color='red', alpha=0.5)
    
    ax.set_xlabel(f'{axis.upper()} Position')
    ax.set_ylabel('Accutance')
    ax.set_title(f'Accutance Profile Along {axis.upper()} Axis')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    profile = {
        'positions': positions,
        'mean_accutance': mean_accutance,
        'max_accutance': max_accutance
    }
    
    return fig, ax, profile

def plot_accutance_comparison(tile1_data, tile2_data, axis='z', slice_range=None, step=1):
    """
    Plot accutance profiles for two tiles side by side.
    
    Args:
        tile1_data, tile2_data: 3D numpy arrays (z,y,x)
        axis: Axis along which to calculate profile ('z', 'y', or 'x')
        slice_range: Tuple of (start, end) indices. If None, uses full range
        step: Step size for sampling slices (default 1)
        
    Returns:
        fig: Figure object
        axes: List of axes objects
        profiles: Dict containing accutance profiles for both tiles
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Calculate and plot profiles
    _, _, profile1 = plot_accutance_profile(tile1_data, axis, slice_range, step)
    _, _, profile2 = plot_accutance_profile(tile2_data, axis, slice_range, step)
    
    # Plot on first axis
    ax1.plot(profile1['positions'], profile1['mean_accutance'], 
             label='Tile 1 Mean', color='blue')
    ax1.plot(profile1['positions'], profile1['max_accutance'], 
             label='Tile 1 Max', color='blue', alpha=0.5)
    ax1.plot(profile2['positions'], profile2['mean_accutance'], 
             label='Tile 2 Mean', color='red')
    ax1.plot(profile2['positions'], profile2['max_accutance'], 
             label='Tile 2 Max', color='red', alpha=0.5)
    
    ax1.set_xlabel(f'{axis.upper()} Position')
    ax1.set_ylabel('Accutance')
    ax1.set_title('Accutance Profiles Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot difference on second axis
    mean_diff = np.array(profile1['mean_accutance']) - np.array(profile2['mean_accutance'])
    max_diff = np.array(profile1['max_accutance']) - np.array(profile2['max_accutance'])
    
    ax2.plot(profile1['positions'], mean_diff, label='Mean Difference', color='green')
    ax2.plot(profile1['positions'], max_diff, label='Max Difference', color='green', alpha=0.5)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    ax2.set_xlabel(f'{axis.upper()} Position')
    ax2.set_ylabel('Accutance Difference')
    ax2.set_title('Accutance Difference (Tile 1 - Tile 2)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    profiles = {
        'tile1': profile1,
        'tile2': profile2,
        'difference': {
            'positions': profile1['positions'],
            'mean_difference': mean_diff,
            'max_difference': max_diff
        }
    }
    
    return fig, [ax1, ax2], profiles

def calculate_block_accutance_profiles(tile_data, n_blocks=3, axis='z', slice_range=None, step=1):
    """
    Calculate accutance profiles for each block in an nxn grid of the tile.
    
    Args:
        tile_data: 3D numpy array (z,y,x)
        n_blocks: Number of blocks in each dimension (default 3 for 3x3 grid)
        axis: Axis along which to calculate profile ('z', 'y', or 'x')
        slice_range: Tuple of (start, end) indices. If None, uses full range
        step: Step size for sampling slices (default 1)
        
    Returns:
        profiles: Dict containing accutance profiles for each block
        block_bounds: Dict containing block boundaries
    """
    # Get dimensions
    z_dim, y_dim, x_dim = tile_data.shape
    
    # Calculate block sizes
    y_block_size = y_dim // n_blocks
    x_block_size = x_dim // n_blocks
    
    # Calculate block boundaries
    y_bounds = [(i * y_block_size, (i + 1) * y_block_size) for i in range(n_blocks)]
    x_bounds = [(i * x_block_size, (i + 1) * x_block_size) for i in range(n_blocks)]
    
    # Set up axis mapping
    axis_map = {'z': 0, 'y': 1, 'x': 2}
    axis_idx = axis_map[axis]
    
    # Determine slice range
    if slice_range is None:
        slice_range = (0, tile_data.shape[axis_idx])
    
    # Initialize arrays for profiles
    positions = range(slice_range[0], slice_range[1], step)
    profiles = {}
    
    # Calculate accutance for each block
    for i in range(n_blocks):
        for j in range(n_blocks):
            block_id = f'block_{i}_{j}'
            y_start, y_end = y_bounds[i]
            x_start, x_end = x_bounds[j]
            
            mean_accutance = []
            max_accutance = []
            
            # Calculate accutance for each slice
            for pos in positions:
                if axis == 'z':
                    slice_data = tile_data[pos, y_start:y_end, x_start:x_end]
                elif axis == 'y':
                    slice_data = tile_data[:, pos, x_start:x_end]
                else:  # x
                    slice_data = tile_data[:, x_start:x_end, pos]
                
                acc = calculate_accutance(slice_data)
                mean_accutance.append(acc['mean_accutance'])
                max_accutance.append(acc['max_accutance'])
            
            profiles[block_id] = {
                'positions': positions,
                'mean_accutance': mean_accutance,
                'max_accutance': max_accutance,
                'bounds': {
                    'y': (y_start, y_end),
                    'x': (x_start, x_end)
                }
            }
    
    block_bounds = {
        'y': y_bounds,
        'x': x_bounds
    }
    
    return profiles, block_bounds

def plot_block_accutance_profiles(tile_data, tile_name=None, n_blocks=3, axis='z', 
                                slice_range=None, step=1, plot_max=False):
    """
    Plot accutance profiles for each block in an nxn grid of the tile.
    
    Args:
        tile_data: 3D numpy array (z,y,x)
        tile_name: Name of tile for title (optional)
        n_blocks: Number of blocks in each dimension (default 3 for 3x3 grid)
        axis: Axis along which to calculate profile ('z', 'y', or 'x')
        slice_range: Tuple of (start, end) indices. If None, uses full range
        step: Step size for sampling slices (default 1)
        plot_max: Whether to plot max accutance (default False, plots mean)
        
    Returns:
        fig: Figure object
        ax: Axes object
        profiles: Dict containing accutance profiles for each block
    """
    # Calculate profiles for each block
    profiles, block_bounds = calculate_block_accutance_profiles(
        tile_data, n_blocks, axis, slice_range, step
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot profiles for each block with different colors
    colors = plt.cm.viridis(np.linspace(0, 1, n_blocks * n_blocks))
    
    for idx, (block_id, profile) in enumerate(profiles.items()):
        i, j = map(int, block_id.split('_')[1:])
        label = f'Block ({i},{j})'
        
        if plot_max:
            ax.plot(profile['positions'], profile['max_accutance'], 
                   label=label, color=colors[idx], alpha=0.7)
        else:
            ax.plot(profile['positions'], profile['mean_accutance'], 
                   label=label, color=colors[idx], alpha=0.7)
    
    # Add labels and title
    ax.set_xlabel(f'{axis.upper()} Position')
    ax.set_ylabel('Accutance')
    title = f'Block Accutance Profiles Along {axis.upper()} Axis'
    if tile_name:
        pos = parse_tile_name(tile_name)
        title += f'\nTile {pos}'
    if plot_max:
        title += ' (Maximum Values)'
    else:
        title += ' (Mean Values)'
    ax.set_title(title)
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    return fig, ax, profiles

def plot_block_accutance_heatmap(tile_data, z_slice, n_blocks=3, percentile_threshold=99, 
                                minmax_lims=(1, 99),
                               normalize_method: str = None):
    """
    Plot a heatmap of accutance values for each block in an nxn grid at a specific z-slice,
    alongside the original image slice with grid overlay and detected edges.
    
    Args:
        tile_data: 3D numpy array (z,y,x)
        z_slice: Z-slice index to analyze
        minmax_lims: tuple of (vmin, vmax) to use for image slice
        n_blocks: Number of blocks in each dimension (default 3 for 3x3 grid)
        percentile_threshold: Threshold for edge detection
        normalize_method: Method to normalize accutance values (default None)
    Returns:
        fig: Figure object
        axes: List of axes objects [ax_image, ax_heatmap, ax_edges]
        accutance_values: 2D array of accutance values
    """
    # Get dimensions
    _, y_dim, x_dim = tile_data.shape
    
    # Calculate block sizes
    y_block_size = y_dim // n_blocks
    x_block_size = x_dim // n_blocks
    
    # Initialize array for accutance values
    accutance_values = np.zeros((n_blocks, n_blocks))
    
    # Create an array to store the full edge mask for visualization
    full_edge_mask = np.zeros((y_dim, x_dim), dtype=bool)
    full_accutance_map = np.zeros((y_dim, x_dim))
    
    # Calculate accutance for each block
    for i in range(n_blocks):
        for j in range(n_blocks):
            y_start = i * y_block_size
            y_end = (i + 1) * y_block_size
            x_start = j * x_block_size
            x_end = (j + 1) * x_block_size
            
            block_data = tile_data[z_slice, y_start:y_end, x_start:x_end]
            
            acc = calculate_normalized_accutance(block_data, percentile_threshold, normalize_method)
            if normalize_method:
                accutance_values[i, j] = acc['normalized_mean_accutance']
            else:
                accutance_values[i, j] = acc['raw_mean_accutance']
            
            # Store edge mask and accutance map for visualization
            full_edge_mask[y_start:y_end, x_start:x_end] = acc['edge_mask']
            full_accutance_map[y_start:y_end, x_start:x_end] = acc['accutance_map']
    
    # Create figure with three subplots
    sns.set_context('talk')
    fig, (ax_image, ax_heatmap, ax_edges) = plt.subplots(1, 3, figsize=(24, 7))
    
    # Plot original image
    image_slice = tile_data[z_slice]
    if minmax_lims is None:
        im_image = ax_image.imshow(image_slice, cmap='gray')
    else:
        vmin, vmax = np.percentile(image_slice, minmax_lims)
        im_image = ax_image.imshow(image_slice, cmap='gray', vmin=vmin, vmax=vmax)
    ax_image.set_title(f'Original Image\nZ-slice {z_slice}')
    
    # Add grid lines to original image
    for i in range(1, n_blocks):
        ax_image.axhline(y=i * y_block_size, color='r', linestyle='--', alpha=0.5)
        ax_image.axvline(x=i * x_block_size, color='r', linestyle='--', alpha=0.5)
    
    # # Add block numbers to original image
    # for i in range(n_blocks):
    #     for j in range(n_blocks):
    #         center_y = (i + 0.5) * y_block_size
    #         center_x = (j + 0.5) * x_block_size
    #         ax_image.text(center_x, center_y, f'({i},{j})', 
    #                     ha='center', va='center', 
    #                     color='red', fontweight='bold',
    #                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    # Plot heatmap
    im_heatmap = ax_heatmap.imshow(accutance_values, cmap='magma')
    ax_heatmap.set_title(f'Accutance Heatmap\nNormalized: {normalize_method}')
    
    # Add colorbar to heatmap
    plt.colorbar(im_heatmap, ax=ax_heatmap, label='Accutance')
    
    # Add block labels to heatmap
    for i in range(n_blocks):
        for j in range(n_blocks):
            text = f'{accutance_values[i, j]:.2f}'
            # add text with a white background
            ax_heatmap.text(j, i, text, ha='center', va='center', 
                          color='black', fontweight='bold', fontsize=18,
                          bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    # Add labels to heatmap
    ax_heatmap.set_xticks(range(n_blocks))
    ax_heatmap.set_yticks(range(n_blocks))
    ax_heatmap.set_xlabel('X Block')
    ax_heatmap.set_ylabel('Y Block')
    
    # Add grid lines to heatmap
    ax_heatmap.grid(True, color='white', linestyle='-', alpha=0.2)
    
    # Plot edge visualization
    # Create an overlay with original image and detected edges
    edge_overlay = np.zeros((y_dim, x_dim, 3))
    # Add grayscale image to all channels
    for c in range(3):
        edge_overlay[:, :, c] = image_slice / np.max(image_slice) if np.max(image_slice) > 0 else 0
    
    # Highlight edges in red
    edge_overlay[full_edge_mask, 0] = 1.0  # Red channel
    edge_overlay[full_edge_mask, 1] = 0.0  # Green channel
    edge_overlay[full_edge_mask, 2] = 0.0  # Blue channel
    
    ax_edges.imshow(edge_overlay)
    ax_edges.set_title(f'Detected Edges\nPercentile: {percentile_threshold}')
    
    # Add grid lines to edge visualization
    for i in range(1, n_blocks):
        ax_edges.axhline(y=i * y_block_size, color='cyan', linestyle='--', alpha=0.5)
        ax_edges.axvline(x=i * x_block_size, color='cyan', linestyle='--', alpha=0.5)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig, [ax_image, ax_heatmap, ax_edges], accutance_values

####
# All tiles accutance
####
def calculate_tile_grid_block_accutance(tile_dict, bucket_name, dataset_path,
                                      z_slice, n_blocks=3, pyramid_level=0, percentile_threshold=99):
    """
    Calculate accutance values for 3x3 blocks within each tile across the entire tile grid.
    
    Args:
        tile_dict: Dictionary mapping tile IDs to tile names
        transforms: Dictionary mapping tile IDs to transformation matrices
        tile_names: Dictionary mapping tile IDs to tile names
        bucket_name: S3 bucket name
        dataset_path: Path to dataset in bucket
        z_slice: Z-slice to analyze
        n_blocks: Number of blocks per tile dimension (default 3)
        pyramid_level: Pyramid level to load
        percentile_threshold: Threshold for edge detection
        
    Returns:
        dict containing:
            grid_accutance: 2D array of shape (grid_y * n_blocks, grid_x * n_blocks) with accutance values
            tile_positions: Dictionary mapping tile IDs to their grid positions
            coverage_map: 2D boolean array showing tile presence
    """
    # First analyze the tile grid to get dimensions and positions
    grid_info = analyze_tile_grid(tile_dict, plot=False)
    grid_x, grid_y = grid_info['dimensions'][:2]
    coverage_map = grid_info['coverage_map']
    
    # Initialize the full accutance grid
    full_grid = np.full((grid_y * n_blocks, grid_x * n_blocks), np.nan)
    
    # Process each tile
    for tile_id, tile_name in tile_dict.items():
        print(f'Processing tile {tile_name}')
        # Extract tile position from name
        parts = tile_name.split('_')
        tile_x = int(parts[2])
        tile_y = int(parts[4])
        
        # Load and process tile
        tile_data = load_tile_data(tile_name, bucket_name, dataset_path, pyramid_level)
        
        # Calculate accutance for each block in the tile
        y_block_size = tile_data.shape[0] // n_blocks
        x_block_size = tile_data.shape[1] // n_blocks
        
        for i in range(n_blocks):
            for j in range(n_blocks):
                y_start = i * y_block_size
                y_end = (i + 1) * y_block_size
                x_start = j * x_block_size
                x_end = (j + 1) * x_block_size
                
                block_data = tile_data[x_start:x_end, y_start:y_end, z_slice]
                acc = calculate_accutance(block_data, percentile_threshold)
                
                # Calculate position in full grid
                grid_y_pos = tile_y * n_blocks + i
                grid_x_pos = tile_x * n_blocks + j
                
                full_grid[grid_y_pos, grid_x_pos] = acc['mean_accutance']
    
    return {
        'grid_accutance': full_grid,
        'coverage_map': coverage_map,
        'dimensions': (grid_x, grid_y),
        'blocks_per_tile': n_blocks
    }

def plot_tile_grid_block_accutance(grid_data, z_slice, show_tile_boundaries=True):
    """
    Plot the accutance heatmap for all tiles with their block subdivisions.
    
    Args:
        grid_data: Output from calculate_tile_grid_block_accutance
        z_slice: Z-slice being displayed
        show_tile_boundaries: Whether to show tile boundaries
        
    Returns:
        fig: Figure object
        ax: Axes object
    """
    grid_accutance = grid_data['grid_accutance']
    coverage_map = grid_data['coverage_map']
    grid_x, grid_y = grid_data['dimensions']
    n_blocks = grid_data['blocks_per_tile']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Plot heatmap
    im = ax.imshow(grid_accutance, cmap='viridis')
    plt.colorbar(im, ax=ax, label='Mean Accutance')
    
    # Add tile boundaries
    if show_tile_boundaries:
        for i in range(grid_y):
            ax.axhline(y=(i+1) * n_blocks - 0.5, color='red', linestyle='-', alpha=0.5)
        for j in range(grid_x):
            ax.axvline(x=(j+1) * n_blocks - 0.5, color='red', linestyle='-', alpha=0.5)
    
    # Add block grid lines
    for i in range(grid_accutance.shape[0]):
        ax.axhline(y=i-0.5, color='white', linestyle='-', alpha=0.1)
    for j in range(grid_accutance.shape[1]):
        ax.axvline(x=j-0.5, color='white', linestyle='-', alpha=0.1)
    
    # Add labels
    ax.set_title(f'Tile Grid Block Accutance Map (Z-slice {z_slice})')
    ax.set_xlabel('X Position (Blocks)')
    ax.set_ylabel('Y Position (Blocks)')
    
    # Add text showing accutance values
    for i in range(grid_accutance.shape[0]):
        for j in range(grid_accutance.shape[1]):
            if not np.isnan(grid_accutance[i, j]):
                text = f'{grid_accutance[i, j]:.2f}'
                ax.text(j, i, text, ha='center', va='center', 
                       color='white', fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    return fig, ax

def plot_tile_grid_block_accutance_with_image(tile_dict, transforms, tile_names, 
                                             bucket_name, dataset_path, z_slice, 
                                             n_blocks=3, pyramid_level=0):
    """
    Plot both the original stitched image and the accutance heatmap side by side.
    
    Args:
        ... (same as calculate_tile_grid_block_accutance) ...
        
    Returns:
        fig: Figure object
        axes: List of axes objects [ax_image, ax_heatmap]
    """
    # Calculate accutance values
    grid_data = calculate_tile_grid_block_accutance(
        tile_dict, transforms, tile_names, bucket_name, dataset_path,
        z_slice, n_blocks, pyramid_level
    )
    
    # Create figure with two subplots
    fig, (ax_image, ax_heatmap) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot stitched image
    # (You'll need to implement this part based on your stitching functionality)
    # For now, we'll just show the coverage map
    ax_image.imshow(grid_data['coverage_map'], cmap='gray')
    ax_image.set_title(f'Tile Coverage Map\nZ-slice {z_slice}')
    
    # Plot heatmap
    im = ax_heatmap.imshow(grid_data['grid_accutance'], cmap='viridis')
    plt.colorbar(im, ax=ax_heatmap, label='Mean Accutance')
    
    # Add tile boundaries
    grid_x, grid_y = grid_data['dimensions']
    n_blocks = grid_data['blocks_per_tile']
    
    for i in range(grid_y):
        ax_heatmap.axhline(y=(i+1) * n_blocks - 0.5, color='red', linestyle='-', alpha=0.5)
    for j in range(grid_x):
        ax_heatmap.axvline(x=(j+1) * n_blocks - 0.5, color='red', linestyle='-', alpha=0.5)
    
    # Add labels
    ax_heatmap.set_title('Block Accutance Heatmap')
    ax_heatmap.set_xlabel('X Position (Blocks)')
    ax_heatmap.set_ylabel('Y Position (Blocks)')
    
    plt.tight_layout()
    return fig, [ax_image, ax_heatmap]

def calculate_normalized_accutance(image_slice, percentile_threshold=99, normalization_method='edge_density'):
    """
    Calculate accutance (edge sharpness) normalized for feature density.
    
    Args:
        image_slice: 2D numpy array containing the image slice
        percentile_threshold: Percentile threshold for edge detection (default 99)
        normalization_method: Method for normalization
            - 'edge_density': Normalize by the percentage of pixels that are edges
            - 'edge_count': Normalize by the absolute count of edge pixels
            - 'local_contrast': Use local contrast normalization before edge detection
            - 'structure_tensor': Use eigenvalues of structure tensor approach
        
    Returns:
        dict containing normalized and raw accutance metrics
    """
    
    # Normalize image to [0,1]
    img_norm = image_slice.astype(float)
    if img_norm.max() > 0:
        img_norm = img_norm / img_norm.max()
    
    # Calculate gradients using Sobel operator
    grad_x = ndimage.sobel(img_norm, axis=1)
    grad_y = ndimage.sobel(img_norm, axis=0)
    
    # Calculate gradient magnitude
    accutance_map = np.sqrt(grad_x**2 + grad_y**2)
    
    # Create edge mask using threshold
    edge_threshold = np.percentile(accutance_map, percentile_threshold)
    edge_mask = accutance_map > edge_threshold
    edge_count = np.sum(edge_mask)
    total_pixels = edge_mask.size
    edge_density = edge_count / total_pixels if total_pixels > 0 else 0
    
    # Calculate statistics for detected edges
    edge_values = accutance_map[edge_mask]
    raw_mean_accutance = edge_values.mean() if edge_values.size > 0 else 0
    raw_max_accutance = edge_values.max() if edge_values.size > 0 else 0
    
    # Normalize based on selected method
    if normalization_method == 'edge_density':
        # Normalize by edge density - adjusts for different feature densities
        # Areas with few edges but sharp will score higher than areas with many edges but blurry
        norm_factor = max(0.001, edge_density)  # Avoid division by zero
        normalized_mean = raw_mean_accutance / norm_factor
        normalized_max = raw_max_accutance / norm_factor
        
    elif normalization_method == 'edge_count':
        # Use logarithmic scaling based on edge count
        norm_factor = max(1, np.log10(edge_count + 1))
        normalized_mean = raw_mean_accutance / norm_factor
        normalized_max = raw_max_accutance / norm_factor
        
    elif normalization_method == 'local_contrast':
        # This method needs to be applied before edge detection
        # Use local contrast normalization before calculating accutance
        local_mean = ndimage.uniform_filter(img_norm, size=15)
        local_std = np.sqrt(ndimage.uniform_filter(img_norm**2, size=15) - local_mean**2)
        local_std = np.maximum(local_std, 0.0001)  # Avoid division by zero
        normalized_img = (img_norm - local_mean) / local_std
        
        # Recalculate edges on contrast-normalized image
        norm_grad_x = ndimage.sobel(normalized_img, axis=1)
        norm_grad_y = ndimage.sobel(normalized_img, axis=0)
        norm_accutance_map = np.sqrt(norm_grad_x**2 + norm_grad_y**2)
        
        # Use same edge threshold approach
        norm_edge_threshold = np.percentile(norm_accutance_map, percentile_threshold)
        norm_edge_mask = norm_accutance_map > norm_edge_threshold
        norm_edge_values = norm_accutance_map[norm_edge_mask]
        
        normalized_mean = norm_edge_values.mean() if norm_edge_values.size > 0 else 0
        normalized_max = norm_edge_values.max() if norm_edge_values.size > 0 else 0
        
    elif normalization_method == 'structure_tensor':
        # Structure tensor approach (based on Harris corner detector)
        # This weights edge strength by local structure importance
        gaussian_filter = lambda x, sigma: ndimage.gaussian_filter(x, sigma)
        gx2 = gaussian_filter(grad_x * grad_x, 1.5)
        gy2 = gaussian_filter(grad_y * grad_y, 1.5)
        gxy = gaussian_filter(grad_x * grad_y, 1.5)
        
        # Eigenvalues represent strength of edge/corner response
        # For each pixel, calculate the trace and determinant
        trace = gx2 + gy2
        det = gx2 * gy2 - gxy * gxy
        
        # Calculate eigenvalues (λ1 ≥ λ2)
        # Using: λ1,λ2 = (trace/2) ± sqrt((trace/2)^2 - det)
        trace_half = trace / 2
        discriminant = np.sqrt(np.maximum(0, trace_half**2 - det))
        lambda1 = trace_half + discriminant  # Larger eigenvalue
        
        # Use largest eigenvalue as corner/edge strength measure
        structure_strength = lambda1
        normalized_mean = np.mean(structure_strength)
        normalized_max = np.max(structure_strength)
    
    else:
        normalized_mean = raw_mean_accutance
        normalized_max = raw_max_accutance
    
    return {
        'raw_mean_accutance': raw_mean_accutance,
        'raw_max_accutance': raw_max_accutance,
        'normalized_mean_accutance': normalized_mean,
        'normalized_max_accutance': normalized_max,
        'edge_density': edge_density,
        'edge_count': edge_count,
        'accutance_map': accutance_map,
        'edge_mask': edge_mask,
        'normalization_method': normalization_method
    }

def analyze_tile_overlap(tile1: TileData, tile2: TileData, 
                        transform1: np.ndarray, transform2: np.ndarray,
                        padding: int = 50):
    """
    Analyze the overlap between two tiles and return sliced data from overlapping regions.
    Automatically scales transformations based on pyramid level.
    """
    # Get tile dimensions
    tile1.connect()
    tile2.connect()
    
    # Scale transformations based on pyramid level
    def scale_transform(transform, pyramid_level):
        # Create scaled transform
        scale_factor = 2**pyramid_level
        scaled = transform.copy()
        # Scale translation components (last column)
        scaled[:2, 3] = scaled[:2, 3] / scale_factor
        return scaled
    
    # Scale transforms for current pyramid levels
    transform1_scaled = scale_transform(transform1, tile1.pyramid_level)
    transform2_scaled = scale_transform(transform2, tile2.pyramid_level)
    
    # Rest of the function remains the same but uses scaled transforms
    def get_tile_corners(shape, transform):
        y_dim, x_dim = shape[1:3]
        corners = np.array([
            [0, 0, 1],
            [x_dim, 0, 1],
            [x_dim, y_dim, 1],
            [0, y_dim, 1]
        ])
        
        global_corners = np.zeros((4, 2))
        for i, corner in enumerate(corners):
            transformed = transform[:2, :3] @ corner[:, np.newaxis]
            global_corners[i] = transformed.flatten() + transform[:2, 3]
            
        return global_corners
    
    corners1 = get_tile_corners(tile1.shape, transform1_scaled)
    corners2 = get_tile_corners(tile2.shape, transform2_scaled)
    
    # Calculate bounding boxes
    def get_bbox(corners):
        min_x = np.min(corners[:, 0])
        max_x = np.max(corners[:, 0])
        min_y = np.min(corners[:, 1])
        max_y = np.max(corners[:, 1])
        return np.array([[min_x, min_y], [max_x, max_y]])
    
    bbox1 = get_bbox(corners1)
    bbox2 = get_bbox(corners2)
    
    # Calculate overlap
    overlap_bbox = np.array([
        [max(bbox1[0, 0], bbox2[0, 0]), max(bbox1[0, 1], bbox2[0, 1])],
        [min(bbox1[1, 0], bbox2[1, 0]), min(bbox1[1, 1], bbox2[1, 1])]
    ])
    
    # Check if there is actually an overlap
    if (overlap_bbox[1] <= overlap_bbox[0]).any():
        print("No overlap between tiles")
        return None
    
    # For each tile, extend the overlap region inward by padding amount
    # If tile1 is on the left, extend right. If on right, extend left
    if bbox1[0, 0] < bbox2[0, 0]:  # tile1 is on the left
        overlap_bbox[1, 0] += padding  # extend right for tile1
        overlap_bbox[0, 0] -= padding  # extend left for tile2
    else:  # tile1 is on the right
        overlap_bbox[0, 0] -= padding  # extend left for tile1
        overlap_bbox[1, 0] += padding  # extend right for tile2
        
    # If tile1 is above, extend down. If below, extend up
    if bbox1[0, 1] < bbox2[0, 1]:  # tile1 is above
        overlap_bbox[1, 1] += padding  # extend down for tile1
        overlap_bbox[0, 1] -= padding  # extend up for tile2
    else:  # tile1 is below
        overlap_bbox[0, 1] -= padding  # extend up for tile1
        overlap_bbox[1, 1] += padding  # extend down for tile2
    
    # Convert global coordinates back to tile coordinates
    def global_to_tile_coords(points, transform):
        # Create full inverse transform (3x4 -> 3x3)
        full_transform = np.eye(3)
        full_transform[:2, :2] = transform[:2, :2]  # Copy rotation/scale part
        full_transform[:2, 2] = transform[:2, 3]    # Copy translation part
        inv_transform = np.linalg.inv(full_transform)
        
        # Convert points to homogeneous coordinates
        points_h = np.hstack([points, np.ones((points.shape[0], 1))])
        
        # Apply inverse transform
        tile_coords = points_h @ inv_transform.T
        return tile_coords[:, :2].T
    
    # Get slice bounds for each tile
    overlap_corners = np.array([
        [overlap_bbox[0, 0], overlap_bbox[0, 1]],  # Top-left
        [overlap_bbox[1, 0], overlap_bbox[0, 1]],  # Top-right
        [overlap_bbox[1, 0], overlap_bbox[1, 1]],  # Bottom-right
        [overlap_bbox[0, 0], overlap_bbox[1, 1]]   # Bottom-left
    ])
    
    tile1_coords = global_to_tile_coords(overlap_corners, transform1_scaled)
    tile2_coords = global_to_tile_coords(overlap_corners, transform2_scaled)
    
    # Get integer slice bounds
    def get_slice_bounds(coords, shape):
        min_x = max(0, int(np.floor(np.min(coords[0]))))
        max_x = min(int(np.ceil(np.max(coords[0]))), shape[2])
        min_y = max(0, int(np.floor(np.min(coords[1]))))
        max_y = min(int(np.ceil(np.max(coords[1]))), shape[1])
        return (min_y, max_y, min_x, max_x)
    
    tile1_bounds = get_slice_bounds(tile1_coords, tile1.shape)
    tile2_bounds = get_slice_bounds(tile2_coords, tile2.shape)
    
    # # Print debug info
    # print(f"Tile 1 global bbox: {bbox1}")
    # print(f"Tile 2 global bbox: {bbox2}")
    # print(f"Overlap bbox: {overlap_bbox}")
    # print(f"Tile 1 local coords:\n{tile1_coords.T}")
    # print(f"Tile 2 local coords:\n{tile2_coords.T}")
    
    return {
        'overlap_bbox': overlap_bbox,
        'tile1_bounds': tile1_bounds,
        'tile2_bounds': tile2_bounds,
        'global_bbox1': bbox1,
        'global_bbox2': bbox2,
        'tile1_coords': tile1_coords,
        'tile2_coords': tile2_coords
    }

def visualize_tile_overlap(tile1: TileData, tile2: TileData,
                          transform1: np.ndarray, transform2: np.ndarray,
                          z_slice: int, padding: int = 50,
                          verbose: bool = False):
    """
    Create an RGB visualization of overlapping regions between two tiles.
    Creates a composite image of just the overlap region plus padding.

    Note: this is before PairedTiles class implemented
    """
    # Get overlap information
    overlap_info = analyze_tile_overlap(tile1, tile2, transform1, transform2, padding)
    
    # Extract slices from each tile
    t1_y1, t1_y2, t1_x1, t1_x2 = overlap_info['tile1_bounds']
    t2_y1, t2_y2, t2_x1, t2_x2 = overlap_info['tile2_bounds']
    
    # Get data slices
    tile1_data = tile1.get_slice(z_slice, 'xy')[t1_y1:t1_y2, t1_x1:t1_x2]
    tile2_data = tile2.get_slice(z_slice, 'xy')[t2_y1:t2_y2, t2_x1:t2_x2]
    
    # Normalize data to [0,1]
    def normalize(data):
        if data.max() > data.min():
            return (data - data.min()) / (data.max() - data.min())
        return data - data.min()
    
    tile1_norm = normalize(tile1_data)
    tile2_norm = normalize(tile2_data)
    
    # Get bounding boxes
    bbox1 = overlap_info['global_bbox1']
    bbox2 = overlap_info['global_bbox2']
    overlap_bbox = overlap_info['overlap_bbox']
    
    # Print debug info for bounding boxes
    # print("Global bounding boxes:")
    # print(f"Tile 1: {bbox1}")
    # print(f"Tile 2: {bbox2}")
    # print(f"Overlap: {overlap_bbox}")
    
    # Calculate composite image dimensions
    height = int(np.ceil(overlap_bbox[1, 1] - overlap_bbox[0, 1]))
    width = int(np.ceil(overlap_bbox[1, 0] - overlap_bbox[0, 0]))
    
    
    # Create empty RGB image
    composite = np.zeros((height, width, 3))
    
    # Calculate relative positions in the overlap region
    def get_relative_coords(data_bbox):
        # Ensure coordinates are non-negative
        x = max(0, int(data_bbox[0, 0] - overlap_bbox[0, 0]))
        y = max(0, int(data_bbox[0, 1] - overlap_bbox[0, 1]))
        return y, x
    
    # Get positions and print debug info
    t1_y, t1_x = get_relative_coords(bbox1)
    t2_y, t2_x = get_relative_coords(bbox2)
    
    
    
    # Ensure we're not trying to place data outside the composite image
    def safe_place_data(data, y_start, x_start, channel):
        y_end = min(y_start + data.shape[0], composite.shape[0])
        x_end = min(x_start + data.shape[1], composite.shape[1])
        y_start = max(0, y_start)
        x_start = max(0, x_start)
        
        if y_end > y_start and x_end > x_start:
            y_data = min(data.shape[0], y_end - y_start)
            x_data = min(data.shape[1], x_end - x_start)
            composite[y_start:y_end, x_start:x_end, channel] = data[:y_data, :x_data]
    
    # Place the data safely
    safe_place_data(tile1_norm, t1_y, t1_x, 0)
    safe_place_data(tile2_norm, t2_y, t2_x, 1)

    if verbose:
        print(f"Composite dimensions: {width}x{height}")
        print("\nPlacement coordinates:")
        print(f"Tile 1: y={t1_y}, x={t1_x}, shape={tile1_norm.shape}")
        print(f"Tile 2: y={t2_y}, x={t2_x}, shape={tile2_norm.shape}")
    
    return {
        'composite': composite,
        'overlap_info': overlap_info,
        'tile1_data': tile1_data,
        'tile2_data': tile2_data,
        'image_coords': {
            'tile1': (t1_y, t1_x),
            'tile2': (t2_y, t2_x)
        }
    }

def plot_transformed_tiles(tile1: TileData, tile2: TileData, 
                         transform1: np.ndarray, transform2: np.ndarray,
                         padding: int = 50):
    """
    Create a visualization of the transformed tile boundaries and their overlap.
    Automatically scales transformations based on pyramid level.
    """    
    
    # Scale transformations based on pyramid level
    def scale_transform(transform, pyramid_level):
        scale_factor = 2**pyramid_level
        scaled = transform.copy()
        scaled[:2, 3] = scaled[:2, 3] / scale_factor
        return scaled
    
    # Scale transforms for current pyramid levels
    transform1_scaled = scale_transform(transform1, tile1.pyramid_level)
    transform2_scaled = scale_transform(transform2, tile2.pyramid_level)
    
    # Get tile corners and bounding boxes
    def get_tile_corners(shape, transform):
        y_dim, x_dim = shape[1:3]
        corners = np.array([
            [0, 0, 1],
            [x_dim, 0, 1],
            [x_dim, y_dim, 1],
            [0, y_dim, 1]
        ])
        
        global_corners = np.zeros((4, 2))
        for i, corner in enumerate(corners):
            transformed = transform[:2, :3] @ corner[:, np.newaxis]
            global_corners[i] = transformed.flatten() + transform[:2, 3]
            
        return global_corners
    
    corners1 = get_tile_corners(tile1.shape, transform1_scaled)
    corners2 = get_tile_corners(tile2.shape, transform2_scaled)
    
    # Add pyramid level info to title
    title = f"Transformed Tile Boundaries\n"
    title += f"Tile 1 shape: {tile1.shape} (level {tile1.pyramid_level})\n"
    title += f"Tile 2 shape: {tile2.shape} (level {tile2.pyramid_level})"
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot tile boundaries
    poly1 = Polygon(corners1, alpha=0.3, color='red', label='Tile 1')
    poly2 = Polygon(corners2, alpha=0.3, color='green', label='Tile 2')
    ax.add_patch(poly1)
    ax.add_patch(poly2)
    
    # Plot corner points
    ax.scatter(corners1[:, 0], corners1[:, 1], color='red', marker='o')
    ax.scatter(corners2[:, 0], corners2[:, 1], color='green', marker='o')
    
    # Add corner labels
    for i, (x, y) in enumerate(corners1):
        ax.annotate(f'T1-{i}', (x, y), xytext=(5, 5), textcoords='offset points')
    for i, (x, y) in enumerate(corners2):
        ax.annotate(f'T2-{i}', (x, y), xytext=(5, 5), textcoords='offset points')
    
    # Plot centers and translations
    center1 = np.mean(corners1, axis=0)
    center2 = np.mean(corners2, axis=0)
    ax.scatter([center1[0]], [center1[1]], color='red', marker='x', s=100, label='Tile 1 Center')
    ax.scatter([center2[0]], [center2[1]], color='green', marker='x', s=100, label='Tile 2 Center')
    
    # Plot translation vectors
    # ax.arrow(center1[0], center1[1], 
    #         transform1_scaled[0, 3], transform1_scaled[1, 3],
    #         head_width=20, head_length=20, fc='red', ec='red', alpha=0.5)
    # ax.arrow(center2[0], center2[1],
    #         transform2_scaled[0, 3], transform2_scaled[1, 3],
    #         head_width=20, head_length=20, fc='green', ec='green', alpha=0.5)
    
    # Calculate and plot overlap region if it exists
    bbox1 = np.array([[np.min(corners1[:, 0]), np.min(corners1[:, 1])],
                     [np.max(corners1[:, 0]), np.max(corners1[:, 1])]])
    bbox2 = np.array([[np.min(corners2[:, 0]), np.min(corners2[:, 1])],
                     [np.max(corners2[:, 0]), np.max(corners2[:, 1])]])
    
    overlap_bbox = np.array([
        [max(bbox1[0, 0], bbox2[0, 0]), max(bbox1[0, 1], bbox2[0, 1])],
        [min(bbox1[1, 0], bbox2[1, 0]), min(bbox1[1, 1], bbox2[1, 1])]
    ])
    
    if (overlap_bbox[1] > overlap_bbox[0]).all():
        overlap_corners = np.array([
            [overlap_bbox[0, 0], overlap_bbox[0, 1]],
            [overlap_bbox[1, 0], overlap_bbox[0, 1]],
            [overlap_bbox[1, 0], overlap_bbox[1, 1]],
            [overlap_bbox[0, 0], overlap_bbox[1, 1]]
        ])
        poly_overlap = Polygon(overlap_corners, alpha=0.3, color='yellow', label='Overlap')
        ax.add_patch(poly_overlap)
    
    # Set plot limits and labels
    ax.set_xlim(min(corners1[:, 0].min(), corners2[:, 0].min()) - padding, 
                max(corners1[:, 0].max(), corners2[:, 0].max()) + padding)
    ax.set_ylim(min(corners1[:, 1].min(), corners2[:, 1].min()) - padding, 
                max(corners1[:, 1].max(), corners2[:, 1].max()) + padding)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.legend()
    
    # Add title with shape information
    ax.set_title(title)
    
    return fig, ax

def visualize_orthogonal_views(self, z_slice=None, y_slice=None, x_slice=None, overlap_only=False):
    """
    Visualize orthogonal views of the paired tiles.
    Data is in (X,Y,Z) order.
    """
    # Use middle slices by default
    if x_slice is None:
        x_slice = self.composite_shape[0] // 2  # X is first dimension
    if y_slice is None:
        y_slice = self.composite_shape[1] // 2  # Y is middle dimension
    if z_slice is None:
        z_slice = self.composite_shape[2] // 2  # Z is last dimension
    
    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # XY slice (fix Z)
    xy_slice = self.composite[:, :, z_slice]  # Get XY plane at Z
    axes[0].imshow(xy_slice)
    axes[0].set_title(f"XY slice at Z={z_slice}")
    axes[0].set_xlabel('Y')
    axes[0].set_ylabel('X')
    
    # XZ slice (fix Y)
    xz_slice = self.composite[:, y_slice, :]  # Get XZ plane at Y
    axes[1].imshow(xz_slice)
    axes[1].set_title(f"XZ slice at Y={y_slice}")
    axes[1].set_xlabel('Z')
    axes[1].set_ylabel('X')
    
    # YZ slice (fix X)
    yz_slice = self.composite[x_slice, :, :]  # Get YZ plane at X
    axes[2].imshow(yz_slice)
    axes[2].set_title(f"YZ slice at X={x_slice}")
    axes[2].set_xlabel('Z')
    axes[2].set_ylabel('Y')
    
    # Add legend
    legend_elements = [
        Line2D([0], [0], color='r', lw=4, label=f'Tile 1 ({self.name1})'),
        Line2D([0], [0], color='g', lw=4, label=f'Tile 2 ({self.name2})'),
        Line2D([0], [0], color='y', lw=4, label='Overlap')
    ]
    for ax in axes:
        ax.legend(handles=legend_elements, loc='upper right')
    
    # Add overall title
    plt.suptitle(f"Orthogonal Views of Paired Tiles\n({self.name1}) (red) and ({self.name2}) (green)", fontsize=12)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    return fig, axes

# ---
# Tile Alignment QC
# ---
def qc_tile_alignment(stitched_xml, pairs, save_dir, bucket_name="aind-open-data", pyramid_level=3):
    """
    Generate tile alignment QC plots for all adjacent pairs.
    
    Parameters:
    -----------
    stitched_xml : dict
        Parsed BigStitcher XML data
    pairs : list
        List of adjacent tile pairs
    save_dir : Path
        Directory to save plots
    bucket_name : str
        S3 bucket name
    pyramid_level : int
        Pyramid level for analysis
    """
    from pathlib import Path
    
    # Ensure save directory exists
    save_dir = Path(save_dir/ "tile_alignment_qc")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    channels = ["405", "spots-avg"]
    
    print(f"Processing {len(pairs)} tile pairs for channels {channels}")
    
    for pair_idx, (tile1_name, tile2_name) in enumerate(pairs):
        print(f"Processing pair {pair_idx + 1}/{len(pairs)}: {tile1_name} <-> {tile2_name}")
        
        for channel in channels:
            try:
                print(f"  Channel: {channel}")
                
                # Generate plot for this pair and channel
                fig_tile_overlap_4_slices(
                    tile1_name, 
                    tile2_name,
                    stitched_xml, 
                    pyramid_level=pyramid_level,
                    channel=channel,
                    save=True,
                    output_dir=save_dir
                )
                
            except Exception as e:
                print(f"    Error processing pair {pair_idx} channel {channel}: {e}")
                continue
    
    print(f"QC plots saved to {save_dir}")