# constants.py
CHANNEL_COLORS = {
    "405": "#9E9E9E",  # Grey
    "488": "#4CAF50",  # Soft green
    "514": "#F44336",  # Vibrant red
    "561": "#2196F3",  # Sky blue
    "594": "#00BCD4",  # Teal
    "638": "#9C27B0"   # Rich purple
}

VOXEL_SIZE = [0.245055, 0.245055, 1.0]  # [X, Y, Z]
EASI_FISH_THRESHOLD = 0.01

# For data paths, thresholds, etc.
DEFAULT_VOLUME_PERCENTILES = (5, 95)
DEFAULT_SPOT_THRESHOLD = 10