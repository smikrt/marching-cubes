import os
from collections import defaultdict

import pydicom
import numpy as np

folder = r"C:\Users\tkimt\coding_projects\polyscope_project\DICOM"

series = defaultdict(list)

# Read DICOM files and group normal 512x512 CT slices by series
for name in os.listdir(folder):
    path = os.path.join(folder, name)

    try:
        ds = pydicom.dcmread(path)
        arr = ds.pixel_array

        if arr.shape == (512, 512):
            series[ds.SeriesInstanceUID].append(ds)

    except Exception as e:
        pass

print("Found series:")
for uid, imgs in series.items():
    desc = getattr(imgs[0], "SeriesDescription", "No description")
    print(len(imgs), uid, "|", desc)

# Choose the largest 512x512 stack
slices = max(series.values(), key=len)

print("\nChosen series:")
print("slices before remove:", len(slices))
print("SeriesDescription:", getattr(slices[0], "SeriesDescription", "No description"))
print("SliceThickness:", getattr(slices[0], "SliceThickness", "None"))
print("SpacingBetweenSlices:", getattr(slices[0], "SpacingBetweenSlices", "None"))

# Compute slice direction
iop = np.array(slices[0].ImageOrientationPatient, dtype=float)
row_dir = iop[:3]
col_dir = iop[3:]
normal = np.cross(row_dir, col_dir)

def slice_position(ds):
    ipp = np.array(ds.ImagePositionPatient, dtype=float)
    return np.dot(ipp, normal)

# Sort slices in correct z order
slices.sort(key=slice_position)

# Remove first bad slice
slices = slices[1:]

print("slices after remove:", len(slices))

# Compute positions after removing bad slice
positions = np.array([slice_position(s) for s in slices], dtype=float)

# x/y spacing from DICOM metadata
sx = float(slices[0].PixelSpacing[1])
sy = float(slices[0].PixelSpacing[0])

# z spacing from slice positions if possible
diffs = np.diff(np.sort(positions))
diffs = diffs[np.abs(diffs) > 1e-4]

if len(diffs) > 0:
    sz = float(np.median(np.abs(diffs)))
else:
    sz = float(getattr(
        slices[0],
        "SpacingBetweenSlices",
        getattr(slices[0], "SliceThickness", 1.0)
    ))

print("spacing:", sx, sy, sz)

# Stack slices into 3D volume
volume = np.stack([s.pixel_array for s in slices]).astype(np.float32)

# Convert raw pixel values to CT Hounsfield Units
slope = float(getattr(slices[0], "RescaleSlope", 1))
intercept = float(getattr(slices[0], "RescaleIntercept", 0))

print("RescaleSlope:", slope)
print("RescaleIntercept:", intercept)

volume = volume * slope + intercept

print("HU min/max:", volume.min(), volume.max())

# Save raw volume
volume.tofile("volume.raw")

# Save size and spacing info
with open("volume_info.txt", "w") as f:
    f.write(f"{volume.shape[2]} {volume.shape[1]} {volume.shape[0]}\n")
    f.write(f"{sx} {sy} {sz}\n")

print("saved volume.raw")
print("saved volume_info.txt")
print("size:", volume.shape[2], volume.shape[1], volume.shape[0])
