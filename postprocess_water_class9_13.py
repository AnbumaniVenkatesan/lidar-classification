import os
import numpy as np
import laspy
from tqdm import tqdm
from scipy.ndimage import label

# ---------------------------
IN_LAZ  = r"d:\lidarrrrr\anbu\classified_output_HAGfix_out7.laz"
OUT_LAZ = r"d:\lidarrrrr\anbu\classified_output_FINAL.laz"
# ---------------------------

SEA_CLASS = 9
LAKE_CLASS = 13
GROUND_CLASS = 2
DEFAULT_CLASS = 1

GRID = 1.0     # 1 meter grid
MIN_WATER_SIZE = 10   # meters

print("Reading:", IN_LAZ)
las = laspy.read(IN_LAZ)

x = np.array(las.x, dtype=np.float32)
y = np.array(las.y, dtype=np.float32)
z = np.array(las.z, dtype=np.float32)
cls = np.array(las.classification, dtype=np.int32)

has_intensity = "intensity" in las.point_format.dimension_names
if has_intensity:
    intensity = np.array(las.intensity, dtype=np.float32)
    intensity = (intensity - intensity.min())/(intensity.max()-intensity.min()+1e-6)

print("Intensity available:", has_intensity)

xmin, ymin = x.min(), y.min()
ix = np.floor((x - xmin)/GRID).astype(np.int32)
iy = np.floor((y - ymin)/GRID).astype(np.int32)

nx = ix.max()+1
ny = iy.max()+1

grid_zvar = np.zeros((nx,ny),dtype=np.float32)
grid_count = np.zeros((nx,ny),dtype=np.int32)
grid_int = np.zeros((nx,ny),dtype=np.float32)

print("Building flatness grid...")
for i in tqdm(range(len(x))):
    cx = ix[i]; cy = iy[i]
    grid_zvar[cx,cy]+=z[i]
    grid_count[cx,cy]+=1
    if has_intensity:
        grid_int[cx,cy]+=intensity[i]

valid = grid_count>5
grid_zvar[valid] = grid_zvar[valid]/grid_count[valid]
if has_intensity:
    grid_int[valid] = grid_int[valid]/grid_count[valid]

water_mask = valid.copy()

# flat areas
for i in tqdm(range(len(x)),desc="Flat test"):
    if not valid[ix[i],iy[i]]: continue
    if abs(z[i]-grid_zvar[ix[i],iy[i]])>0.2:
        water_mask[ix[i],iy[i]] = False

if has_intensity:
    water_mask = water_mask & (grid_int<0.25)

print("Detecting connected water bodies...")
structure = np.ones((3,3))
labeled, ncomp = label(water_mask,structure=structure)

out_cls = cls.copy()

print("Classifying water bodies...")
for comp in tqdm(range(1,ncomp+1)):
    pts = np.where(labeled==comp)
    if len(pts[0])==0: continue
    width = max(np.ptp(pts[0]), np.ptp(pts[1])) * GRID

    if width>=MIN_WATER_SIZE:
        # inland water
        mask = (labeled[ix,iy]==comp)
        out_cls[mask] = LAKE_CLASS
    else:
        mask = (labeled[ix,iy]==comp)
        out_cls[mask] = SEA_CLASS

out = laspy.LasData(header=las.header)
out.points = las.points
out.classification = out_cls.astype(np.uint8)
out.write(OUT_LAZ)

print("\nSaved:",OUT_LAZ)
u,c = np.unique(out_cls,return_counts=True)
for a,b in zip(u,c):
    print("Class",a,":",b)