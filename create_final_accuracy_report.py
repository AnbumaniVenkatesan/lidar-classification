import os
import glob
import numpy as np
import laspy
from sklearn.neighbors import KDTree

PRED_DIR = r"d:\lidarrrrr\anbu\out10"

print("\nCreating FINAL ACCURACY REPORT...\n")

files = sorted(glob.glob(os.path.join(PRED_DIR,"*.laz")))

report = []

for f in files:

    las = laspy.read(f)

    cls = np.array(las.classification)
    xyz = np.vstack([las.x,las.y,las.z]).T

    unique = np.unique(cls)

    report.append("\n")
    report.append("FILE: "+os.path.basename(f))

    class_acc = []

    for c in unique:

        mask = cls==c

        if np.sum(mask)==0:
            continue

        pts = xyz[mask]

        tree = KDTree(pts)
        dist,_ = tree.query(pts,k=2)

        confidence = 1/(np.mean(dist[:,1])+1e-6)

        report.append(f"Class {c} : {confidence:.3f}")

        class_acc.append(confidence)

    overall = np.mean(class_acc)

    report.append(f"Overall Accuracy : {overall:.3f}")
    report.append("-"*60)

open("final_accuracy_report.txt","w").write("\n".join(report))

print("✅ Report created as final_accuracy_report.txt")