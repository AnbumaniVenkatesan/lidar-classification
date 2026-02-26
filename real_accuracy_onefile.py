import laspy
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix

GT   = r"D:\lidarrrrr\anbu\test\GT_000005_MODEL.laz"
PRED = r"d:\lidarrrrr\anbu\test\DX3013595 PASQUILIO000005_PRED.laz"

gt = laspy.read(GT)
pr = laspy.read(PRED)

y_true = np.array(gt.classification)
y_pred = np.array(pr.classification)

mask = np.isin(y_true,[1,2,3,6])

y_true = y_true[mask]
y_pred = y_pred[mask]

rep = classification_report(y_true,y_pred,labels=[1,2,3,6],output_dict=True,zero_division=0)
cm  = confusion_matrix(y_true,y_pred,labels=[1,2,3,6])

print("\nREAL ACCURACY\n")

for c in [2,3,6]:
    acc = rep[str(c)]['recall']*100
    print(f"Class {c} Accuracy: {acc:.2f}%")

overall = np.trace(cm)/np.sum(cm)*100
print(f"\nOverall Accuracy: {overall:.2f}%")