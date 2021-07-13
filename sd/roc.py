from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

fpr = np.load('fpr-plain.npy')
tpr = np.load('tpr-plain.npy')
fpr2 = np.load('fpr-sultani.npy')
tpr2 = np.load('tpr-sultani.npy')
fpr3 = np.load('../rtfm/fpr-rtfm.npy')
tpr3 = np.load('../rtfm/tpr-rtfm.npy')

#plt.subplots(1, figsize=(10,10))
#plt.title('Receiver Operating Characteristic - CAD')
#plt.plot(fpr, tpr)
#plt.plot([0, 1], ls="--")
#plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
#plt.ylabel('True Positive Rate')
#plt.xlabel('False Positive Rate')
#plt.savefig('roc.png')

plt.plot(fpr, tpr, linestyle='solid', label='Ours')
plt.plot(fpr2, tpr2, linestyle='solid', label='Sultani et al.')
plt.plot(fpr3, tpr3, linestyle='solid', label='Tian et al.')

plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend()
plt.savefig('roc-5.png')
