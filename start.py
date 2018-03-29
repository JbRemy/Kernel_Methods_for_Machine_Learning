import numpy as np
from Codes.svm import svm
import pandas as pd

regs = [40, 40, 40]

out = np.zeros((3000, 2))

out[:, 0] = [_ for _ in range(3000)]

for i in range(3):
    y = np.array(pd.read_csv('Data/Ytr{}.csv'.format(i)))
    X = np.load('Data/X_{}_embeded.npy'.format(i))
    X_te = np.load('Data/X_{}_test_embeded.npy'.format(i))

    for _ in range(y.shape[0]):
        if y[_, 1] == 0:
            y[_, 1] = -1

    SVM = svm()
    SVM.fit(X, y[:, 1], regs[i], kernel='polynomial', l=1)
    y_pred, scores = SVM.predict(X_te)

    for _ in range(len(y_pred)):
        if y_pred[_] == -1:
            y_pred[_] = 0

    out[i*1000: (i + 1)*1000, 1] = y_pred

Yte = pd.DataFrame(out, columns=["Id", "Bound"], dtype='int')
Yte.to_csv("Yte.csv", index=False)