from matplotlib import pyplot as plt
from aucStuff import cal_roc, calc_auc
import numpy as np
import subprocess

# check readme.md for explanation of exercise

test_eng = "english.test"
test_tag = "tagalog.test"
langs = ["hiligaynon.txt", "middle-english.txt", "plautdietsch.txt", "xhosa.txt"]


# call the negative selection on every string in the given test file, and return
# an array with the score for every string
def negativeSelection(testFile, r):
    # call on each line in testFile
    matchesPerStr = subprocess.check_output("java -jar negsel2.jar -self english.train -n 10 -r " + str(r) + " -c -l < " + testFile, shell=True)
    # convert to array of floats
    res = [float(i) for i in matchesPerStr.splitlines()]
    return np.asarray(res)



avgMatchesEng = negativeSelection(test_eng, r=4)
avgMatchesTag = negativeSelection(test_tag, r=4)

sens, spec = cal_roc(avgMatchesEng,avgMatchesTag)
auc = calc_auc(sens,spec)

print ("AUC for r=4 is", auc)

plt.figure()
plt.plot(1-spec,sens)
plt.plot(np.arange(0,1.1,0.1),np.arange(0,1.1,0.1), color='orange', linestyle='--')
plt.xlim([0,1])
plt.ylim([0,1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
