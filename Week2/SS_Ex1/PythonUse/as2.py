import numpy as np
from numpy import trapz
import matplotlib.pyplot as plt
from sklearn import metrics
import os

test_eng = "english.test"
test_tag = "tagalog.test"

langs =["hiligaynon.txt", "middle-english.txt","plautdietsch.txt","xhosa.txt"] # Other four testing languages

def readtext(fname):
    with open(fname) as f:
        content = f.readlines()
    content = [float(x.strip()) for x in content]
    return content

def stats(x0,x1):
    AUC = SS_AUC(x0,x1)
    avgmatchx0 = np.mean(x0)
    avgmatchx1 = np.mean(x1)
    print("Average matching strings ground:",str(avgmatchx0) +"|avg anomaly:"+str(avgmatchx1) + ". AUC:", str(AUC))
    return avgmatchx0,avgmatchx1, AUC

def negsel(testfile, n=10,r=4):
    WorkDir = os.getcwd()
    javaDir = WorkDir + "\\negative-selection"
    # command = "java -jar negsel2.jar -self english.train -n 10 -r 4 -c -l < english.test > test.txt 2>&1"
    command = "java -jar negsel2.jar -self english.train -n " + str(n) + " -r " + str(r) + " -c -l < " + testfile + " > test.txt 2>&1"
    os.chdir(javaDir)
    os.system(command)
    results = readtext("test.txt") # The python variable
    os.chdir(WorkDir)
    return results

def SS_AUC(x0,x1):
    """
    :param x0: "normal" test set answers
    :param x1: "anomalies" test values
    :return:
    """
    x = sorted(np.concatenate([x0,x1]))
    n0 = len(x0)  # Normal
    n1 = len(x1)  # Anomalous
    nAll = n0+n1

    sensitivities = np.zeros(nAll)
    specificity = np.zeros(nAll)

    for i in range(nAll):
        sensitivities[i] = np.sum(x1 > x[i]) / len(x1)
        specificity[i] = np.sum(x0 < x[i]) / len(x0)

    return metrics.auc(1-specificity, sensitivities)

eng_means = []
tag_means =[]
all_AUC = []

for ri in range(1,10):
    xeng = negsel(test_eng, n=10, r=ri)
    xtag = negsel(test_tag, n=10, r=ri)

    ri_meanx0, ri_meanx1, ri_AUC = stats(xeng,xtag)
    eng_means.append(ri_meanx0)
    tag_means.append(ri_meanx1)
    all_AUC.append(ri_AUC)


# Plots
plt.figure()
plt.plot(range(1,10),eng_means, color=[0.8, 0.2, 0.2])
plt.plot(range(1,10),tag_means, color=[0.2, 0.2, 0.8])
plt.title("Avg # of matching strings")
plt.legend(["english.test", "tagalog.test"])
plt.xlabel("r")
plt.ylabel("Avg output")


plt.figure()
plt.plot(range(1,10),all_AUC, color=[0.8, 0.2, 0.2])
plt.title("AUC per r")
plt.xlabel("r")
plt.ylabel("AUC")


# Yourtask 3
nLang = len(langs)
nR = 9

all_means = np.zeros((nLang, nR))
all_AUC = np.zeros((nLang, nR))

for ri in range(1,nR +1):
    x0 = negsel(test_eng, n=10, r=ri)
    for langi in range(nLang):
        x1 = negsel(langs[langi], n=10, r=ri)
        ri_ignore, ri_mean, ri_AUC = stats(x0,x1)
        all_means[langi][ri-1] = ri_mean
        all_AUC[langi][ri-1] = ri_AUC


# Plots
plt.figure()
for i in range(4):
    plt.plot(range(1,10), all_means[i])
plt.legend(langs)
plt.title("Avg # of matching strings")
plt.xlabel("r")
plt.ylabel("Avg output")

plt.figure()
for i in range(4):
    plt.plot(range(1,10), all_AUC[i])
plt.title("AUC per r")
plt.xlabel("r")
plt.ylabel("AUC")
plt.legend(langs)



# AUC testing

x0 = negsel(test_eng)
x1 = negsel(test_tag)

