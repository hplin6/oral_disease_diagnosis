from pycm import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

def plot_confusion_matrix(cmat, title = "Confusion matrix",
                          cmap = plt.cm.Blues, classnum=5,normalize= False, save_flg = True):

    classes = ['Normal','Ulcer','Low-risk','High-risk',"Cancer"]#[str(i) for i in range(classnum)]
    labels = range(classnum)

    if normalize:
        cmat = cmat.astype('float') / cmat.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure(figsize=(14, 12))
    plt.imshow(cmat, interpolation='nearest', cmap=cmap)
    #plt.title(title, fontsize=40)
    cb=plt.colorbar()
    cb.ax.tick_params(labelsize=20)  # 设置色标刻度字体大小。
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=20)
    plt.yticks(tick_marks, classes, fontsize=20)

    fmt = '.3f' if normalize else 'd'
    thresh = cmat.max() / 2.
    for i, j in itertools.product(range(cmat.shape[0]), range(cmat.shape[1])):
        plt.text(j, i, format(cmat[i, j],fmt),
                 horizontalalignment="center",
                 color="white" if cmat[i, j] > thresh else "black", fontsize=24)
        plt.text(j, i, format(cmat[i, j],fmt),
                 horizontalalignment="center",
                 color="white" if cmat[i, j] > thresh else "black", fontsize=24)

    plt.ylabel('True label', fontsize=26)
    plt.xlabel('Predicted label', fontsize=26)

    if save_flg:
        plt.savefig("./output/confusion_matrix.png")
    #plt.show()



def plot_ci(cm,param,alpha=0.05,method="normal-approx"):
    """
    Plot two-sided confidence interval.
    
    :param cm: ConfusionMatrix
    :type cm : pycm.ConfusionMatrix object
    :param param: input parameter
    :type param: str
    :param alpha: type I error
    :type alpha: float
    :param method: binomial confidence intervals method
    :type method: str
    :return: None
    """
    conf_str = str(round(100*(1-alpha)))
    print(conf_str+"%CI :")
    if param in cm.class_stat.keys():
        mean = []
        error = [[],[]]
        data = cm.CI(param,alpha=alpha,binom_method=method)
        class_names_str = list(map(str,(cm.classes)))
        for class_index, class_name in enumerate(cm.classes):
            print(str(class_name)+" : "+str(data[class_name][1]))
            mean.append(cm.class_stat[param][class_name])
            error[0].append(cm.class_stat[param][class_name]-data[class_name][1][0])
            error[1].append(data[class_name][1][1]-cm.class_stat[param][class_name])
        fig = plt.figure()
        plt.errorbar(mean,class_names_str,xerr = error,fmt='o',capsize=5,linestyle="dotted")
        plt.ylabel('Class')
        fig.suptitle("Param :"+param + ", Alpha:"+str(alpha), fontsize=16)
        for index,value in enumerate(mean):
            down_point = data[cm.classes[index]][1][0]
            up_point = data[cm.classes[index]][1][1]
            plt.text(value, class_names_str[index], "%f" %value, ha="center",va="top",color="red")
            plt.text(down_point, class_names_str[index], "%f" %down_point, ha="right",va="bottom",color="red")
            plt.text(up_point , class_names_str[index], "%f" %up_point, ha="left",va="bottom",color="red")
    else:
        mean = cm.overall_stat[param]
        data = cm.CI(param,alpha=alpha,binom_method=method)
        print(data[1])
        error = [[],[]]
        up_point = data[1][1]
        down_point = data[1][0]
        error[0] = [cm.overall_stat[param] - down_point]
        error[1] = [up_point - cm.overall_stat[param]]
        fig = plt.figure()
        plt.errorbar(mean,[param],xerr = error,fmt='o',capsize=5,linestyle="dotted")
        fig.suptitle("Alpha:"+str(alpha), fontsize=16)
        plt.text(mean, param, "%f" %mean, ha="center",va="top",color="red")
        plt.text(down_point, param, "%f" %down_point, ha="right",va="bottom",color="red")
        plt.text(up_point, param, "%f" %up_point, ha="left",va="bottom",color="red")

    plt.show()

csv_filename="./output/test_result.csv"
den = pd.read_csv(csv_filename, sep=',',header=None).values
den = den.astype(np.float32, copy=False)
#den=np.loadtxt(csv_filename)
y_true=den[0][:]
y_pred=den[1][:]

cmat = confusion_matrix(y_true, y_pred)
print(cmat)
plot_confusion_matrix(cmat)

cm = ConfusionMatrix(actual_vector=y_true, predict_vector=y_pred) # Create CM From Data
print(cm.classes)
print(cm.table)
print(cm)
print(cm.binary,cm.imbalance,cm.recommended_list)
plot_ci(cm,param="AUC")
