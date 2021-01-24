import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


def autolabel(ax,rects,fontSize):
    #Attach a text label above each bar in *rects*, displaying its height.
    for rect in rects: 
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom', fontsize=fontSize, rotation=90)


def reportText(y_pred,realpred,label):
    print("Matrice de confusion :")
    print(confusion_matrix(realpred, y_pred))
    print("report :")
    print(classification_report(realpred, y_pred, target_names=label))

def reportGraphe(y_pred,realpred,label):
    report = classification_report(realpred, y_pred, target_names=label,output_dict=True)
    precision=[]
    recall=[]
    f1score=[]
    for entity in label:
        precision.append(report[entity]["precision"])
        recall.append(report[entity]["recall"])
        f1score.append(report[entity]["f1-score"])
    precisionCut = [round(scr,2) for scr in precision]
    recallCut = [round(scr,2) for scr in recall]
    f1scoreCut = [round(scr,2) for scr in f1score]

    x = np.arange(len(label))  # the label locations
    width = 0.7  # the width of the bars

    fig, ax = plt.subplots(figsize=(20, 10))
    rects1 = ax.bar(x - width/3, precisionCut, width/3, color='b', label='precision')

    rects2 = ax.bar(x , recallCut, width/3, color='g', label='recall')

    rects3 = ax.bar(x + width/3, f1scoreCut, width/3, color='y', label='f1-score')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Value (%)')

    ax.set_title('Precision, Recall and F1-score for each class')
    ax.set_xticks(x)
    ax.set_xticklabels(label)
    
    plt.legend(handles=[rects1, rects2,rects3], title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left')

    #label rotation
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)

    autolabel(ax,rects1,10)
    autolabel(ax,rects2,10)
    autolabel(ax,rects3,10)

    fig.tight_layout()

    plt.subplots_adjust(left=0.05, bottom=0.2, right=0.86, top=0.95, wspace=0.2, hspace=0.2)
    plt.savefig("class_plot.png")
    plt.show()
