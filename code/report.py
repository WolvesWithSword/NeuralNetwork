import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

#gestion des label pour le graphique en bar
def autolabel(ax,rects,fontSize):
    #Attach a text label above each bar in *rects*, displaying its height.
    for rect in rects: 
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom', fontsize=fontSize, rotation=90)

#permet de faire un report textuel des données (matrice de confusion)
def reportText(y_pred,realpred,label):
    print("Matrice de confusion :")
    print(confusion_matrix(realpred, y_pred))
    print("report :")
    print(classification_report(realpred, y_pred, target_names=label))

#permet de créer le graphe avec le score pour chaque classe
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

#permet de créer le graphe du recall des classe
def reportRepartition(y_pred,realpred,label):
    confusion = confusion_matrix(realpred, y_pred)
    allCircle = []
    i = 0
    for entity in label:
        total = sum(confusion[i])
        # on fait un pourcentage sur le total
        allCircle.append([elt*100/total for elt in confusion[i]])
        print(allCircle[i])
        i+=1

    fig, axArray = plt.subplots(3,2,figsize=(20,20))
    i = 0
    for line in axArray:
        for ax in line:
            legend=label[:]
            for numLab in range(len(legend)):
                legend[numLab] = legend[numLab] + " " + str(round(allCircle[i][numLab],2))+"% ("+ str(confusion[i][numLab]) + " img)"
            wedges, texts = ax.pie(allCircle[i])
            ax.legend(wedges, legend,
                    title="class - total : "+str(sum(confusion[i]))+" img",
                    loc="center left",
                    bbox_to_anchor=(1, 0, 0.5, 1))

            #plt.setp(autotexts, size=8, weight="bold")

            ax.set_title(label[i])
            i+=1
    plt.savefig("pie_diagram.png")
    plt.show()

    
