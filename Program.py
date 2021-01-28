from PRE.PRE import *
from FIG.FIG import *
from R.R import *

savePath = "model.h5"
# Methode principale qui permet de créer le model et l'executer
#autoTrain permet de dire si l'on veux train le modele avant d'executer les test
def main(autotrain = True):

    #permet de train le modele (mettre en commentaire si le modele est train)
    if(autotrain):
        autoTrain(60,430,savePath)

    #permet de charger le model
    models = loadModel(savePath)
    models.summary()

    #permet de predict une image et d'avoir les information demander
    predict(models,img_dir+"seg_test/seg_test/glacier/20087.jpg",'probabilities')

    #report sur le jeu de test
    y_test,real = reportTestData(models)
    reportText(y_test,real,LABEL)
    reportGraphe(y_test,real,LABEL)
    reportRepartition(y_test,real,LABEL)


    
#permet de lancer le programme
if __name__ == "__main__":
    main(False)