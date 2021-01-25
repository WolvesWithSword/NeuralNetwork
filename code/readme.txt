Pour lancer le programme, créer un fichier image dans le dossier d'execution du programme ou modifier la variable img_dir
situer dans programme.

Dans le dossier du quel le programme est executer créer un dossier model qui permetra au programme d'ecrire le model qu'il a train

Pour executer le programme lancer la main qui ce trouve dans le fichier Program.py, (après avoir créer un modele, il est possible 
de mettre en parametre autotrain a False pour eviter d'executer l'entrainement du modele qui est long)

De preference, executer le projet depuis la dossier source du code pour eviter les probleme de path, avec l'arboressence suivante:

source
|-code
|-|les script python
|
|-|image
|-|seg_pred
|-|-|seg_pred
|-|-|-|les dossier d'image (buildings,...)
|
|-|seg_test
|-|-|seg_pred
|-|-|-|les dossier d'image (buildings,...)
|
|-|seg_train
|-|-|seg_pred
|-|-|-|les image
|
|-|modele
|-|-| Les model enregistrer

