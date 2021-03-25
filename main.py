from numpy import dot
from numpy.linalg import norm
import scipy as scp
import fonction as fct
import matplotlib.pyplot as plt
import itertools as itr
import math
import numpy as np
import random
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.spatial import distance_matrix
from sklearn.isotonic import IsotonicRegression
from sklearn.datasets import make_regression



som = fct.Cluster()
condition,W,C = som.accueil()
data = som.prétraitement()
if condition in [0,2]:
	W,C = som.som()
else:
	pass

item = random.choice([x for x in range(len(data))])
n1 = som.distance(v=data[item],W=W)
biais = [0,0]
H0 = -1
dim = -1
low = -1
high = -1
taille = -1

reglage = -1
while reglage != 0:
	reglage = input("Que voulez vous faire ? \n \
					0) Arrêter le programme \n \
					1) Acceder aux réglage de la MCMC \n\
					2) Explorer les Sous Additivité / Surper Additivité et Weak Evidence Effect ? \n")
	reglage = int(reglage)

	if reglage == 1:
		choix_reglage = -1
		while choix_reglage != 0:
			choix_reglage = input("Choisissez un paramètre \n \
				0) Quitter \n \
				1) Biais \n \
				2) Hypothèse initiale \n \
				3) Nombre de dimensions du stimulus \n \
				4) La fenêtre et le poid du Pool d'hypothèses \n")
			choix_reglage = int(choix_reglage)
			if choix_reglage == 1:
				biais[1] = input("Entrez la position de l'hypothèses que vous voulez biaiser. \n La position correspond à la distance de l'hypothèse avec le stimulus, plus il est élevé, moins l'hypothèse sera vraisemblable\n")
				biais[1] = int(biais[1])
				biais[0] = input("Entrez la valeur du biais. \n Le biais se surajoute à la probabilité à priori, un biais trop élevé pourra faire dépasser la prieur de 1\n")
				biais[0] = int(biais[0])
			elif choix_reglage == 2:
				H0 = input("Entrez le rang de l'hypothèse que vous voulez voir amorcer la chaîne dans la condition 1. \n Plus le rang est élevé, moins l'hypothèse qui amorçera la chaîne sera vraisemblable. \n Si vous ne choisissez pas une fenêtre pour le pool d'hypothèse, il sera automatiquement formé des 4 hypothèses les plus proches de celle que vous choisirez \n")
				H0 = int(H0)
			elif choix_reglage == 3:
				print("Sachant qu'il y a ",len(data[0])," dimensions")
				dim = input("combien voulez vous que le stimulus ait de dimensions ?")
				dim = int(dim)
			elif choix_reglage == 4:
				taille = input("Si vous ne voulez que changer la taille de la fenêtre, rentrez la taille que vous désirez et donnez, au deux prochaines question, la même réponse")
				taille = int(taille)
				low = input("Entrez la position du premier noeud. \n La position du noeud correspond à sa distance du stimulus \n")
				low = int(low)
				high = input("Entrez la position du dernier noeud \n")
				high = int(low)
				if low == high:
					low = -1

		mcmc = fct.MCMC(W=W,C=C,test=data[random.choice([x for x in range(len(data))])],biais=0.5,cov=distance_matrix(W,data,p=2,threshold=1000))
		mcmc.prétraitement()

		nb_mcmc = 10
		len_mcmc = 500
		pas =20

		packed = [0 for x in range(nb_mcmc)]
		unpacked = [0 for x in range(nb_mcmc)]
		

		# Choix des noeuds
		# On récupère les noeuds les plus proches de l'élément sur lequel on veut générer des hypothèses
		
		# Ensuite on choisit un des 5 noeuds les plus proches
		if H0 != -1:
			n1 = H0
		else:
			n1 = random.choice(n1[0:5])
			n1 = n1[1]
		# On sélectionne les 4 noeuds les plus proches de celui ci, ils constitueront le cluster d'hypothèses que nous testerons
		n2 = som.distance(v=W[n1],W=W)
		n2 = [x[1] for x in n2 if x[1] != n1]

		if low == -1 and taille == -1:
			pool = [n1] + n2[0:5]
		elif taille != -1:
			pool = [n1] + n2[0:taille]
		else:
			pool = [n1] + n2[low:high]

		if H0 != -1:
			poi = [1] + [0 for x in range(len(pool)-1)]
		else:
			poi = [len(pool)-2,len(pool)-2] + [1 for x in range(len(pool)-2)]
			random.shuffle(pool)

		# Choix du nombre de dimensions qui seront utilisé pour deviner l'hypothèse la plus vraisemblable
		if dim == -1:
			k = len(data[0])
		else:
			k = dim

		dim = random.choices([x for x in range(len(data[item]))],k=k)


		for i in range(nb_mcmc):
			# Les 5 noeuds n'ont pas la même probabilité d'apparaître puisque les deux noeuds décompréssé ont la même probabilité d'initialisation que les 3 autres réunit
			init = random.choices(pool,weights=poi)
			unpacked[i],accpt,test = mcmc.mh(len_mcmc,init=init[0],test=data[item],dim=dim,testif=1,bias=biais)
			# Tous les noeuds sont à égalité,
			init = random.choice(pool)
			packed[i],accpt,test = mcmc.mh(len_mcmc,init=init,test=data[item],dim=dim,testif=1)

		deb = 0
		j = 0
		p_packed = [0 for x in range(pas,len_mcmc+1,pas)]
		p_unpacked = [0 for x in range(pas,len_mcmc+1,pas)]
		for i in range(pas,len_mcmc+1,pas):
			raw_packed = [packed[k][0:i] for k in range(nb_mcmc)]
			raw_unpacked = [unpacked[k][0:i] for k in range(nb_mcmc)]
			p_packed[j] = mcmc.pvalue(raw_packed,pool)
			p_unpacked[j] = mcmc.pvalue(raw_unpacked,pool)
			j+=1
			deb = i

		pval = [sum(p_packed[x])-sum(p_unpacked[x]) for x in range(len(p_packed))]
		plt.plot(pval)
		plt.show()



	else:

		for cyc in range(3):

			mcmc = fct.MCMC(W=W,C=C,test=data[random.choice([x for x in range(len(data))])],biais=0.5,cov=distance_matrix(W,data,p=2,threshold=1000))
			mcmc.prétraitement()
			nb_mcmc = 10
			len_mcmc = 500
			pas =20

			packed = [0 for x in range(nb_mcmc)]
			unpacked = [0 for x in range(nb_mcmc)]
			item = random.choice([x for x in range(len(data))])

			# Choix des noeuds
			# On récupère les noeuds les plus proches de l'élément sur lequel on veut générer des hypothèses
			n1 = som.distance(v=data[item],W=W)
			# Ensuite on choisit un des 5 noeuds les plus proches
			n1 = random.choice(n1[0:5])
			# On sélectionne les 4 noeuds les plus proches de celui ci, ils constitueront le cluster d'hypothèses que nous testerons
			n2 = som.distance(v=W[n1[1]],W=W)
			n2 = [x[1] for x in n2 if x[1] != n1[1]]
			# On constitue le pool des noeuds qui vont être évalué
			if cyc == 0:
				pool = [n1[1]] + n2[0:5]
				random.shuffle(pool)
			elif cyc == 1:
				pool = [n1[1]] + n2[20:25]
				random.shuffle(pool)
			else:
				n1 = som.distance(v=data[item],W=W)
				
				pool = [n1[20]]+[x[1] for x in n1[14:19]]
				n1 = random.choice(n1[0:5])
			

			# Choix du nombre de dimensions qui seront utilisé pour deviner l'hypothèse la plus vraisemblable

			k = len(data[0])

			dim = random.choices([x for x in range(len(data[item]))],k=k)


			for i in range(nb_mcmc):
				# Les 5 noeuds n'ont pas la même probabilité d'apparaître puisque les deux noeuds décompréssé ont la même probabilité d'initialisation que les 3 autres réunit
				
				init = random.choices(pool,weights=[9,9,1,1,1,1])
				
				bi = [[0,init[0]],[0,init[0]],[0.1,init[0]]]
				unpacked[i],accpt,test = mcmc.mh(len_mcmc,init=init[0],test=data[item],dim=dim,testif=1,bias=bi[cyc])
				# Tous les noeuds sont à égalité,
				init = random.choice(pool)
				packed[i],accpt,test = mcmc.mh(len_mcmc,init=init,test=data[item],dim=dim,testif=1)

			deb = 0
			j = 0
			p_packed = [0 for x in range(pas,len_mcmc+1,pas)]
			p_unpacked = [0 for x in range(pas,len_mcmc+1,pas)]
			for i in range(pas,len_mcmc+1,pas):
				raw_packed = [packed[k][0:i] for k in range(nb_mcmc)]
				raw_unpacked = [unpacked[k][0:i] for k in range(nb_mcmc)]
				p_packed[j] = mcmc.pvalue(raw_packed,pool)
				p_unpacked[j] = mcmc.pvalue(raw_unpacked,pool)
				j+=1
				deb = i

			pval = [sum(p_packed[x])-sum(p_unpacked[x]) for x in range(len(p_packed))]
			plt.plot(pval)
			plt.show()

