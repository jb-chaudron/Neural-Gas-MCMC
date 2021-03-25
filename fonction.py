import math
import numpy as np
import pandas as pd 
import random
import itertools as itr
import os
from pathlib import PurePath
from scipy.sparse import coo_matrix
from scipy.spatial import distance_matrix
import scipy.stats as sts 
from sklearn.preprocessing import normalize
import time
import functools
import operator
import collections
import csv


class Cluster():
	"""
		Classe pour le Neural Gas SOM
	"""
	def __init__(self):
		self.data = 0
		self.obj = 0
		self.N = 0
		self.limi = 2
		self.limf = 6
		self.tmax = 0
		self.ei = 0.3
		self.ef = 0.05
		self.li = 30
		self.lf = 0.01

		
	def accueil(self):
		choix_set = -1
		choix_methode = -1
		condition = 0
		W = 0
		C = 0
		while not choix_methode in [0,1,2]:
			choix_methode = input("Voulez vous utiliser un de vos jeux de données ou bien un préexistant ?\n \
				   0) Votre jeu de données\n \
				   1) Un jeu de données préexistant sans recalculer les poids des Clusters\n \
				   2) Un jeu de données préexistant en recalculant les poids des Clusters \n")
			choix_methode = int(choix_methode)
		if choix_methode in [1,2]:
			while not choix_set in [1,2]:
				choix_set = input("\n\
				   1) Jeu de données 1 sur les comportements d'utilisateur de Reddit \n \
				   2) Jeu de données 2 sur la fréquence de vente de certains produits\n")
				choix_set = int(choix_set)

		if choix_methode == 0:
			path = input("Entrez le chemin vers votre data set, ce doit être un csv ! attention la première colonne disparaît !!\n")
			while not os.path.exists(path):
				path = input("Veuillez rentrer un chemin valide \n")
			self.data = pd.read_csv(path)
			self.obj = 'df'
		elif choix_methode == 1:
			if choix_set == 1:
				with open("reddit_W.csv",newline='') as file:
					rid = csv.reader(file,delimiter=',')
					W = list(rid)
				W = [[float(x) for x in W[i]] for i in range(len(W))]
				C = [[20], [25], [24, 29, 18, 14], [13], [0], [25, 16, 2], [20], [8, 27], [7], [27, 25], [27, 1, 25], [5, 1, 25], [25, 5, 1], [3, 15, 25], [29], [18, 25], [25, 5, 12], [12, 5, 25], [15, 25, 11], [5], [6], [0, 20], [3], [26], [12], [9], [23], [15], [23, 6], []]
				path = 'reddit_raw.csv'
				self.data = pd.read_csv(path)
				self.obj = 'coo'
			else:
				with open("transaction_W.csv",newline='') as file:
					rid = csv.reader(file,delimiter=',')
					W = list(rid)
				W = [[float(x) for x in W[i]] for i in range(len(W))]
				C = [[10, 20, 2, 22], [16], [4, 8, 0, 21], [14, 27], [2, 10], [19, 16, 29], [4, 28, 10, 13, 25], [13, 23], [20, 2, 18], [1, 14, 24, 3, 19], [18, 6], [24, 12, 15, 29], [11, 5, 26], [10, 23], [9], [23, 11, 24, 1], [26, 24, 27, 1], [25, 4, 13], [4, 21, 22, 10], [27], [28, 21, 18], [2, 22, 20], [21, 8, 18], [24, 25, 6, 3], [23, 15, 26], [10, 8, 17], [15, 1, 24], [5, 29], [10, 6, 13], [5, 16, 12]]
				path = 'transaction_raw.csv'
				self.data = pd.read_csv("transaction_raw.csv",usecols=[x for x in range(55,107)])
				
				self.obj = 'df'
		elif choix_set == 1:
			path = 'reddit_raw.csv'
			self.data = pd.read_csv(path)
			self.obj = 'coo'
		elif choix_set == 2:
			path = 'transaction_raw.csv'
			self.data = pd.read_csv("transaction_raw.csv",usecols=[x for x in range(55,107)])
			self.obj = 'df'


		return(choix_methode,W,C)

	def prétraitement(self):
		d = self.data
		if self.obj == 'df':
			h = pd.DataFrame(columns=[x for x in d.columns if x != d.columns[0]],index=[x for x in d.index])
			for i,j in itr.product(d.index,h.columns):
				h.loc[i,j] = (d.loc[i,j] - d.loc[:,j].mean())/d.loc[:,j].std()
				
			f = [[x for x in h.iloc[i,:]] for i in range(len(h.index))]
			self.N = 30
			self.tmax = self.N*self.limf*4
			self.data=f
		elif self.obj == 'coo':
			d.sort_values(d.columns[0], axis=0, ascending=True, inplace=True)
			a = pd.factorize(d.iloc[:,0])
			d.iloc[:,0] = [x for x in a[0]]
			d.sort_values(d.columns[1],axis=0,ascending=True,inplace=True)
			b = pd.factorize(d.iloc[:,1])
			d.iloc[:,1] = [x for x in b[0]]

			row = [x for x in d.iloc[:,0]]
			col = [x for x in d.iloc[:,1]]
			row.sort()
			col.sort()
			row = pd.factorize(row)
			col = pd.factorize(col)

			l_row = len(set(d.iloc[:,0]))
			l_col = len(set(d.iloc[:,1]))

			coo = coo_matrix((d.iloc[:,2],(d.iloc[:,0],d.iloc[:,1])),shape=(l_row,l_col)).toarray()
			normalize(coo,axis=1,copy=False)
		
			self.N = 30
			self.tmax = self.N*self.limf*4
			self.data = coo
		return(self.data)


	def distance(self,v,W):
		dis = distance_matrix(W,[v],p=2,threshold = 1000)
		dis = [(dis[i][0],i) for i in range(len(dis))]
		dis.sort(key=lambda x : x[0], reverse=False)
		return([x for x in dis])


	def som(self):
		un=time.time()
		cycle1 = 0
		"""
			N : le nombre de neurones à mettre
			e : le pas d'apprentissage
			tmax : le nombre d'itération
			lim : le temps maximum sans que deux noeuds soient proche avant la perte de la connexion entre eux
			path : le chemin vers le jeu de données 
		"""
		d = self.data
		
		# Initialisation de l'algorithme
		"""
		if self.obj == 'df':
			W = [[np.random.normal(0,1) for i in range(len(d.columns))] for j in range(self.N)]
		else :
		"""
		W = [[np.random.normal(0,1) for i in range(len(d[0]))] for j in range(self.N)]
		# Ces deux listes prendront l'une le numéro d'un noeud l'autre le temps sans visite
		C = [[] for i in range(self.N)]
		t = [[] for i in range(self.N)]


		for ep in range(self.tmax):
			#Extraction du candidat

			v = [x for x in d[random.randint(0,len(d)-1)]]
			"""
				Calcule des distances et classement du plus proche au plus loin
				Un peu d'explication, dis[i] contient (la distance, le numéro du noeud)
				dis[i][1] contient donc le numéro du noeud, en appelant W[dis[i][1]] on appelle
				le noeud correspondant.
				"i" quant à lui est le classement du noeud car on a trier la liste dans l'ordre déscendant
			"""
			dis = self.distance(v,W)
			d0 = dis[0][1]
			d1 = dis[1][1]
			eta = self.ei * math.pow((self.ef/self.ei),(ep/self.tmax))
			lamb = self.li * math.pow((self.lf/self.li),(ep/self.tmax))
			lim = self.limi * math.pow((self.limf/self.limi),(ep/self.tmax))
			for i in range(len(dis)):
				f = eta*math.exp(-i/lamb)
				W[dis[i][1]] = [W[dis[i][1]][j] + f*(v[j] -W[dis[i][1]][j]) for j in range(len(W[dis[i][1]]))]

			ret = [i for i in range(len(C[d0])) if (t[d0][i]+1 >= lim and C[d0][i] != d1)]
		
			C[d0] = [C[d0][i] for i in range(len(C[d0])) if not i in ret]
			t[d0] = [t[d0][i] for i in range(len(t[d0])) if not i in ret]
			if d1 in C[d0]:
				t[d0] = [t[d0][i]+1 if C[d0] != d1 else 0 for i in range(len(t[d0]))]
			else:
				C[d0] += [d1]
				t[d0] = [t[d0][i]+1 for i in range(len(t[d0]))] + [0]
			print((time.time()-un)*self.tmax/3600)
			

		return(W,C)



class MCMC():	
	"""
		Classe pour la Markov Chain Monte Carlo
		C : Contient les liens qui ont été établit lors du clusering par neural gas
		W : Contient les poids des neurones du neural gas
		test : Contient l'élément que l'on va tenter de découvrir
		biais : 
			Initialement le poid que l'on veut attribuer à l'ensemble des neurones en lien avec un neurone
				Par exemple, si on met 0.8, l'ensemble des neurones liés à un neurones se partagerons 80% des probabilité d'être tirée à partir de ce noed
			Ensuite contient un dataframe contenant la probabilité de transition d'un noeud à un autre
	"""
	def __init__(self,C,W,biais,test,cov):
		self.C = C
		self.cov = cov
		self.W = W
		self.test = test
		self.nb_clust = 0
		self.biais = biais
		self.prior = 0

	def prétraitement(self):

		self.nb_clust = len(self.C)

		C = [set(x) for x in self.C]

		for i in range(len(self.C)):
			for j in self.C[i]:
				C[j].add((x for x in self.C[i]))

		df = pd.DataFrame(columns=[x for x in range(self.nb_clust)],index=[x for x in range(self.nb_clust)])
		for i,j in itr.product(df.index,df.columns):
			l = len(C[i])
			if i == j :
				df.loc[i,j] = 0
			elif j in C[i]:
				df.loc[i,j] = self.biais/l 
			else:
				df.loc[i,j] = (1-self.biais)/(self.nb_clust-l)

		tot = sum([sum(df.loc[i:]) for i in df.index])
		self.prior = [sum(df.loc[i:])/tot for i in df.index]
		self.var = [sorted(x)[math.floor(0.75*len(x))] for x in self.cov]
		self.biais = df
		print(self.prior)

	def alpha(self,cand,al_now,now,dim):
		#Donne la vraisemblance des données sachant une loi normale centrée sur chaque 
		al_cand = sts.norm.pdf(np.linalg.norm([self.test[i]-self.W[cand][i] for i in dim]),loc=0,scale=self.var[cand])
		#On multiplie par la probabilité à priori et la probabilité de 
		al_cand *= self.prior[cand]
		

		
		alpha = al_cand*self.biais.iloc[cand,now]/(al_now*self.biais.iloc[now,cand])		
		print("alpha = ",alpha)
		return(alpha,al_cand)

		

	def mh(self,n_iter,init,test,dim=None,testif=None,bias=None):
		if testif == None:
			pass
		else:
			self.test = test

		if dim == None:
			dim = range(len(self.test))
		else:
			dim = dim

		if bias == None:
			pass
		else:
			self.prior[bias[1]] += bias[0]

		out = [0 for x in range(n_iter)]
		accpt = 0

		now = init

		al_now = sts.norm.pdf(np.linalg.norm([self.test[i]-self.W[now][i] for i in dim]),loc=0,scale=self.var[now])
		#al_now = np.prod([sts.norm.pdf(self.test[i],loc=self.W[now][i],scale=1) for i in range(len(self.test))])
		al_now *= self.prior[now]

		for i in range(n_iter):
			cand = random.choices([x for x in range(self.nb_clust)],weights=[x for x in self.biais.iloc[now]])
			cand = cand[0]
			alpha,al_cand = self.alpha(cand,al_now,now,dim)

			u = np.random.uniform(low=0.0,high=1.0)
			if u < alpha:
				now = cand
				al_now = al_cand
				accpt = accpt + 1

			out[i] = now
		
		return(out,accpt/n_iter,self.test)

	def pvalue(self,entrée,dim):
		tot = len(entrée[0])
		out = [sum([1/tot for j in entrée[i] if j in dim]) for i in range(len(entrée))]
		"""
		out = {j:[1/tot ] for i,j in itr.product(range(len(entrée)),dim)}
		out = [entrée[i][j] for i,j in itr.product(range(len(entrée)),range(len(entrée[0])))]
		tot = len(out)
		out = [1/tot for i in out if i in dim]

		out = sum(out)
		print(out)
		"""		
		if dim == None:		
			return(out)
		else:
			return(out)
"""
	def permutation(self,a,b,pas,dim):

		echant = [0 for x in range(math.floor(len(a[0])/pas))]
		test = [0 for x in range(len(echant)-1)]
		deb = 0

		for i in range(0,math.floor(len(a[0])/pas)-1):
			j = (i+1)*pas
			deb *= pas
			echant[i] = [a[k][deb:j]+b[k][deb:j] for k in range(len(a))]
			test[i] = [[random.choice(echant[i])[0] for k in range(deb,j)] for l in range(1000)]
			deb = i
		print(test)
		mcval = [self.pvalue(test[i],dim) for i in range(len(test))]
		print(mcval)
		print(len(mcval))
		print(len(mcval[0]))
		mcval = [[mcval[i][j] for i in range(len(mcval))] for j in range(1000)]
		ran = [(random.choice([x for x in range(1000)]),random.choice([x for x in range(1000)])) for u in range(500)]
		print(len(mcval[0]))
		mcval = [[mcval[u[0]][v] - mcval[u[1]][v] for v in range(len(mcval[0]))] for u in ran]
		mcval = [[1 if (sts.wilcoxon(mcval[i][j:j+25],zero_method="pratt")[1] <0.05) else 0 for j in range(0,len(mcval[i]),25)] for i in range(len(mcval))]
		#test = [[1 if (((sts.wilcoxon(test[i][j],zero_method="pratt")[1]/2) <0.05) and sum(test[i][j]) < 0) else 0 for j in range(len(test[i]))] for i in range(len(test))]
		out = [[] for i in range(len(mcval))]

		for i in range(len(mcval)):
			lon = 0
			for j in range(len(mcval[i])):
				if mcval[i][j] == 1:
					lon+=1
				else:
					out[i] += [lon]
					lon = 0
			out[i] += [lon]

		print(len(mcval))		
		pval = functools.reduce(operator.concat, out)
		pval = max(pval)
		print(pval)
		return(pval)

"""




		

