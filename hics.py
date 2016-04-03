import matplotlib.pyplot as plt
plt.style.use('ggplot')
from scipy.io import arff
import pandas as pd
from scipy import stats
import itertools
import random
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics

#We use arff from scipy.io to read the file
#Download the data from here http://ipd.kit.edu/~muellere/HiCS/
#


#Below functions are used to calculate LOF
def knn(df,k):
	nbrs = NearestNeighbors(n_neighbors=k)
	nbrs.fit(df)
	distances, indices = nbrs.kneighbors(df)
	return distances, indices

def reachDist(df,MinPts,knnDist):
	nbrs = NearestNeighbors(n_neighbors=MinPts)
	nbrs.fit(df)
	distancesMinPts, indicesMinPts = nbrs.kneighbors(df)
	distancesMinPts[:,0] = np.amax(distancesMinPts,axis=1)
	distancesMinPts[:,1] = np.amax(distancesMinPts,axis=1)
	distancesMinPts[:,2] = np.amax(distancesMinPts,axis=1)
	return distancesMinPts, indicesMinPts

def ird(MinPts,knnDistMinPts):
 return (MinPts/np.sum(knnDistMinPts,axis=1))

def lof(Ird,MinPts,dsts):
	lof=[]
	for item in dsts:
		tempIrd = np.divide(Ird[item[1:]],Ird[item[0]])
		lof.append(tempIrd.sum()/MinPts)
	return lof

#Below function is used to create possible subspaces in each step
def comboGenerator(startPoint,space,n):
	combosFinal=[]
	for item in itertools.combinations(list(set(space)-set(startPoint)),(n-len(startPoint))):
		combosFinal.append(sorted(startPoint+list(item)))
	return combosFinal

from time import time

#input is data np array with last column 0s and 1s
def hics(data,nk=20):
	#data, meta = arff.loadarff('ann_thyroid.arff')
	#data = np.array(data)
	#
	datan = len(data)
	datam = len(data[0])
	
	#print data.shape
	
	df = pd.DataFrame(data)
	#print df.columns[:-1]

	#calculate the index, we use this for selecting random sections in subspace
	index_df = (df.rank()/df.rank().max()).iloc[:,:-1]

	#We calculate AUC score from simple LOF
	t0 = time()
	m=20
	knndist, knnindices = knn(df.iloc[:,:-1],nk)
	reachdist, reachindices = reachDist(df.iloc[:,:-1],m,knndist)
	irdMatrix = ird(m,reachdist)
	lofScores = lof(irdMatrix,m,reachindices)
	ss=MinMaxScaler().fit_transform(np.array(lofScores).reshape(-1,1))
	#print "LOF AUC Score"
	LOFauc = metrics.roc_auc_score(pd.to_numeric(df[df.columns[-1]].values),ss)
	LOFtime = time()-t0

	t0 = time()

	#We start with 2-D subspaces
	listOfCombos = comboGenerator([],df.columns[:-1],2)
	testedCombos=[]
	selection=[]

	#We calculate the contrast score for each subspace
	#For each subspace that satisfies the cut_off point criteria
	#We add additional dimensions
	maxitem = max(100,datam*4)

	while(len(listOfCombos)>0 and len(selection) < maxitem):
		if (time() - t0 > 7200):
			return 0, LOFauc, np.zeros(datan), ss, 0, LOFtime
		if listOfCombos[0] not in testedCombos:
			alpha1 = pow(0.2,(float(1)/float(len(listOfCombos[0]))))
			pvalue_Total =0
			pvalue_cnt = 0
			avg_pvalue=0
		for i in range(0,50):
			lband = random.random()
			uband = lband+alpha1
		#print listOfCombos
		v = random.randint(0,(len(listOfCombos[0])-1))
		rest = list(set(listOfCombos[0])-set([listOfCombos[0][v]]))
		k=stats.ks_2samp(df[listOfCombos[0][v]].values, df[((index_df[rest]<uband) & (index_df[rest]>lband)).all(axis=1)][listOfCombos[0][v]].values)
		if not(np.isnan(k.pvalue)):
			pvalue_Total = pvalue_Total+k.pvalue
			pvalue_cnt = pvalue_cnt+1
		if pvalue_cnt>0:
			avg_pvalue = pvalue_Total/pvalue_cnt
		if (1.0-avg_pvalue)>0.6:
			selection.append(listOfCombos[0])
			listOfCombos = listOfCombos + comboGenerator(listOfCombos[0],df.columns[:-1],(len(listOfCombos[0])+1))
			testedCombos.append(listOfCombos[0])
			listOfCombos.pop(0)
			listOfCombos = [list(t) for t in set(map(tuple,listOfCombos))]
		else:
			listOfCombos.pop(0)

	#We calculate the contrast score 50 times for each subspace
	#We average the contrast scores from iterations

	scoresList=[]
	#maxitem = datan*10
	curitem = 0
	for item in selection:
		curitem += 1
		if (curitem > maxitem):
			break
		if (time() - t0 > 7200):
			return 0, LOFauc, np.zeros(datan), ss, 0, LOFtime
		m=20
		knndist, knnindices = knn(df[item],nk)
		reachdist, reachindices = reachDist(df[item],m,knndist)
		irdMatrix = ird(m,reachdist)
		lofScores = lof(irdMatrix,m,reachindices)
		scoresList.append(lofScores)

	#We calculate average LOF score for each data point from each subspace

	avgs = np.nanmean(np.ma.masked_invalid(np.array(scoresList)),axis=0)

	#We scale the results to 0,1 range
	scaled_avgs = MinMaxScaler().fit_transform(avgs.reshape(-1,1))

	hicstime = time()-t0



	#Here is the AUC score from HiCS
	#print "HCiS AUC Score"
	HCISauc = metrics.roc_auc_score(pd.to_numeric(df[df.columns[-1]].values),scaled_avgs)

	
	return HCISauc, LOFauc, scaled_avgs, ss, hicstime, LOFtime
#hics(0)