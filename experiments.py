
import time

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
#
from loop import LoOP
from hics import hics
from scipy.spatial.distance import cityblock, euclidean, braycurtis, chebyshev, jaccard
from sklearn import manifold
from scipy.io import arff
from sklearn.metrics import roc_curve
from mpi4py import MPI



#### GENERAL SETTINGS  ##########################################################




availablemarkers = np.array([".",",","o","v","^"	,"<",">","1","2","3","4","8","s","p","*","h","H","+","x","D","d","|","_"])
colors = ["#000000","#FF0000","#0FF000","#00FF00","#000FF0"	,"#0000FF","#F0000F","#FFF000","#0FFF00","#00FFF0","#000FFF","#F000FF","#FF00FF","#F0FF0F","#888888","#880000","#008800","#000088","#888800","#008888","#880088","#440044","#440000"]
colors = np.array(colors)
availablemarkers= np.array(availablemarkers)

def generate(nClusters=3,dimentions=140,nOutliers=40,nPoints=800, rrange=6,seed=9, showNeighbors=True, plot=False):
	clusters = []
	allData = []
	clusterLabels = []
	np.random.seed(seed)
	PointsLeft = nPoints
	cluster_starts = []
	for c in range(nClusters):
		c_starts = []
		
		c_size = np.random.randint(1,max(2,PointsLeft-(nClusters-c)*(nPoints/nClusters)/2 ))
		if (c==nClusters-1):
			c_size = PointsLeft
		PointsLeft -= c_size
		c_data = np.random.rand(c_size,dimentions)
		for d in range(dimentions):
			new_start = np.random.randint(0,rrange)
			c_starts.append(new_start)

			c_data[:,d] += c_starts[d]
		cluster_starts.append(c_starts)
		clusters.append(c_data)
		for d in c_data:
			allData.append(d)
			clusterLabels.append(c)
		c_data = np.array(c_data)
		print c_data.shape
	clusters = np.array(clusters)
	#plt.show()
	allData = np.array(allData)

	outliers = []
	for o in range(nOutliers):
		#create outliers by mixing clusters
		#get random sample
		ri = np.random.randint(0,len(allData))
		ri_c = clusterLabels[ri]
		original_start = cluster_starts[ri_c]
		#transform the point "weird" in one of the dimentions
		ri_d = np.random.randint(0,dimentions/2)

		allData[ri,ri_d*2] -= original_start[ri_d*2]
		allData[ri,ri_d*2+1] -= original_start[ri_d*2+1]

		#transform the point differently
		rand_c = ri_c
		while rand_c == ri_c:
			rand_c = np.random.randint(0,nClusters)
		allData[ri,ri_d*2] += np.array(cluster_starts)[rand_c,ri_d*2]
		allData[ri,ri_d*2+1] += np.array(cluster_starts)[rand_c,ri_d*2+1]
		outliers.append(ri)

	if (showNeighbors):
		comb_loop = LoOP(allData,normalize=False, distance_function=euclidean)
		comb_loop_values = comb_loop.local_outlier_probabilities(verbose=False,feature_end=-1)
		#comb_loop_values = np.array(comb_loop_values)
		neighbours = comb_loop.get_neighbours()
		instances = comb_loop.get_instances()
		instancesnp = np.array(instances)
	else:
		instancesnp = np.array(allData)

	if (plot):
	
		#amax = np.amax(adist)
		#adist /= amax
		mds = manifold.MDS(n_components=2, dissimilarity="euclidean", random_state=seed,n_init=10)
		results = mds.fit(instancesnp)
		coords = results.embedding_
		plt.figure(figsize=[20,15])
		plt.subplot((dimentions/2+1)/2,2,1)
		plt.title("Global Space")
		clusterLabels = np.array(clusterLabels)
		cols = colors[clusterLabels]
		if (showNeighbors):
			cols = 'black';
		plt.scatter(coords[:,0],coords[:,1], c=cols,alpha=1.,linewidth=0.,s=30)
		outlier_colors = ['r','b','g','y']
		
		if (showNeighbors):
			oc = 0
			for o in outliers:
				color = outlier_colors[oc]
				oc += 1
				S = neighbours[instances[o]]
				S_index = []
				for s in S:
					S_index.append(list(instances).index(s))
				S_index = np.array(S_index)
				#print S_index
				plt.scatter(coords[S_index,0],coords[S_index,1],alpha=0.5,s=60, c=color, marker="o")
		oc = 0
		for o in outliers:
			color = outlier_colors[oc]
			oc += 1
			plt.scatter(coords[o,0],coords[o,1],alpha=1.,s=180, c=color, marker="*")
		
		for d in range(dimentions/2):
			
			plt.subplot((dimentions/2+1)/2,2,d+2)
			plt.title("Dimention Space "+str(d+1))
			
			mds = manifold.MDS(n_components=2, dissimilarity="euclidean", random_state=seed,n_init=10)
			results = mds.fit(instancesnp[:,d*2:d*2+2])
			coords = results.embedding_
			
			cols = colors[clusterLabels]
			if (showNeighbors):
				cols = 'black';

			plt.scatter(coords[:,0],coords[:,1],alpha=1.,linewidth=0.,s=30, c=cols)
			if (showNeighbors):
				oc = 0
				for o in outliers:
					S = neighbours[instances[o]]
					S_index = []
					for s in S:
						S_index.append(list(instances).index(s))
					S_index = np.array(S_index)
					color = outlier_colors[oc]
					oc += 1
					plt.scatter(coords[S_index,0],coords[S_index,1],alpha=0.5,s=60, c=color, marker="o")
			oc = 0
			for o in outliers:
				color = outlier_colors[oc]
				oc += 1
				plt.scatter(coords[o,0],coords[o,1],alpha=1.,s=180, c=color, marker="*")
		plt.savefig("img/example"+str(seed)+".png")
	return allData, outliers

from time import time
def syntheticData():
	configuration = []
	size = MPI.COMM_WORLD.Get_size() 
	rank = MPI.COMM_WORLD.Get_rank() 
	name = MPI.Get_processor_name()
	comm = MPI.COMM_WORLD
	rs = [2,3,4,5,10]
	r = rs[rank]
	#dims = [10,20,50,100,200,400]
	#dim = dims[rank]
	for dim in [10,20,50,100,200,400]:
		for c in [2,3,5]:
			configuration.append([c,dim,50,1000,r])

	conf_teller = 0
	for conf in configuration:
		print "conf",conf_teller

		dim = conf[1]
		d,outliers = generate(nClusters=conf[0],dimentions=conf[1],nOutliers=conf[2],nPoints=conf[3], rrange=conf[4] ,seed=42, showNeighbors=False, plot=False)
		y_true = np.zeros(len(d))
		y_true[outliers] = 1

		dataset = np.c_[d,y_true]
		print "loaded data"
		HCISauc, LOFauc, hics_scores, lof_scores, hics_time, lof_time = hics(dataset)

		fpr_hics,tpr_hics, th = roc_curve(y_true, hics_scores)
		fpr_lof,tpr_lof, th = roc_curve(y_true, lof_scores)

		print "Done hics and lof"
		#run LoOP
		#
		t0 = time()
		loop = LoOP(d)
		loop_scores = loop.local_outlier_probabilities(verbose=False,feature_end=-1)
		loop_time = time() - t0
		loop_scores = np.array(loop_scores)
		fpr_loop,tpr_loop, th = roc_curve(y_true, loop_scores)

		#train Soup
		t0 = time()
		soup = LoOP(d)
		soup_initial = soup.local_outlier_probabilities(verbose=False,feature_end=-1)
		soup_time = time() - t0

		#run LoOP on every feature window
		loop_window_scores = []
		t0 = time()
		for di_window in range(dim/2):
			loop = LoOP(d[:,di_window*2:di_window*2+2])
			loop_scores2 = loop.local_outlier_probabilities(verbose=False,feature_end=-1)
			loop_window_scores.append(loop_scores2)
		loop_local_time = time() - t0
		loop_window_scores = np.array(loop_window_scores)
		local_loop_scores = loop_window_scores.max(axis=0)
		#print local_loop_scores.shape
		fpr_lloop,tpr_lloop, th = roc_curve(y_true, local_loop_scores)
		print "Done Local LoOP"

		soup_window_scores = []
		t0 = time()
		for di_window in range(dim/2):
			soup_scores = soup.local_outlier_search(feature_start=di_window*2,feature_end=di_window*2+2)
			soup_window_scores.append(soup_scores)
		soup_time += time() - t0
		#soup_window_scores.append(soup_initial)
		soup_window_scores = np.array(soup_window_scores)

		soup_scores = soup_window_scores.max(axis=0)
		fpr_soup,tpr_soup, th = roc_curve(y_true, soup_scores)
		#print soup_scores.shape
		#print "Done Soup"

		print "Times:", loop_time, loop_local_time, soup_time,  hics_time, lof_time

		#plot
		#Save all the results!
		#
		to_save = []
		hics_scores = np.array(hics_scores)[:,0]
		lof_scores = np.array(lof_scores)[:,0]
		print y_true.shape , hics_scores.shape, lof_scores.shape, loop_scores.shape, local_loop_scores.shape, soup_scores.shape
		to_save.append(y_true)
		to_save.append(hics_scores)
		to_save.append(lof_scores)
		to_save.append(loop_scores)
		to_save.append(local_loop_scores)
		to_save.append(soup_scores)
		
		times = np.array([loop_time, loop_local_time, soup_time, hics_time, lof_time])
		to_save = np.array(to_save)
		np.save("roc_conf_scores"+str(r)+"_"+str(conf_teller)+".npy",to_save)
		np.save("roc_conf_times"+str(r)+"_"+str(conf_teller)+".npy",times)
		
		#plot ROC curves
		plt.figure()
		plt.plot(fpr_loop, tpr_loop, label='LoOP')
		plt.plot(fpr_lof, tpr_lof, label='LOF')
		plt.plot(fpr_hics, tpr_hics, label='HiCS')
		plt.plot(fpr_soup, tpr_soup, label='GLocal')
		plt.plot(fpr_lloop, tpr_lloop, label='LoOP local')
		plt.plot([0, 1], [0, 1], 'k--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('ROC')
		plt.legend(loc="lower right")
		plt.savefig("img/roc_r"+str(r)+"-"+str(conf_teller)+".png")
		conf_teller += 1
		plt.clf()
	exit()

def existingData(rank=0, k=10): #'ann_thyroid.arff','Ann Thyroid',
	datasets = ['ann_thyroid.arff','arrhythmia.arff','diabetes.arff','glass.arff','ionosphere.arff','pendigits16.arff', 'breast.arff', 'breast_diagnostic.arff']
	names = ['Ann Thyroid','Arrhythmia','Diabetes','Glass','Ionosphere','Pen Digits 16', 'Breast', 'Breast Diag.']
	size = MPI.COMM_WORLD.Get_size() 
	#rank = MPI.COMM_WORLD.Get_rank() 
	name = MPI.Get_processor_name()
	comm = MPI.COMM_WORLD
	ds = datasets[rank]
	name = names[rank]
	conf_teller = rank
	print "conf",conf_teller
	d, meta = arff.loadarff("datasets/"+ds)
	d = np.asarray(d.tolist(), dtype=np.float32)
	print d.shape
	
	y_true = d[:,-1]
	y_true = np.array(y_true)
	d = d[:,:-1] #remove last column

	dim = len(d[0])
	dataset = np.c_[d,y_true]
	print "loaded data"
	HCISauc, LOFauc, hics_scores, lof_scores, hics_time, lof_time = hics(dataset,k)

	fpr_hics,tpr_hics, th = roc_curve(y_true, hics_scores)
	fpr_lof,tpr_lof, th = roc_curve(y_true, lof_scores)

	print "Done hics and lof"
	#run LoOP
	#
	t0 = time()
	loop = LoOP(d)
	loop_scores = loop.local_outlier_probabilities(verbose=False,feature_end=-1,k=k)
	loop_time = time() - t0
	loop_scores = np.array(loop_scores)
	fpr_loop,tpr_loop, th = roc_curve(y_true, loop_scores)

	#train Soup
	t0 = time()
	soup = LoOP(d)
	soup_initial = soup.local_outlier_probabilities(verbose=False,feature_end=-1,k=k)
	soup_time = time() - t0

	#run LoOP on every feature window
	loop_window_scores = []
	t0 = time()
	for di_window in range(dim-1):
		loop = LoOP(d[:,di_window:di_window+2])
		loop_scores2 = loop.local_outlier_probabilities(verbose=False,feature_end=-1,k=k)
		loop_window_scores.append(loop_scores2)
	loop_local_time = time() - t0
	loop_window_scores = np.array(loop_window_scores)
	local_loop_scores = loop_window_scores.max(axis=0)
	#print local_loop_scores.shape
	fpr_lloop,tpr_lloop, th = roc_curve(y_true, local_loop_scores)
	print "Done Local LoOP"

	soup_window_scores = []
	t0 = time()
	for di_window in range(dim-1):
		soup_scores = soup.local_outlier_search(feature_start=di_window,feature_end=di_window+2,k=k)
		soup_window_scores.append(soup_scores)
	soup_time += time() - t0
	soup_window_scores.append(soup_initial)
	soup_window_scores = np.array(soup_window_scores)

	soup_scores = soup_window_scores.max(axis=0)
	fpr_soup,tpr_soup, th = roc_curve(y_true, soup_scores)
	#print soup_scores.shape
	#print "Done Soup"
	print "Times:", loop_time, loop_local_time, soup_time,  hics_time, lof_time

	#plot
	#
	#Save all the results!
	#
	to_save = []
	hics_scores = np.array(hics_scores)[:,0]
	lof_scores = np.array(lof_scores)[:,0]
	print y_true.shape , hics_scores.shape, lof_scores.shape, loop_scores.shape, local_loop_scores.shape, soup_scores.shape
	to_save.append(y_true)
	to_save.append(hics_scores)
	to_save.append(lof_scores)
	to_save.append(loop_scores)
	to_save.append(local_loop_scores)
	to_save.append(soup_scores)
	
	times = np.array([loop_time, loop_local_time, soup_time, hics_time, lof_time])
	

	to_save = np.array(to_save)
	np.save("roc_conf_scores"+str(ds)+"_"+str(conf_teller)+".npy",to_save)
	np.save("roc_conf_times"+str(ds)+"_"+str(conf_teller)+".npy",times)
	
	plt.figure()
	plt.plot(fpr_loop, tpr_loop, label='LoOP')
	plt.plot(fpr_lof, tpr_lof, label='LOF')
	plt.plot(fpr_hics, tpr_hics, label='HiCS')
	plt.plot(fpr_soup, tpr_soup, label='GLoss')
	plt.plot(fpr_lloop, tpr_lloop, label='LoOP local')
	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	#plt.title('ROC curve '+str(name))
	plt.legend(loc="lower right")
	plt.savefig("img/roc_r"+str(ds)+"-"+str(conf_teller)+".png", bbox_inches='tight')
	conf_teller += 1
	plt.clf()
		


#run either existing Data experiments or synthetic 
#exsting :
existingData(0)
existingData(1)
existingData(2)
existingData(3)
existingData(4)
existingData(5)
existingData(6)
existingData(7)

#synthetic:
#syntheticData()



