
import numpy as np
from matplotlib import pyplot as plt

from sklearn import metrics


#### GENERAL SETTINGS  ##########################################################




availablemarkers = np.array([".",",","o","v","^"	,"<",">","1","2","3","4","8","s","p","*","h","H","+","x","D","d","|","_"])
colors = ["#000000","#FF0000","#0FF000","#00FF00","#000FF0"	,"#0000FF","#F0000F","#FFF000","#0FFF00","#00FFF0","#000FFF","#F000FF","#FF00FF","#F0FF0F","#888888","#880000","#008800","#000088","#888800","#008888","#880088","#440044","#440000"]
colors = np.array(colors)
availablemarkers= np.array(availablemarkers)



#dims = [10,20,50,100,200,400]
#dim = dims[rank]

allresults = []

def avg(l):
	return sum(l) / float(len(l))
	
			
'''
to_save.append(y_true)
to_save.append(hics_scores)
to_save.append(lof_scores)
to_save.append(loop_scores)
to_save.append(local_loop_scores)
to_save.append(soup_scores)
times = np.array([loop_time, loop_local_time, soup_time, hics_time, lof_time])
'''

#load all the 10d results, those are teller=0,1,2
#20d = 3,4,5
#50d = 6,7,8
#100d = 9,10,11
#200d = 12,13,14
#400d = 15,16,17
#
lenr = 15
#calculate auc for each experiment
#
allresults = {}
alltime = {}

for r in [2,3,4,5,10]:
	teller = 0
	for d in  [10,20,50,100,200,400]: #,50,100,200,400
		if (d not in allresults.keys()):
			allresults[d] = []
			alltime[d] = []
		for c in range(3):
			allresults[d].append(np.load("roc_conf_scores"+str(r)+"_"+str(teller)+".npy"))
			alltime[d].append(np.load("roc_conf_times"+str(r)+"_"+str(teller)+".npy"))
			teller += 1


print "{\\#D\\ } & {\\ HiCS\\ } & {\\ LOF\\ } & {\\ LoOP\\ } & {Local LoOP} & {\\ \\algName{}\\ } \\\\"
for d in  [10,20,50,100,200,400]: #,50,100,200,400
	aucs_hics = []
	aucs_lof = []
	aucs_loop = []
	aucs_lloop = []
	aucs_soup = []
	allscores = np.array(allresults[d])
	for scores in allscores:
		aucs_hics.append( metrics.roc_auc_score(scores[0],scores[1]))
		aucs_lof.append( metrics.roc_auc_score(scores[0],scores[2]))
		aucs_loop.append( metrics.roc_auc_score(scores[0],scores[3]))
		aucs_lloop.append( metrics.roc_auc_score(scores[0],scores[4]))
		aucs_soup.append( metrics.roc_auc_score(scores[0],scores[5]))
	print d,"&", avg(aucs_hics),"&", avg(aucs_lof),"&", avg(aucs_loop),"&", avg(aucs_lloop),"&", avg(aucs_soup) ,"\\\\"

print "time"

print "{\\#D\\ } & {\\ HiCS\\ } & {\\ LOF\\ } & {\\ LoOP\\ } & {Local LoOP} & {\\ \\algName{}\\ } \\\\"
for d in  [10,20,50,100,200,400]: #,50,100,200,400
	times_hics = []
	times_lof = []
	times_loop = []
	times_lloop = []
	times_soup = []
	alltimes = np.array(alltime[d])
	for t in alltimes:
		#np.array([loop_time, loop_local_time, soup_time, hics_time, lof_time])
		times_hics.append( t[3] )
		times_lof.append( t[4] )
		times_loop.append( t[0] )
		times_lloop.append( t[1] )
		times_soup.append( t[2] )
	print d,"&", avg(times_hics),"&", avg(times_lof),"&", avg(times_loop),"&", avg(times_lloop),"&", avg(times_soup) ,"\\\\"




#existing datasets
print "existing datasets"

datasets = ['uci-20070111-hypothyroid.arff','uci-20070111-arrhythmia.arff','uci-20070111-glass.arff' ,'uci-20070111-diabetes.arff', 'uci-20070111-ionosphere.arff', 'uci-20070111-pendigits.arff' ]
names = ['Ann Thyroid','Arrhythmia','Glass', 'Diabetes','Ionosphere', 'Pen Digits 16']

#np.save("roc_conf_scores"+str(ds)+"_"+str(conf_teller)+".npy",to_save)
#np.save("roc_conf_times"+str(ds)+"_"+str(conf_teller)+".npy",times)

averages = {}
averages["hics"] = []
averages["lof"] = []
averages["loop"] = []
averages["lloop"] = []
averages["soup"] = []

print "{\\Dataset\\ } & {\\ HiCS\\ } & {\\ LOF\\ } & {\\ LoOP\\ } & {Local LoOP} & {\\ \\algName{}\\ } \\\\"
for ds in  range(len(names)): 
	name = names[ds]
	dataset = datasets[ds]
	scores = np.load("roc_conf_scores"+str(dataset)+"_"+str(ds)+".npy")
	aucs_hics =  metrics.roc_auc_score(scores[0],scores[1])
	averages["hics"].append(aucs_hics)
	aucs_lof = metrics.roc_auc_score(scores[0],scores[2])
	averages["lof"].append(aucs_lof)
	aucs_loop =  metrics.roc_auc_score(scores[0],scores[3])
	averages["loop"].append(aucs_loop)
	aucs_lloop =  metrics.roc_auc_score(scores[0],scores[4])
	averages["lloop"].append(aucs_lloop)
	aucs_soup = metrics.roc_auc_score(scores[0],scores[5])
	averages["soup"].append(aucs_soup)
	print name,"&", aucs_hics,"&", aucs_lof,"&", aucs_loop,"&", aucs_lloop,"&", aucs_soup ,"\\\\"

print "Average","&", np.array(averages["hics"]).mean(),"&", np.array(averages["lof"]).mean(),"&", np.array(averages["loop"]).mean(),"&", np.array(averages["lloop"]).mean(),"&", np.array(averages["soup"]).mean() ,"\\\\"
print "Std","&", np.array(averages["hics"]).std(),"&", np.array(averages["lof"]).std(),"&", np.array(averages["loop"]).std(),"&", np.array(averages["lloop"]).std(),"&", np.array(averages["soup"]).std() ,"\\\\"

print "time"
print "{\\Dataset\\ } & {\\ HiCS\\ } & {\\ LOF\\ } & {\\ LoOP\\ } & {Local LoOP} & {\\ \\algName{}\\ } \\\\"
for ds in  range(len(names)): 
	name = names[ds]
	dataset = datasets[ds]
	scores = np.load("roc_conf_times"+str(dataset)+"_"+str(ds)+".npy")
	t = np.array(scores)
	print name,"&", t[3],"&", t[4],"&", t[0],"&", t[1],"&", t[2],"\\\\"


exit()


#d400_results = []
#d_teller = 5
#for r in range(len(rs)):
#	for c in range(3):
#		d400_results.append(allresults[r*18+3*d_teller+c])

#now take the average fpr and tpr
d20_results = np.array(d20_results)
print d20_results.shape
average_scores =  d20_results[:,:-1] #do not take the time into acount
print average_scores.shape
average_scores = np.average(average_scores,axis=1)
print average_scores.shape
exit()

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
plt.title('ROC curves')
plt.legend(loc="lower right")
plt.savefig("img/roc_r"+str(r)+"-"+str(conf_teller)+".png")
conf_teller += 1
plt.clf()
exit()



