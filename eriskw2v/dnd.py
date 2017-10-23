import pandas as pd 
import numpy as np 
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score,classification_report,make_scorer
from sklearn.ensemble import RandomForestClassifier as RFC,AdaBoostClassifier as Ada
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB as NB 
import argparse

"""
     USAGE : python feorm.py --train <train-csv> --test <test-csv> --trorte <cross> or <test>

     outputs submission.txt
"""

#relabel = lambda x: 0 if x=='female'  else 1
#unlabel = lambda y: 'female' if y==0  else 'male' 

def get_data_frame(csv_file):
	df = pd.read_csv(csv_file)
	labels = np.asarray([ int(row[-1]) for row in df.values])
	df = df.drop(labels=df.columns[-1],axis=1)
	return df,labels

def get_fnames(unpruned_arff):
	fnames = []
	with open(unpruned_arff) as af:
		data_flag = False
		for line in af:
			pieces = line.split()
			if data_flag:
				temp = pieces[-1].split('_')
				fnames.append(temp[-2]+'.'+temp[-1])

			if pieces[0] == '@data':
				data_flag = True
	return fnames

def write_submission(outfile,preds,fnames):
	with open(outfile,'w') as out:
		for fname,pred in zip(fnames,preds):
			out.write('{0},{1}'.format(fname,unlabel(pred)))
			out.write('\n')



if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='python feorm.py --train <train-csv> --test <test-csv> --fnames <unpruned-test-arff file>')
	parser.add_argument('-tr','--train',help='path to train-csv',required=True)
	parser.add_argument('-te','--test',help='path to test-csv',required=True)
	parser.add_argument('-tt','--trorte',help='type cross or test',required=True)
	args= vars(parser.parse_args())

	trdf , labels = get_data_frame(args['train'])
	tedf , truth = get_data_frame(args['test'])
	#fnames = get_fnames(args['fnames'])


	rf = RFC(n_estimators=50,
	 criterion='entropy', 
	 max_depth=None, 
	 min_samples_split=2, 
	  max_leaf_nodes=None,
	  class_weight = 'balanced'
	)
	svm_rbf = svm.SVC(C=20,kernel='rbf',gamma = 'auto',class_weight = 'balanced')
	svm_linear = svm.LinearSVC(C=1,class_weight = 'balanced')
	nb = NB()
	
	clfs = [rf,svm_rbf,svm_linear,nb]

	if args['trorte'] == 'cross':
		scoring = ['f1_macro','accuracy','precision_macro','recall_macro']
		for clf in clfs:
			print(clf) 
			scores = cross_validate(clf,trdf,labels,scoring = scoring,cv=10,return_train_score=False)
			for s in scores.keys():
				print(s)
				for v in scores[s]:
					print(v)
				print ('###############################')
			print ('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
	elif args['trorte'] == 'test':
	#clf.fit(trdf,labels)
		for clf in clfs:
			clf.fit(trdf,labels)
		zs =[ clf.predict(tedf) for clf in clfs]
		#z = clf.predict(tedf)
		#zrbf = clf2.predict(tedf)
		#zlinear = clf3.predict(tedf)
		for clf,z in zip(clfs,zs):
			print (clf)
			print ('{0}'.format(classification_report(y_true=truth,y_pred=z))) #target_names=['female','male']



