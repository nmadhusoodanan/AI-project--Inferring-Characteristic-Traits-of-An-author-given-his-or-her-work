from __future__ import print_function

import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
import matplotlib.pyplot as plt
import itertools
from read_pan import read_pan

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn.datasets import load_files
from sklearn import linear_model
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.cross_validation import train_test_split
#from .base import load_files
import csv

PATH_HOME='/home/namrita/Downloads/AIdata/'
FILESNAME_MASTER='lookuptable.csv'
PATH_TEST='test/'
PATH_TRAIN='train/'

# opts.use_hashing='false'
# opts.select_chi2='false'

# # parse commandline arguments
op = OptionParser()
# op.add_option("--report",
#               action="store_true", dest="print_report",
#               help="Print a detailed classification report.")
op.add_option("--chi2_select",
              action="store", type="int", dest="select_chi2",
              help="Select some number of features using a chi-squared test")
# op.add_option("--confusion_matrix",
#               action="store_true", dest="print_cm",
#               help="Print the confusion matrix.")
# op.add_option("--top10",
#               action="store_true", dest="print_top10",
#               help="Print ten most discriminative terms per class"
#                    " for every classifier.")
# op.add_option("--all_categories",
#               action="store_true", dest="all_categories",
#               help="Whether to use all categories or not.")
op.add_option("--use_hashing",
              action="store_true",
              help="Use a hashing vectorizer.")
# op.add_option("--n_features",
#               action="store", type=int, default=2 ** 16,
#               help="n_features when using the hashing vectorizer.")
# op.add_option("--filtered",
#               action="store_true",
#               help="Remove newsgroup information that is easily overfit: "
#                    "headers, signatures, and quoting.")
# op.add_option("--gender",
#               action="store_true",
#               help="Classify based on getarget# 
(opts, args) = op.parse_args()
# if len(args) > 0:
#     op.error("this script takes no arguments.")
#     sys.exit(1)
	# print(__doc__)
	# op.print_help()
	# print()



# #filehandling
# with open(PATH_HOME+FILESNAME_MASTER,'r') as f:
# 	x = csv.reader(f)
# 	all_stuff = [r for r in x]
# 	filenames = [x[0] for x in all_stuff]
# 	filepaths = [PATH_HOME+PATH_TRAIN+'J/'+x[0] for x in all_stuff]
# 	authors = [x[1] for x in all_stuff]
# 	genders = [x[3] for x in all_stuff]
# 	# print (filenames,authors,genders)

#filehandling
# with open(PATH_HOME+FILESNAME_MASTER,'r') as f:


# ##################################################################################################################
dat = csv.reader(open(PATH_HOME+FILESNAME_MASTER, 'r'), delimiter = ';')

labels = ([h[0] for h in dat][0]).split(',')
columns = []
with open(PATH_HOME+FILESNAME_MASTER,'r') as f:
	reader = csv.DictReader(f)
	g= [r for r in reader]
	f.close()
	for j in range(len(labels)):
		for i in range(len(g)):
			columns.append([v for k,v in g[i].iteritems() if k==labels[j]])
f.close()
columns= np.array(columns).reshape(len(labels),len(g))
no_labels, no_samples = np.shape(columns)

dict_labels = {labels[itr]:columns[itr] for itr in range(no_labels)}
filenames = dict_labels[labels[0]]
# print(labels[0])
filepaths= [PATH_HOME+PATH_TRAIN+'J/'+x for x in dict_labels[labels[0]]]
import sys

###################################################################################################################################
def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = []; lem = []
    sys.stdout.write('.')
    for item in tokens:
        stems.append(PorterStemmer().stem(item))
    for item in stems:
		lem.append(WordNetLemmatizer().lemmatize(item))
    return lem

def loadtrain(label):
	k=load_files(PATH_HOME+PATH_TRAIN, encoding='latin1')
	# print (k.target)
	# print(k.filenames[1].type)
	# k.filenames=np.array(filenames).astype('S')
	filename_with_data = {f:d for f,d in zip(k.filenames, k.data)}
	# print(k.data)
	# print (k.filenames)
	# print(np.array(filepaths))
	# print()
	# print ([v for k,v in filename_with_data.iteritems()])
	# print (filename_with_data.keys())
	# fp=np.array(filepaths).astype('S')
	# exit()
	# print (filename_with_data['/home/namrita/Documents/AI/train/')
	
	# print ([x for x in filepaths])

	# data_o = [filename_with_data[x] for x in fp]
	data_o=[filename_with_data.get(i) for i in np.array(filepaths) if i in k.filenames]
	# print(data_o)
	# print(k)
	k.filenames=np.array([x for x in filepaths if x in k.filenames])
	# print(k.filenames)
	# exit()
	k.data=data_o
	y = search_y(label,k.filenames,PATH_TRAIN+'J/')
	# print (y)
	k.target=y
	# print (k.target)
	return k

def get_y(label):
	y=list()
	if label=='Gender':
		z=dict_labels['Gender']
		l=len(z)
		for i in range(0,l):
			if z[i]=='M':
				y.append(1.0)
			else:
				y.append(0.0)

	if label=='Genre1':
		z=dict_labels['Genre']
		l=len(z)
		z1=['F','R','FA','CF','HF','SF','HD','T','PF','P','PD','H']
		z2=['AB','B','NF']
		z3=['G','A']
		z4=['C']
		for i in range(0,l):
			if z[i] in z1:
				y.append(0.0)
			if z[i] in z2:
				y.append(1.0)
			if z[i] in z3:
				y.append(2.0)
			else:
				y.append(3.0)
	if label=='Genre2':
		z=dict_labels['Genre']
		l=len(z)
		z1=['AB','B']
		z2=['NF']
		z3=['F','R','HD','PF']
		z4=['H']
		z5=['FA']
		z6=['CF','HF','T']
		z7=['P','PD']
		z8=['G','A']
		z9=['C']
		z10=['SF']
		for i in range(0,l):
			if z[i] in z1:
				y.append(0.0)
			if z[i] in z2:
				y.append(1.0)
			if z[i] in z3:
				y.append(2.0)
			if z[i] in z4:
				y.append(3.0)
			if z[i] in z5:
				y.append(4.0)
			if z[i] in z6:
				y.append(5.0)
			if z[i] in z7:
				y.append(6.0)
			if z[i] in z8:
				y.append(7.0)
			if z[i] in z9:
				y.append(8.0)
			else:
				y.append(9.0)

	if label=='yob1':
		z=dict_labels['YOB']
		l=len(z)
		for i in range(0,l):
			if z[i]>1950:
				y.append(2.0)
			if z[i]<1900:
				y.append(0.0)
			else:
				y.append(1.0)
			print(y)
			exit()	

	if label=='yob2':
		z=dict_labels['YOB']
		l=len(z)
		for i in range(0,l):
			y.append(z[i])

	if label=='yop':
		z=dict_labels['YOP']
		l=len(z)
		for i in range(0,l):
			y.append(z[i])			

	if label=='Age':
		z=dict_labels['Age']
		l=len(z)
		for i in range(0,l):
			y.append(z[i])

	if label=='Age2':
		z=dict_labels['Age']
		z=z.astype('float')
		l=len(z)
		# print(type(z))
		# exit()
		# z=(float(x) for x in z)
		# l=len(z)
		for i in range(0,l):
			if z[i]<=35.0:
				y.append(0.0)
			if z[i]>35.0 and z[i]<=50.0:
				y.append(1.0)
			if z[i]>50.0 and z[i]<=65.0:
				y.append(2.0)
			else:
				y.append(3.0)

			# if z[i]>=20.0 and z[i]<30.0:
			# 	y.append(0.0)
			# if z[i]>=30.0 and z[i]<40.0:
			# 	y.append(1.0)
			# if z[i]>=40.0 and z[i]<50.0:
			# 	y.append(2.0)
			# if z[i]>=50.0 and z[i]<60.0:
			# 	y.append(3.0)
			# if z[i]>=60.0 and z[i]<70.0:
			# 	y.append(4.0)
			# if z[i]>=70.0 and z[i]<80.0:
			# 	y.append(5.0)
			# if z[i]>=80.0 and z[i]<90.0:
			# 	y.append(6.0)
			# else:
			# 	y.append(7.0)
											


	return y
def feature_extraction(label,train_data):
	# train_data=loadtrain(label)
	if opts.use_hashing:
	    vectorizer = HashingVectorizer(stop_words='english', non_negative=True,
	                                   n_features=opts.n_features)
	    X_train = vectorizer.transform(train_data.data)
	    feature_names = None

	else:
	    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,stop_words='english')
	    X_train = vectorizer.fit_transform(train_data.data)
	    # mapping from integer feature name to original token string
	    feature_names = vectorizer.get_feature_names()
        feature_names = np.asarray(feature_names)


	ch2=None
	if opts.select_chi2:
	    print("Extracting %d best features by a chi-squared test" %
	          opts.select_chi2)
	    t0 = time()
	    ch2 = SelectKBest(chi2, k=opts.select_chi2)
	    X_train = ch2.fit_transform(X_train, y_train)
	    X_test = ch2.transform(X_test)
	    if feature_names:
	        # keep selected feature names
	        feature_names = [feature_names[i] for i
	                         in ch2.get_support(indices=True)]
	return X_train, feature_names, ch2, vectorizer   

def load_test_files(transformer):
	t=load_files(PATH_HOME+PATH_TEST,encoding='latin1')
	feature_vector = transformer.transform(t.data)

	return t,feature_vector

def search_y(label,namelist,mid_path):
	fp = [PATH_HOME+mid_path+x for x in filenames]

	filename_with_label = {f:d for f,d in zip(fp,get_y(label))}#dict_labels('filepaths'), dict_labels(label))}
	# print(filename_with_label)
	k=[filename_with_label[i] for i in namelist]
	return k


def simple_classify(clf,test_x,test_y,train_x,train_y):


    print(clf)
    t0 = time()
    # print (t0)
    clf.fit(train_x,train_y)

    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(test_x)

    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)
    score = metrics.accuracy_score(test_y, pred)
    print("accuracy:   %0.3f" % score)
    return ' '#zip(pred,test_y)

def bayesian_ridge_regression(test_x,test_y,train_y,train_x):
	reg = linear_model.BayesianRidge()
	train_x=train_x.toarray().astype(np.float)
	# print(type(train_x))
	train_y=np.array(train_y).astype(np.float)
	# print(type(k.target))
	t0 = time()	
	print(reg.fit(train_x,train_y))
	train_time = time() - t0
	print("train time: %0.3fs" % train_time)
  	t0 = time()	
	pred=reg.predict (test_x)
	test_time = time() - t0
 	print("test time:  %0.3fs" % test_time)
 	print(zip(pred,test_y))
	test_y = [float(x) for x in test_y]
	count=0;
	l=len(test_y)
	for i in range(0,l):  
		if (abs(pred[i]-test_y[i])<10):
			count=count+1
	count=float(count)
	l=float(l)
	score=(count/l)
	print("Score",score)
	fig, ax = plt.subplots()
	ax.scatter(test_y, pred)
	ax.set_xlabel('Measured')
	ax.set_ylabel('Predicted Using Bayesian Ridge Regression')

	plt.show()



def ridge_regression(test_x,test_y,train_y,train_x):
	reg = linear_model.Ridge (alpha = .5)
	train_x=train_x.toarray().astype(np.float)
	# print(type(train_x))
	train_y=np.array(train_y).astype(np.float)
	t0 = time()
	# print(type(k.target))
	reg.fit(train_x,train_y)
	train_time = time() - t0
 	print("train time: %0.3fs" % train_time)
  	t0 = time()
	pred=reg.predict (test_x)
	test_time = time() - t0
 	print("test time:  %0.3fs" % test_time)
 	print(zip(pred,test_y))
	test_y = [float(x) for x in test_y]
	count=0;
	l=len(test_y)
	for i in range(0,l):  
		if (abs(pred[i]-test_y[i])<10):
			count=count+1
	count=float(count)
	l=float(l)
	score=(count/l)
	print("Score",score)
	fig, ax = plt.subplots()
	ax.scatter(test_y, pred)
	ax.set_xlabel('Measured')
	ax.set_ylabel('Predicted Using Ridge Regression')

	plt.show()






# Benchmark classifiers
def benchmark(clf):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)

    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

        if opts.print_top10 and feature_names is not None:
            print("top 10 keywords per class:")
            for i, label in enumerate(target_names):
                top10 = np.argsort(clf.coef_[i])[-10:]
                print(trim("%s: %s" % (label, " ".join(feature_names[top10]))))
        print()

    if opts.print_report:
        print("classification report:")
        print(metrics.classification_report(y_test, pred,
                                            target_names=target_names))

    if opts.print_cm:
        print("confusion matrix:")
        print(metrics.confusion_matrix(y_test, pred))

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time

givenlabel='Age'
# k=loadtrain(givenlabel)
# # print (k)
# train_x,f_names,chi,transformer=feature_extraction(givenlabel,k)
# test,test_x=load_test_files(transformer)
# test_y=search_y(givenlabel,test.filenames,PATH_TEST+'J/')

# reg_labels=['yob2','yop','Age']
# if givenlabel in reg_labels:
# 	bayesian_ridge_regression(test_x,test_y,k.target,train_x)
# 	# ridge_regression(test_x,test_y,k.target,train_x)
def feature_extraction2(label,train_data):
# train_data=loadtrain(label)
	if opts.use_hashing:
	    vectorizer = HashingVectorizer(stop_words='english', non_negative=True,
	                                   n_features=opts.n_features)
	    X_train = vectorizer.transform(train_data)
	    feature_names = None

	else:
	    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,stop_words='english')
	    X_train = vectorizer.fit_transform(train_data)
	    # mapping from integer feature name to original token string
	    feature_names = vectorizer.get_feature_names()
	    feature_names = np.asarray(feature_names)


	ch2=None
	if opts.select_chi2:
	    print("Extracting %d best features by a chi-squared test" %
	          opts.select_chi2)
	    t0 = time()
	    ch2 = SelectKBest(chi2, k=opts.select_chi2)
	    X_train = ch2.fit_transform(X_train, y_train)
	    X_test = ch2.transform(X_test)
	    if feature_names:
	        # keep selected feature names
	        feature_names = [feature_names[i] for i
	                         in ch2.get_support(indices=True)]
	return X_train, feature_names, ch2, vectorizer  

def writeTrainToTxt(write_d,X_TRAIN_PATH):

	dump_data = ((write_d).toarray())#.astype(np.float))
	print (np.shape(dump_data))
	np.savetxt(X_TRAIN_PATH, dump_data)
	return

	
# else:
    # print(simple_classify(MultinomialNB(),test_x,test_y,train_x,k.target))
    # print(simple_classify(RandomForestClassifier(),test_x,test_y,train_x,k.target))
    # print(simple_classify(RidgeClassifier(),test_x,test_y,train_x,k.target))
    # print(simple_classify(KNeighborsClassifier(),test_x,test_y,train_x,k.target))
    # print(simple_classify(Perceptron(),test_x,test_y,train_x,k.target))
    # print(simple_classify(PassiveAggressiveClassifier(),test_x,test_y,train_x,k.target))
#New Code For PAN
pan_train='/home/namrita/Downloads/AIdata/pan13-author-profiling-training-corpus-2013-01-09/en'
pan_temp='/home/namrita/Downloads/AIdata/pantemp'
# pan_test='/home/namrita/Downloads/AIdata/pan13-author-profiling-test-corpus2-2013-04-29/en'
print ('Reading database ...')
k,g,y_train = read_pan(pan_train)
# print (y_train)
# k_t, test_y, a_t = read_pan(pan_test)
print('Extracting asset')
train_x,f_names,chi,transformer=feature_extraction2(givenlabel,k)
# print ('Writing database ...')
# writeTrainToTxt(train_x,'feature.txt')
# # writeTrainToTxt(y_train,'y.txt')
# print ('written')

train_X, test_X, train_y, test_y = train_test_split(train_x, y_train, train_size=0.80)
print ('Mission successful')
# print(len(X_train))
# test_x,_,_,_=feature_extraction2(givenlabel,k_t)
# train_x=train_x.toarray().astype(np.float)
# print (train_x.dtype)


# train_x,f_names,chi,transformer=feature_extraction(givenlabel,k)
# print(simple_classify(RandomForestClassifier(),test_X,test_y,train_X,train_y))
print(simple_classify(RidgeClassifier(),test_X,test_y,train_X,train_y))
print(simple_classify(Perceptron(),test_X,test_y,train_X,train_y))
print(simple_classify(PassiveAggressiveClassifier(),test_X,test_y,train_X,train_y))
print(simple_classify(RandomForestClassifier(),test_X,test_y,train_X,train_y))
print(simple_classify(KNeighborsClassifier(),test_X,test_y,train_X,train_y))
print(simple_classify(MultinomialNB(),test_X,test_y,train_X,k))