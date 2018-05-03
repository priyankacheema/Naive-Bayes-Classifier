import os
import glob
import re
import nltk
from nltk.stem.lancaster import LancasterStemmer

arxivPath = os.path.join(os.path.dirname(__file__), 'articles/arxiv/*.txt')
jdmPath = os.path.join(os.path.dirname(__file__), 'articles/jdm/*.txt')
plosPath = os.path.join(os.path.dirname(__file__), 'articles/plos/*.txt')
stoplistPath = os.path.join(os.path.dirname(__file__), 'stoplist.txt')


arxivData = []
for filepath in glob.glob(arxivPath):
	with open(filepath, 'r', errors='ignore') as fileobj:
		wl = []
		for line in fileobj:
			for word in line.split():
				wl.append(word.lower())
		arxivData.append(wl)

jdmData = []
for filepath in glob.glob(jdmPath):
	with open(filepath, 'r', errors='ignore') as fileobj:
		wl = []
		for line in fileobj:
			for word in line.split():
				wl.append(word.lower())
		jdmData.append(wl)

plosData = []
for filepath in glob.glob(plosPath):
	with open(filepath, 'r', errors='ignore') as fileobj:
		wl = []
		for line in fileobj:
			for word in line.split():
				wl.append(word.lower())
		plosData.append(wl)

stopwords = []
with open(stoplistPath, 'r', errors='ignore') as fileobj:
	for line in fileobj:
		stopwords.append(line.strip('\n'))
#print(stopwords)

testingData = [arxivData[:150], jdmData[150:], plosData[:150]]
trainingData = [arxivData[150:], jdmData[:150], plosData[150:]]
trainingDataLength = len(trainingData[0])+len(trainingData[1])+len(trainingData[2]) 
print('\nPre-processing step started....')

print('\nTraining and Testing Data Formed.')

#STEP 1
print("\nForming Vocabulary...")
lancasterStemmer = LancasterStemmer()
wordCount = {}
for td in trainingData:
	for article in td:
		uniqueWords = set(article)
		for word in uniqueWords:
			#print(uniqueWords)
			if word not in stopwords and re.match('^[a-zA-Z_-]*$',word):
				w = lancasterStemmer.stem(word)
				if w in wordCount:
					wordCount[w] += 1
				else:
					wordCount[w] = 1
# Remove frequency one words
vocabulary=[]
for word in wordCount:
	if wordCount[word] > 1:
		vocabulary.append(word)
vocabulary.sort()
print("\nVocabulary: ")
'''print(vocabulary)
print("\nLength of the Vocabulary: ")
print(len(vocabulary))'''
print("\nVocabulary Formed.")



#STEP 2
print('\nConverting Training data to Feature Vector....')

arxivFV = []
jdmFV = []
plosFV = []

for i in range(len(trainingData[0])): 
	article = trainingData[0][i] 
	arxivFV.append([0]*(len(vocabulary)+1))
	arxivFV[i][len(vocabulary)] = 'A'
	for word in article:
		w = lancasterStemmer.stem(word)
		try:
			arxivFV[i][vocabulary.index(w)] = 1
		except ValueError:
			pass

'''print('\nARXIV FV')
print(arxivFV)'''

for i in range(len(trainingData[1])): 
	article = trainingData[1][i] 
	jdmFV.append([0]*(len(vocabulary)+1))
	jdmFV[i][len(vocabulary)] = 'J'
	for word in article:
		w = lancasterStemmer.stem(word)
		try:
			jdmFV[i][vocabulary.index(w)] = 1
		except ValueError:
			pass

'''print('\nJDM FV')
print(jdmFV)'''

for i in range(len(trainingData[2])): 
	article = trainingData[2][i] 
	plosFV.append([0]*(len(vocabulary)+1))
	plosFV[i][len(vocabulary)] = 'P'
	for word in article:
		w = lancasterStemmer.stem(word)
		try:
			plosFV[i][vocabulary.index(w)] = 1
		except ValueError:
			pass

print('\nPre-processing step completed.')
'''print('PLOS FV')
print(plosFV)'''

# Calculate P(class), prior probability
print('\nClassification step started....')
prior_probability_A = len(arxivData)/trainingDataLength
prior_probability_J = len(jdmData)/trainingDataLength
prior_probability_P = len(plosData)/trainingDataLength


# P(article|class) = P(w1,w2,...wm|class), p(w1|class) = n(w1)/n
conditional_probability_A = [0]*len(vocabulary)
conditional_probability_J = [0]*len(vocabulary)
conditional_probability_P = [0]*len(vocabulary)

for row in arxivFV:
	for i in range(len(row)):
		if row[i] == 1:
			conditional_probability_A[i] += 1

for row in jdmFV:
	for i in range(len(row)):
		if row[i] == 1:
			conditional_probability_J[i] += 1

for row in plosFV:
	for i in range(len(row)):
		if row[i] == 1:
			conditional_probability_P[i] += 1

conditional_probability_A = [x/len(arxivFV) for x in conditional_probability_A]
conditional_probability_J = [x/len(jdmFV) for x in conditional_probability_J]
conditional_probability_P = [x/len(plosFV) for x in conditional_probability_P]

# Read test data and convert into feature vectors
test_arxivFV = []
test_jdmFV = []
test_plosFV = []

print('\narxivFeatureVector for Testing data...')
for i in range(len(testingData[0])):
	article = testingData[0][i] 
	test_arxivFV.append([0]*(len(vocabulary)))
	for word in article:
		w = lancasterStemmer.stem(word)
		try:
			test_arxivFV[i][vocabulary.index(w)] = 1
		except ValueError:
			pass

print('jdmFeatureVector for Testing data...')
for i in range(len(testingData[1])):
	article = testingData[1][i] 
	test_jdmFV.append([0]*(len(vocabulary)))
	for word in article:
		w = lancasterStemmer.stem(word)
		try:
			test_jdmFV[i][vocabulary.index(w)] = 1
		except ValueError:
			pass

print('plosFeatureVector for Testing data...')
for i in range(len(testingData[2])):
	article = testingData[2][i] 
	test_plosFV.append([0]*(len(vocabulary)))
	for word in article:
		w = lancasterStemmer.stem(word)
		try:
			test_plosFV[i][vocabulary.index(w)] = 1
		except ValueError:
			pass


predictions_A = 0
for row in test_arxivFV:
	probability_A = prior_probability_A
	probability_J = prior_probability_J
	probability_P = prior_probability_P
	for i in range(len(row)):
		if row[i] == 1:
			if conditional_probability_A!=0:
				probability_A = probability_A*conditional_probability_A[i]
			if conditional_probability_J!=0:
				probability_J = probability_J*conditional_probability_J[i]
			if conditional_probability_P!=0:
				probability_P = probability_P*conditional_probability_P[i]
		else:
			if conditional_probability_A!=1:
				probability_A = probability_A*(1-conditional_probability_A[i])
			if conditional_probability_J!=1:
				probability_J = probability_J*(1-conditional_probability_J[i])
			if conditional_probability_P!=1:
				probability_P = probability_P*(1-conditional_probability_P[i])
	classifiedLabel = ''
	if((probability_A >= probability_J) and (probability_A >= probability_P)):
		classifiedLabel = 'ARXIV'
	elif((probability_J >= probability_A) and (probability_J >= probability_P)):
		classifiedLabel = 'JDM'
	elif((probability_P >= probability_A) and (probability_P >= probability_J)):
		classifiedLabel = 'PLOS'
	print('-----------------------')
	print('Actual Class: ARXIV')
	print('Classified Class: {}'.format(classifiedLabel))
	print('-----------------------')
	if(classifiedLabel == 'ARXIV'):
		predictions_A+=1

predictions_J = 0
for row in test_jdmFV:
	probability_A = prior_probability_A
	probability_J = prior_probability_J
	probability_P = prior_probability_P
	for i in range(len(row)):
		if row[i] == 1:
			probability_A = probability_A*conditional_probability_A[i]
			probability_J = probability_J*conditional_probability_J[i]
			probability_P = probability_P*conditional_probability_P[i]
		else:
			probability_A = probability_A*(1-conditional_probability_A[i])
			probability_J = probability_J*(1-conditional_probability_J[i])
			probability_P = probability_P*(1-conditional_probability_P[i])
	classifiedLabel = ''
	if((probability_A >= probability_J) and (probability_A >= probability_P)):
		classifiedLabel = 'ARXIV'
	elif((probability_J >= probability_A) and (probability_J >= probability_P)):
		classifiedLabel = 'JDM'
	elif((probability_P >= probability_A) and (probability_P >= probability_J)):
		classifiedLabel = 'PLOS'
	print('-----------------------')
	print('Actual Class: JDM')
	print('Classified Class: {}'.format(classifiedLabel))
	print('-----------------------')
	if(classifiedLabel == 'JDM'):
		predictions_J+=1

predictions_P = 0
for row in test_plosFV:
	probability_A = prior_probability_A
	probability_J = prior_probability_J
	probability_P = prior_probability_P
	for i in range(len(row)):
		if row[i] == 1:
			probability_A = probability_A*conditional_probability_A[i]
			probability_J = probability_J*conditional_probability_J[i]
			probability_P = probability_P*conditional_probability_P[i]
		else:
			probability_A = probability_A*(1-conditional_probability_A[i])
			probability_J = probability_J*(1-conditional_probability_J[i])
			probability_P = probability_P*(1-conditional_probability_P[i])
	classifiedLabel = ''
	if((probability_A >= probability_J) and (probability_A >= probability_P)):
		classifiedLabel = 'ARXIV'
	elif((probability_J >= probability_A) and (probability_J >= probability_P)):
		classifiedLabel = 'JDM'
	elif((probability_P >= probability_A) and (probability_P >= probability_J)):
		classifiedLabel = 'PLOS'
	
	print('-----------------------')
	print('Actual Class: PLOS')
	print('Classified Class: {}'.format(classifiedLabel))
	print('-----------------------')
	if(classifiedLabel == 'PLOS'):
		predictions_P+=1

print('\nAccuracy (ARXIV) = {:.2f}%'.format((predictions_A/len(test_arxivFV)*100)))
print('Accuracy (JDM) = {:.2f}%'.format((predictions_J/len(test_jdmFV)*100)))
print('Accuracy (PLOS) = {:.2f}%'.format((predictions_P/len(test_plosFV)*100)))
