#Lap Yan Cheung

import nltk
import stop_list
import sys
import math
import numpy
import string

#Define function for computing consine similarity score
def getConsineSimScore(qVect, aVect, oriQVect, oriAVect):
    numerator = numpy.dot(qVect, aVect);
    if (numerator==0.0):
        return 0.0;
    squareQVect = numpy.square(oriQVect);
    squareAVect = numpy.square(oriAVect);
    squareQVectSum = 0;
    squareAVectSum = 0;
    for q in squareQVect:
        squareQVectSum+=q;
    for a in squareAVect:
        squareAVectSum+=a;
    denom = math.sqrt(squareQVectSum*squareAVectSum);
    return numerator/denom;

#Define function to get cosine similar value from each pair
def getConsineKey(pair):
    return pair[1];

f = open("cran.qry");
fr = open("cran.all.1400");
#Access queries
allQueries = f.read();
allAbstracts = fr.read();

f.close();
fr.close();

tokenizedQueries = nltk.word_tokenize(allQueries);
queries = {};
queryCount = 0;
numQueryTokens = 0;
#Store each query into dictionary
currQuery = [];

for q in tokenizedQueries:
    numQueryTokens += 1;
    #if end is reached - save current string
    if (q==".I"):
        if (len(currQuery)>0):
            queries[queryCount] = currQuery;
            currQuery = [];
            queryCount+=1;
        currQuery.append(q);
    #Add last query
    elif (numQueryTokens==len(tokenizedQueries)):
        queries[queryCount] = currQuery;
        queryCount+=1;
    else:
        currQuery.append(q)

#Debug log
#print(queryCount)

refinedQueries = {};
for q in queries:
    refinedEachQuery = [];
    for i in range (queries[q].index(".W")+1, len(queries[q])):
        refinedEachQuery.append(queries[q][i]);
    refinedQueries[q] = refinedEachQuery;

#Each query now grouped into an array in dictionary
#print(refinedQueries);


queryWordFreq = {};
wordFreqPerEachQuery = {};
#Get IDF and TF scores for each word in the queries
#Get instances of each non-stop word in each query
for q in refinedQueries:
    wordFreqPerEachQuery[q] = {};
    for w in refinedQueries[q]:
        #filter out unwanted words
        if (w not in stop_list.closed_class_stop_words)and(w not in string.punctuation)and(not (w.replace('.','',1).replace(',','',1).replace('-','',1).replace('\\','',1).replace('/','',1).isdigit())):
            if (w not in queryWordFreq):
                queryWordFreq[w]=1;
            elif (w in queryWordFreq):
                queryWordFreq[w]+=1;
            if (w not in wordFreqPerEachQuery[q]):
                wordFreqPerEachQuery[q][w]=1;
            elif (w in wordFreqPerEachQuery[q]):
                wordFreqPerEachQuery[q][w]+=1;

#Get num. of documents containing specific word
numQueryDocsWithWord = {};
for w in queryWordFreq:
    for q in refinedQueries:
        for wq in refinedQueries[q]:
            if (wq==w):
                if (wq not in numQueryDocsWithWord):
                    numQueryDocsWithWord[wq]=1;
                else:
                    numQueryDocsWithWord[wq]+=1;
                break;

#Calculate IDF score for each word
queryWordsIDF = {};
for w in queryWordFreq:
    queryWordsIDF[w] = math.log10(225/numQueryDocsWithWord[w]);

#Calculate TD-IDF score for each word for each query
wordsTFIDFperQuery = {};
for q in wordFreqPerEachQuery:
    wordsTFIDFperQuery[q] = {};
    for w in wordFreqPerEachQuery[q]:
        wordsTFIDFperQuery[q][w] = wordFreqPerEachQuery[q][w]*queryWordsIDF[w];

#Repeat steps for each abstract
tokenizedAbstracts = nltk.word_tokenize(allAbstracts);
#print(tokenizedAbstracts);
abstracts = {};
abstractCount = 0;
numAbstractsTokens = 0;
currAbstract = [];
for a in tokenizedAbstracts:
    numAbstractsTokens+=1;
    #if end is reached - save current string
    if (a==".I"):
        if (len(currAbstract)>0):
            abstracts[abstractCount] = currAbstract;
            currAbstract = [];
            abstractCount+=1;
        currAbstract.append(a);
    #Add last abstract
    elif (numAbstractsTokens==len(tokenizedAbstracts)):
        abstracts[currAbstract[1]] = currAbstract;
        abstractCount+=1;
    else:
        currAbstract.append(a);
#print(abstracts);
refinedAbstracts = {};

for a in abstracts:
    refinedEachAbstract = [];
    for i in range (abstracts[a].index('.W')+1, len(abstracts[a])):
        refinedEachAbstract.append(abstracts[a][i]);
    refinedAbstracts[a] = refinedEachAbstract;

#Debug log
#print(abstractCount);
#rint(len(refinedAbstracts));

'''
To-Do:
1) Scores & Freq for Abstracts
2) For each query -> each abstract:
    Get TF-IDF vectors of scores for query with abstract, abstract with query
3) Do cosine similarity calculation on both vectors
    - use numpy.dot(arrA, arrB) for dot product (make sure index of values are same)
4) Product output ranked by similarity scores

'''
abstractWordFreq = {};
wordFreqPerEachAbstract = {};
#Get IDF and TF scores for each word in the queries
#Get instances of each non-stop word in each query
for a in refinedAbstracts:
    wordFreqPerEachAbstract[a] = {};
    for w in refinedAbstracts[a]:
        #filter out unwanted words
        if (w not in stop_list.closed_class_stop_words)and(w not in string.punctuation)and(not (w.replace('.','',1).replace(',','',1).replace('-','',1).replace('\\','',1).replace('/','',1).isdigit())):
            if (w not in abstractWordFreq):
                abstractWordFreq[w]=1;
            elif (w in abstractWordFreq):
                abstractWordFreq[w]+=1;
            if (w not in wordFreqPerEachAbstract[a]):
                wordFreqPerEachAbstract[a][w]=1;
            elif (w in wordFreqPerEachAbstract[a]):
                wordFreqPerEachAbstract[a][w]+=1;

#print(abstractWordFreq["writings"]);

numAbsDocsWithWord = {};

'''
 #Code for gathering frequency of terms in abstracts
    #- Saved to a text file to speed up processing
count = 0;
for w in abstractWordFreq:
    count+=1;
    print(count*1.0/len(abstractWordFreq));
    for a in refinedAbstracts:
        for wa in refinedAbstracts[a]:
            if (wa.lower()==w.lower()):
                if (wa not in numAbsDocsWithWord):
                    numAbsDocsWithWord[wa]=1;
                else:
                    numAbsDocsWithWord[wa]+=1;
                break;

nf = open("numAbsWordsDocs.txt", "w");
for n in (numAbsDocsWithWord):
    #if (n!=len(numAbsDocsWithWord)-1):
    nf.write(n+" "+str(numAbsDocsWithWord[n])+"\n");
    #else: #Don't add new line for last word
    #    nf.write(n+" "+str(numAbsDocsWithWord[n]));
nf.close();
'''
#Get num. of documents containing specific word
numAbsDocsWithWord = {};
nfw = open("numAbsWordsDocs.txt"); #Saved data into text file - to improve performance
numInstancesOfWords = nfw.read().split("\n");
for w in numInstancesOfWords:
    wordInstances = w.split(" ");
    numAbsDocsWithWord[wordInstances[0]]=int(wordInstances[1]);
#Calculate IDF score for each word
absWordsIDF = {};
for w in abstractWordFreq:
    absWordsIDF[w] = math.log10(1400/numAbsDocsWithWord[w]);

#Calculate TD-IDF score for each word for each query
wordsTFIDFperAbstract = {};
for a in wordFreqPerEachAbstract:
    wordsTFIDFperAbstract[a] = {};
    for w in wordFreqPerEachAbstract[a]:
        wordsTFIDFperAbstract[a][w] = wordFreqPerEachAbstract[a][w]*absWordsIDF[w];
'''
Get consine similarity values for each query -> each abstract
    1) Create two arrays of length of query
    2) For each word in query
        - IF matched in the abstract put their respective TFIDF scores
          in the same indices in arrays
        - ELSE: 0 as value in the arrays indices
    3) Do consine similarity equation and save values in a [abstractID, consineScore] pair
        - Put pairs into an array for query #n
        - Sort abstracts for each query with cosineScore (low->high)
        - Put sorted array of query-abstract pairs into array of queries
    4) Output results for each query: each abstract
'''
queryAbstractScores = {};
count = 0;
for wQ in wordsTFIDFperQuery: #Goes thru each query
    abstractScores = [];
    #print(count*1.0/len(wordsTFIDFperQuery));
    for wA in wordsTFIDFperAbstract: #Goes thru each abstract
        vectQ = [];
        vectA = [];
        oriVectQ = [];
        oriVectA = [];
        for wordsOfQuery in wordsTFIDFperQuery[wQ]:
            oriVectQ.append(wordsTFIDFperQuery[wQ][wordsOfQuery]);
            appendedValue = False;
            for wordsOfAbstract in wordsTFIDFperAbstract[wA]:
                oriVectA.append(wordsTFIDFperAbstract[wA][wordsOfAbstract]);
                #If word is found in abstract - append its value to vectors
                if (wordsOfQuery==wordsOfAbstract):
                    vectQ.append(wordsTFIDFperQuery[wQ][wordsOfQuery]);
                    vectA.append(wordsTFIDFperAbstract[wA][wordsOfAbstract]);
                    appendedValue = True; #Record that value is appended
                    break;
            #If word not found after looping through the specific abstract
            # - append 0 to their indices
            #word not found in current abstract - no value appended
            if (not appendedValue):
                vectQ.append(0);
                vectA.append(0);


        #if(len(wordsTFIDFperQuery[wQ])==len(vectA)):
        #    print("Vectors length are equal");
        cosSim = getConsineSimScore(vectQ, vectA, oriVectQ, oriVectA);
        absCosSimPair = [int(wA)+1,cosSim];
        abstractScores.append(absCosSimPair);
    #sort the array and pass it to dictionary
    queryAbstractScores[count]=sorted(abstractScores, key=getConsineKey, reverse=True);
    count+=1;
#Output results to text file
fsub = open("output.txt", 'w');
for q in range (0, len(queryAbstractScores)):
    for qA in queryAbstractScores[q]:
        fsub.write(str(q+1)+" "+str(qA[0])+" "+str(qA[1])+" \n");

fsub.close();
