# %%
from bs4 import BeautifulSoup

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import *

stemmer = PorterStemmer()

import spacy
nlp = spacy.load("en_core_web_sm")

from math import log10 as log
import string
import re
import os

import time  # Just used for checking how long indexing takes
import json

import matplotlib.pyplot as plt
import numpy as np

# %%

def getNameTokens(tokens, x=0):
    tokenString = (" ").join(tokens)
    #print(tokenString)
    nameTokens = list(nlp(tokenString).ents)
    i = 0
    while i<len(nameTokens):
        if len(nameTokens[i])>20:
            nameTokens.pop(i)
        else:
            nameTokens[i] = str(nameTokens[i])
            i += 1
    #for i in range(len(nameTokens)):
    #    print(str(type(nameTokens[i]))+":"+str(nameTokens[i]))
    return nameTokens

# %%

def tokenize(text):
    used_stopwords = stopwords.words('english')

    unwanted_punctuation = string.punctuation
    unwanted_punctuation = unwanted_punctuation.replace('\'', '')

    # Simplify text into tokens
    tokens = word_tokenize(text)

    # Special Treatments -----------------------------

    # Names - If two consecutive tokens start with capital letters, this is considered a name
    x = 0
    while x < len(tokens):  # Removes - as they can break up names. While loop used due to changing size of tokens list
        token = tokens[x]
        tokens[x] = tokens[x].replace("-", "")
        if len(tokens[x]) == 0:  # Now an empty string (probably was a stopword with a - attached)
            del tokens[x]
        else:
            x += 1

    tokens += getNameTokens(tokens)  # Adds names to list. Original terms as part of names still remain as seperate entries
    # ------------------------------------------------

    # Makes all tokens lower case
    x = 0
    while x < len(tokens):  # While loop used due to changing size of tokens list
        tokens[x] = tokens[x].lower()
        if len(tokens[x]) > 15:  # Manually removes tokens that are considered unreasonably long
            del tokens[x]
        else:
            x += 1


    # Removes punctuation and stopwords, then simplify to stem
    tokens = [stemmer.stem(c) for c in tokens if (not token in used_stopwords) and (not c in unwanted_punctuation)]  # Punctuation removed after name checks so it can seperate two names properly. Stemming done after
    return tokens

def processFile(file):
    soup = BeautifulSoup(file, "lxml")

    # Attempts to get the most relevant starting point to search through
    if soup.find("main"):
        main = soup.find("main")
    elif soup.find("div",{"id":"page"}):
        main = soup.find("div",{"id":"page"})
    else:
        main = soup.find("body") # Defaults to body

    # Cleanup
    divs = soup.find_all('div')
    for div in divs:
        if "style" in div:
            if "display:none" in div["style"]: # Removes hidden divs
                div.decompose()
    for nav in main.find_all("nav"): # Removes all navs as these are usually menus
        nav.decompose()
    # Adds all the relevant text to a string called text
    relevant = main.find_all({re.compile('^h[1-6]$'),"p","li"})
    text = ""
    for element in relevant:
        text += (" ").join(element.findAll(text=True)) # Adds new line so the last word of this element and the first word of the next don't join

    a_elems = main.find_all("a")
    for element in a_elems:
        if element.parent.name != "li": # Filters out the majority of menus and table of contents
            text += element.text

    return tokenize(text)


# %%

def generateIndexes():
    global postings
    global docID
    global vocabID
    # Processes every file in the wiki folder

    # Adds terms to the index
    folder_name = "ueapeople"
    #folder_name = "../ueasmall"
    print("Processing "+str(len(os.listdir(folder_name)))+" files...")
    for file in os.listdir(folder_name):
        f = open("" + folder_name+ "/" + file, "r", encoding="utf8")

        tokens = processFile(f)

        # Adds file to docID
        if file not in docID:
            docID[file] = len(docID)
        d = docID[file]

        for term in tokens:  # Loops through and adds occurrence of term into index
            # Gets vocabID
            if term not in vocabID:
                vocabID[term] = len(vocabID)
            t = vocabID[term]

            # Adds term to postings
            if t not in postings:
                postings[t] = {d: {
                    "frequency": 0}}  # Makes new entry in postings for term for the page with frequency set to 0 to start with
            if d not in postings[t]:
                postings[t][d] = {"frequency": 0}  # Makes new entry for the term with frequency set to 0 to start with
            page = postings[t][d]
            page["frequency"] += 1

    # Saves postings
    print("Saving data")
    with open("postings.json", "w", encoding='utf-8') as file:
        json.dump(postings, file, indent=4)
    # Saves docIDs
    with open("docID.json", "w", encoding='utf-8') as file:
        json.dump(docID, file, indent=4)
    # Saves vocabIDs
    with open("vocabID.json", "w", encoding='utf-8') as file:
        json.dump(vocabID, file, indent=4)
    print("Saved")

# %%

def tf_idf(term_freq, doc_freq, N):
    tfidf = log(1+term_freq) * log(N/doc_freq)
    if term_freq != 0:
        return tfidf
    else:
        return 0

# %%

def queryItems(q):
    global docID
    global vocabID
    global postings
    results = {}
    if type(q) != str:  # Tokenises terms if need be
        terms = q  # Terms may be passed in as list from other instances of this function
    else:
        terms = tokenize(q)
        #print("Start of query")
    print(terms)

    # Searches for terms
    if len(terms) > 1:  # Multi-term query found, will be fed into the recursive process
        docsQ1 = queryItems([terms[0]])  # Gets results of docs with the first term using 1 more recursion
        docsQ2 = queryItems(terms[1:]) # Gets results of docs for the rest of the terms using multiple recursions

        # If a term isn't found, it is ignored # TODO: remove this bit. seems useless
        if terms[0] not in vocabID:
            print(str(terms[0])+" not found")

        for doc in docsQ2: # Combines results
            if doc in docsQ1:
                #print("Doc:\n" +doc+ terms[0] + str(docsQ1[doc]) + "\nAND\n" + str(terms[2:]) + str(docsQ2[doc]))
                docsQ1[doc]["score"] += docsQ2[doc]["score"]
            else:
                docsQ1[doc] = docsQ2[doc]
        return docsQ1

    # Single word query found. Will be formatted to string
    q = ''.join(terms)

    # Base case
    q = q.lower()
    if q in vocabID:  # Known term
        t = str(vocabID[q])
    else:
        return {}
    if t in postings:  # Gets occurrences into results
        for d in postings[t]:
            results[d] = postings[t][d]
            #print("Doc: "+d+str(results[d]))
            if "score" not in results[d]:
                results[d]["score"] = 0
            #print("tf-idf"+str(d)+": "+str(tf_idf(results[d]["frequency"],len(postings[t]),len(docID))))
            results[d]["score"] += tf_idf(results[d]["frequency"],len(postings[t]),len(docID)) # Adds tf-idf to relevancy score
            #print("Results:"+str(results[d]))
    return results

# %%

def sortByFreq(results):  # Basic insertion sort
    rList = []
    for result in results:
        rList.append({result: results[result]})
    if len(results) == 0:
        return []
    sorted = [rList[0]]
    for x in range(1, len(rList)):
        pos = len(sorted)
        for y in range(len(sorted)):
            if list(rList[x].values())[0]["frequency"] > list(sorted[y].values())[0]["frequency"]:
                pos = y
                break
        sorted.insert(pos, rList[x])
    return sorted

def sortByScore(results):  # Basic insertion sort
    rList = []
    print("Sorting")
    for result in results:
        rList.append({result: results[result]})
    if len(results) == 0:
        return []
    sorted = [rList[0]]
    for x in range(1, len(rList)):
        pos = len(sorted)
        for y in range(len(sorted)):
            if list(rList[x].values())[0]["score"] > list(sorted[y].values())[0]["score"]:
                pos = y
                break
            elif list(rList[x].values())[0]["score"] == list(sorted[y].values())[0]["score"]:
                if list(rList[x].values())[0]["frequency"] > list(sorted[y].values())[0]["frequency"]: # Frequency used if scores are the same
                    pos = y
                    break
        sorted.insert(pos, rList[x])
    return sorted

# %%

def query(q):
    global docID
    global postings
    # Prepares query for final results
    results = sortByScore(queryItems(q))
    formattedResults = []
    docIDInv = {d: i for i, d in docID.items()}
    for item in results:  # Swaps out docID for doc name
        key = list(item.keys())[0]
        formatted = {docIDInv[int(key)]: item}
        formatted = {str(docIDInv[int(key)]):str(formatted[docIDInv[int(key)]][key]["score"])}
        formattedResults.append(formatted)
    if len(formattedResults) > 10:
        formattedResults = formattedResults[:10]
    return formattedResults

# %%

docID = {}
postings = {}
vocabID = {}

def loadData():
    global docID,postings,vocabID
    with open("indexes/postings.json", "r", encoding='utf-8') as file:
        postings = json.load(file)
    with open("indexes/docID.json", "r", encoding='utf-8') as file:
        docID = json.load(file)
    with open("indexes/vocabID.json", "r", encoding='utf-8') as file:
        vocabID = json.load(file)

# %%

def clearData():
    print("Deleting data")
    with open("indexes/postings.json", "w", encoding='utf-8') as file:
        json.dump({}, file, indent=4)
    # Saves docIDs
    with open("indexes/docID.json", "w", encoding='utf-8') as file:
        json.dump({}, file, indent=4)
    # Saves vocabIDs
    with open("indexes/vocabID.json", "w", encoding='utf-8') as file:
        json.dump({}, file, indent=4)
    print("Deleted")

# %%

# Console

command = ""
loadData()
while command != "exit":
    command = input("command: ")
    t0 = time.perf_counter()
    if command == "process":
        clearData()
        generateIndexes()
        loadData()
    elif command == "query":
        command = input("query: ")
        while command != "<":
            t0 = time.perf_counter()
            loadData() # Resets document scores
            results = query(command)
            print("Results: " + str(results))

            # Makes graph of results and their scores
            scores = [float(list(r.values())[0]) for r in results]
            x = [list(r.keys())[0] for r in results]
            plt.bar(x,scores)
            plt.title("\""+command+"\"")
            plt.xlabel("Words")
            plt.ylabel("Score")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

            t1 = time.perf_counter()
            print("Time taken: " + str(round(t1 - t0, 100)) + " seconds")
            command = input("query: ")
    elif command == "clear":
        clearData()
    elif command == "help":
        print("\033[1mProgram functions:\033[0m")
        print("exit - exits program")
        print("process - indexes and processes files")
        print("query - enters query mode (exit by entering \"<\")")
        print("clear - clears all stored data")
        print("help - well I think you already know what this does")
        print("")

    t1 = time.perf_counter()
    print("Time taken: " + str(round(t1 - t0, 100)) + " seconds")

# %%