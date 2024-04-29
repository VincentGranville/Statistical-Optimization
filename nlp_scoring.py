# nlp_scoring.py | vincentg@mltechniques.com

import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

mpl.rcParams['lines.linewidth'] = 0.3
mpl.rcParams['axes.linewidth'] = 0.5
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['ytick.labelsize'] = 7


#--- [1] Read data

# read data either from GitHub, or use local copy, depending on path

# path = "https://raw.githubusercontent.com/VincentGranville/Statistical-Optimization/main/"
path = ""
filename = "Articles-Pageviews.txt"
url  = path + filename

data = pd.read_csv(url, sep='\t', engine='python', encoding='cp1252')
header = ['Title', 'URL', 'Author', 'Page views', 'Creation date', 'Status']


#--- [2] Core functions and tables

# pv is aggregated normalized pageview; pv_rel is average pv for a specific group

arr_titles = data['Title']       # article titles
arr_status = data['Status']      # blog post, forum question, etc.
arr_pv     = data['Page views']  # article absolute (un-normalized) pageviews 
arr_url    = data['URL']         # article URL
arr_author = data['Author']      # article author
arr_categories = {}              # article category (combo of author, status, URL extracts) 

category_pv    = {}     # key: article category; value: average pv per category
category_count = {}     # key: article category; value: number of articles in that category
hash_words_count = {}   # key: word; value: number of occurrences across all articles 
hash_pv = {}            # key: word; value: aggregated pv across all those articles
hash_titles = {}        # key: word; value: hash of articles containing word (value: pv)
hash_authors_count = {} # key: author; 

def compress_status(status):
    status = status.lower()
    if 'forum' in status:
        status = 'forum_question'
    elif 'resource' in status:
        status = 'resource'
    else:
        status = 'blog'
    return(status)

def compress_url(url):
    url = url.lower()
    if 'datasciencecentral' in url:
        url = 'DSC'
    else:
        url = 'Other'
    return(url)

def  update_hash(titleID, word, hash_words_count, hash_pv, hash_titles):

    if word in hash_words_count:
        hash_words_count[word] += 1
        hash_pv[word] += pv    # what if words found twice in same title ??
        hash = hash_titles[word]
        hash[titleID] = pv
        hash_titles[word] = hash
    else:
        hash_words_count[word] = 1
        hash_pv[word] = pv
        hash_titles[word] = { titleID : pv }
    return(hash_words_count, hash_pv, hash_titles)

def  update_single_tokens(titleID, words, hash_words_count, hash_pv, hash_titles):

    # words is an array of words (found in title)

    for word in words:
        hash_words_count, hash_pv, hash_titles = update_hash(titleID, word, 
                                   hash_words_count, hash_pv, hash_titles)
    return(hash_words_count, hash_pv, hash_titles)

def  update_joint_tokens(titleID, words, hash_words_count, hash_pv, hash_titles): 

    # words is an array of words (found in title)
    # capturing adjacent tokens

    for idx in range(len(words)):
        word = words[idx]
        if idx < len(words)-1 and words[idx+1] != '':
            word += "~" + words[idx+1]
            hash_words_count, hash_pv, hash_titles = update_hash(titleID, word, 
                                       hash_words_count, hash_pv, hash_titles)
        if idx < len(words)-2 and words[idx+2] != '':
            word += "~" + words[idx+2]
            hash_words_count, hash_pv, hash_titles = update_hash(titleID, word, 
                                       hash_words_count, hash_pv, hash_titles)
    return(hash_words_count, hash_pv, hash_titles)

def update_disjoint_tokens(titleID, words, hash_words_count, hash_pv, hash_titles):
    
    # words is an array of words (found in title)
    # word1 and word2 cannot be adjacent: 
    #    if they were, this is already captured in the update_joint_tokens function

    param_D = 1   # word1 and word2 must be separated by at least param_D tokens

    for k in range(len(words)):
        for l in range(len(words)):
            word1 = words[k]
            word2 = words[l]
            distance = abs(k - l)
            if word1 < word2 and distance > param_D and word1 != '' and word2 != '':
                word12 = word1 + "^" + word2
                hash_words_count, hash_pv, hash_titles = update_hash(titleID, 
                                  word12, hash_words_count, hash_pv, hash_titles)
    return(hash_words_count, hash_pv, hash_titles)

def get_article_pv(titleID, arr_pv):  
    # using log: it gives a better normalization and fit than sqrt, for pv distribution
    return(np.log(float(arr_pv[titleID])))


#--- [3] De-trend pv 

param_T1 = 0.80
param_T2 = 0.11 
arr_pv_new = np.zeros(len(arr_pv)) 

for k in range(len(arr_pv)):
    energy_boost = param_T1 * np.sqrt(k + param_T2 * len(arr_pv))
    arr_pv_new[k] = arr_pv[k] * (1 + energy_boost) 
arr_pv = np.copy(arr_pv_new)    


#--- [4] Populate core tables 

for k in range(len(data)):
    author = arr_author[k]
    if author in hash_authors_count:
        hash_authors_count[author] +=1
    else:
        hash_authors_count[author] =1

param_A = 50   # authors with fewer than param_A articles are bundled together

for k in range(len(data)):

    pv = get_article_pv(k, arr_pv) 
    cstatus = compress_status(arr_status[k])
    curl = compress_url(arr_url[k])
    category = curl + "~" + cstatus
    author = arr_author[k]
    if hash_authors_count[author] > param_A:
        arr_categories[k] = category + "~" + author
    else:
        arr_categories[k] = category

    words = str(arr_titles[k]).replace(',',' ').replace(':',' ').replace('?', ' ')
    words = words.replace('.',' ').replace('(',' ').replace(')', ' ')
    words = words.replace('-',' ').replace('  ',' ').replace('\xa0', ' ') 
    # words = words.lower() 
    words = words.split(' ')

    if 'DSC~resource' in category or 'DSC~blog' in category: 
        hash_words_count, hash_pv, hash_titles = update_single_tokens(k, words, 
                                   hash_words_count, hash_pv, hash_titles)
        hash_words_count, hash_pv, hash_titles = update_joint_tokens(k, words, 
                                   hash_words_count, hash_pv, hash_titles)
        hash_words_count, hash_pv, hash_titles = update_disjoint_tokens(k, words, 
                                   hash_words_count, hash_pv, hash_titles)

mean_pv = sum(hash_pv.values()) / sum(hash_words_count.values())
print("Mean pv: %6.3f" % (mean_pv))             


#--- [5] Sort, normalize, and dedupe hash_pv

# Words with identical pv are all attached to the same set of titles
# We only keep one of them (the largest one) to reduce the number of words

eps = 0.000000000001
hash_pv_rel = {}

for word in hash_pv:
    hash_pv_rel[word] = hash_pv[word]/hash_words_count[word]

hash_pv_rel = dict(sorted(hash_pv_rel.items(), key=lambda item: item[1], reverse=True))

hash_pv_deduped = {}
old_pv = -1
old_word = ''
for word in hash_pv_rel:
    pv = hash_pv_rel[word]
    if abs(pv - old_pv) > eps:
        if old_pv != -1:
            hash_pv_deduped[old_word] = old_pv 
        old_word = word
        old_pv = pv
    else:
        if len(word) > len(old_word):
            old_word = word
hash_pv_deduped[old_word] = old_pv

print()
print("=== DEDUPED WORDS: titles count, relative pv (avg = 1.00), word\n")
input("> Press <Enter> to continue")

for word in hash_pv_deduped: 
    count = hash_words_count[word]
    if count > 20 or count > 5 and '~' in word:
        print("%6d %6.3f %s" %(count, hash_pv_rel[word]/mean_pv, word))


#--- [6] Compute average pv per category

# Needed to predict title pv, in addition to word pv's
# Surprisingly, word pv's have much more predictive power than category pv's
# For new titles with no decent word in historical tables, it is very useful

for k in range(len(data)):
    category = arr_categories[k]
    pv = get_article_pv(k, arr_pv)
    if category in category_count: 
        category_pv[category] += pv
        category_count[category] += 1
    else:
        category_pv[category] = pv
        category_count[category] = 1

print()
input("> Press <Enter> to continue")
print()
print("=== CATEGORIES: titles count, pv, category name\n")

for category in category_count:
    count = category_count[category]
    category_pv[category] /= count
    print("%5d %6.3f %s" %(count, category_pv[category], category)) 
print()


#--- [7] Create short list of frequent words with great performance

# This reduces the list of words for the word clustering algo in next step
# Goal: In next steps, we cluster groups of words with good pv to
#       categorize various sources of good performance

short_list = {}
keep = 'good'    # options: 'bad' or 'good'

param_G1 = 1.10  # must be above 1, large value to get titles with highest pv
parma_G2 = 0.90  # must be below 1, low value to get articles with lowest pv
param_C1 = 10    # single-token word with count <= param_C1 not included in short_list
param_C2 = 4     # multi-token words with count <= param_C2 not included in short_list

for word in hash_pv_deduped: 
    count = hash_words_count[word]
    pv = hash_pv[word]/count
    if keep == 'good':
        flag = bool(pv > param_G1 * mean_pv)
    elif keep == 'bad':
        flag = bool(pv < param_G2 * mean_pv)  
    if flag and (count > param_C1 or count > param_C2 and '~' in word): 
        short_list[word] = 1


#--- [8] compute similarity between words in short list, based on common titles

# Find list of articles S1 and S2, containing respectively word1 and word2
# word1 is similar to word2 if |S1 intersection S2| / |S1 union S2| is high

hash_pairs = {}
aux_list = {}
param_S = 0.20  # if similarity score about this threshold, word1 and word2 are linked

for word1 in short_list:
    for word2 in short_list:

        set1 = set()
        for titleID1 in hash_titles[word1]:
            set1.add(titleID1)
        set2 = set()
        for titleID2 in hash_titles[word2]:
            set2.add(titleID2)

        count1 = len(set1)
        count2 = len(set2)
        count12 = len(set.union(set1, set2))
        similarity = len(set.intersection(set1, set2)) / count12

        if similarity > param_S and word1 < word2: 
            hash_pairs[(word1, word2)] = similarity
            hash_pairs[(word2, word1)] = similarity
            hash_pairs[(word1, word1)] = 1.00
            hash_pairs[(word2, word2)] = 1.00
            aux_list[word1] = 1
            aux_list[word2] = 1 


#--- [9] Turn hash_pairs{} into distance matrix dist_matrix, then perform clustering

# Keyword clustering based on similarity metric computed in previous step 
# Alternative to exploite sparse matrix: connected components algorithm on hash_pairs
# Connected components is much faster, works with big graphs
# In addition, not subject to deprecated parameters unlike Sklearn clustering
# https://github.com/VincentGranville/Point-Processes/blob/main/Source%20Code/PB_NN_graph.py

param_N = 20  # prespecified number of clusters in word clustering
n = len(aux_list)
dist_matrix  = [[0 for x in range(n)] for y in range(n)] 
arr_word = []

i = 0
for word1 in aux_list:
    arr_word.append(word1)
    j = 0
    for word2 in aux_list:
        key = (word1, word2)
        if key in hash_pairs:
            # hash_pairs is based on similarity; dist_matrix = 1 - hash_pairs is distance
            dist_matrix[i][j] = 1 - hash_pairs[(word1, word2)]
        else:
            # assign maximum possible distance if i, j are not linked
            dist_matrix[i][j] = 1.00   # maximum possible distance
        j = j+1
    i = i+1

#- Clustering, two models: hierarchichal and k-medoids, based on distance matrix

from sklearn.cluster import AgglomerativeClustering
hierarch = AgglomerativeClustering(n_clusters=param_N,linkage='average').fit(dist_matrix)

# !pip install scikit-learn-extra 
from sklearn_extra.cluster import KMedoids
kmedoids = KMedoids(n_clusters=param_N,random_state=0).fit(dist_matrix)

#- Now showing the clusters obtained from each model 

def show_clusters(model, hash_titles, arr_titles, arr_pv):

    groups = model.labels_
    hash_group_words = {}
    for k in range(len(groups)):
        group = groups[k]
        word = arr_word[k]
        if group in hash_group_words:
            hash_group_words[group] = (*hash_group_words[group], word)
        else:
            hash_group_words[group] = (word,)

    hash_group_titles = {}
    for group in hash_group_words:
        words = hash_group_words[group]
        thash = {}
        for word in words:
            for titleID in hash_titles[word]: 
                thash[titleID] = 1
        hash_group_titles[group] = thash

    for group in hash_group_words:
        print("-------------------------------------------")
        print("Group", group)
        print()
        print("keywords:", hash_group_words[group])
        print()
        print("Titles with normalized pv on the left:")
        print()
        for titleID in hash_group_titles[group]:
            pv = get_article_pv(titleID, arr_pv)
            print("%6.3f %s" %(pv, arr_titles[titleID]))
        print("\n")
 
    return(hash_group_words, hash_group_titles)

print("\n\n=== CLUSTERS obtained via hierarchical clustering\n")
input("> Press <Enter> to continue")
show_clusters(hierarch, hash_titles, arr_titles, arr_pv)

print("\n\n=== CLUSTERS obtained via k-medoid clustering\n")
input("> Press <Enter> to continue")
show_clusters(kmedoids, hash_titles, arr_titles, arr_pv)

input(">Press <Enter> to continue")

#- plot dendogram related to dist_matrix

from scipy.cluster.hierarchy import dendrogram, linkage
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html

Z = linkage(dist_matrix)    # Exercise: use Z to split the large group
dendrogram(Z)  
plt.show()


#--- [10] Predicting pv 

# Need to do cross-validation in the future
# Influenced by: param_W1, param_W2, param_G1, param_C1, param_C2, param_A, param_D
#                param_T1, param_T2
# Not influenced by: param_N, param_S
# Large param_W2 combined with small param_W1 may lead to overfitting

# We need the build inverted ("transposed") hash_titles, named reversed_hash_titles

reversed_hash_titles = {}

for word in hash_titles:
    pv_rel = hash_pv_rel[word]
    hash = hash_titles[word]
    for  titleID in hash:
        if titleID in reversed_hash_titles:
            rhash = reversed_hash_titles[titleID]
        else:
            rhash ={}
        rhash[word] = pv_rel
        reversed_hash_titles[titleID] = rhash

# Now predicting pv

observed = []
predicted = []
missed = 0
n = 0
param_W1 = 1     # low value increases overfitting, should be >= 1
param_W2 = 2.00  # chosen to minimize error between pv and estimated_pv 

for titleID in reversed_hash_titles:
    pv = get_article_pv(titleID, arr_pv)
    rhash = reversed_hash_titles[titleID]
    n += 1
    count = 0
    sum = 0
    for word in rhash:
        weight = hash_words_count[word]
        booster = 1.00
        if '~' in word:
            booster = 1.00  # test a different value
        elif '^' in word:
            booster = 1.00  # test a different value
        if weight > param_W1:
            count += booster * (1/weight)**param_W2
            sum += booster * (1/weight)**param_W2 * rhash[word]
    if count > 0:
        estimated_pv = sum / count
    else:
        missed += 1
        category = arr_categories[titleID]
        estimated_pv = category_pv[category]
    observed.append(pv)
    predicted.append(estimated_pv)
    
observed = np.array(observed)
predicted = np.array(predicted)
mean_pv =  np.mean(observed)
min_loss = 999999999.99
param_Z = 0.00

for test_param_Z in np.arange(-0.50, 0.50, 0.05):

    scaled_predicted = predicted + test_param_Z * (predicted - mean_pv)
    loss = 0
    for q in (.10, .25, .50, .75, .90):
        delta_ecdf = abs(np.quantile(observed,q)-np.quantile(scaled_predicted,q))
        if delta_ecdf > loss:
            loss = delta_ecdf
    if loss < min_loss:
        min_loss = loss
        param_Z = test_param_Z

predicted = predicted + param_Z * (predicted - mean_pv)
loss = min_loss    
mean_estimated_pv = np.mean(predicted)
mean_error = np.mean(np.abs(observed-predicted))
correl = np.corrcoef(observed, predicted)

plt.axline((min(observed),min(observed)),(max(observed),max(observed)),c='red')
plt.scatter(predicted, observed, s=0.2, c ="lightgray", alpha = 1.0)

plt.show()

print()
print("=== PREDICTIONS\n")
print("Predicted vs observed pageviews, for 4000 articles\n")
print("Loss: %6.3f" %(loss))
print("Missed titles [with pv estimared via category]: ",missed, "out of", n)
print("Mean pv (observed) : %8.3f" %(mean_pv)) 
print("Mean pv (estimated): %8.3f" %(mean_estimated_pv)) 
print("Mean absolute error: %8.3f" %(mean_error)) 
print("Correl b/w observed and estimated pv: %8.3f" %(correl[0][1]))
print()
print("Observed quantiles (left) vs prediction-based (right)")
print("P.10: %8.3f %8.3f"  %(np.quantile(observed, .10), np.quantile(predicted, .10)))
print("P.25: %8.3f %8.3f"  %(np.quantile(observed, .25), np.quantile(predicted, .25)))
print("P.50: %8.3f %8.3f"  %(np.quantile(observed, .50), np.quantile(predicted, .50)))
print("P.75: %8.3f %8.3f"  %(np.quantile(observed, .75), np.quantile(predicted, .75)))
print("P.90: %8.3f %8.3f"  %(np.quantile(observed, .90), np.quantile(predicted, .90)))

#- Plot normalized pv of articles over time

y = np.zeros(len(arr_pv))
for k in range(len(arr_pv)):
   y[k] = get_article_pv(k, arr_pv)

z = np.zeros(len(arr_pv))
window = 120  # for moving average
for k in range(len(arr_pv)):
   if k-window < 0:
       z[k] = np.mean(y[0:k+window])
   elif k+window > len(arr_pv):
       z[k] = np.mean(y[k-window:len(arr_pv)-1])
   else:
       z[k] = np.mean(y[k-window:k+window])
   
plt.plot(range(len(arr_pv)), y, linewidth = 0.2, alpha = 0.5)
plt.plot(range(len(arr_pv)), z, linewidth = 0.8, c='red', alpha = 1.0)

plt.xlim(0, len(arr_pv))
# plt.ylim(4, 10)
plt.grid(color='red', linewidth = 0.2, linestyle='--')
plt.show()
