from lib import InputDocument

__author__ = 'kristy'

from lib.LanguageModel import *

class ClusterContainer(object):
    '''A container that takes care of what's in each cluster.

    Methods are those used to create new clusters and to combine or delete current clusters.
    Other specific types of clusters inherit from this parent type.
    Handles input of data, eg getting words from tweets.

    Methods:
    __init__: Y
    __str__: Y
    merge_two_clusters Y
    insert_into_blank_cluster Y
    remove_from_cluster Y
    give_all_cluster_text Y
    print_clusters_to_files Y
    cleanup_empty_clusters Y
    print_merges N
    '''
    #TODO print_clusters_to_files cannot take arguments yet
    #TODO tweets do not mask numbers

    def __init__(self, source_filename, goalclusters, user_mask=True, hash_mask=True, url_mask=True):
        '''Initialise with corpus filename, goal cluster count m.'''

        #input from user
        self.corpus = InputDocument(source_filename)  #should be an input document object
        self.m = goalclusters

        #settings for filtering items from tweets (passed to TweetText object)
        self.resubUser = user_mask
        self.resubHash = hash_mask
        self.resubUrl = url_mask

        #holders for cluster info:
        self.clusters = [] #identity of documents in cluster
        self.tokencount = [] #total tokens in the cluster
        self.wordcount = [] #Counter object for words in the cluster

        #information about steps so far
        self.iters = 0
        self.merge_count = 0

        #writing information
        self.outdir = './outtopics/'
        self.prefix = 'topic_'

        print("Cluster container created, corpus found.")

    def __str__(self):
        return '''
        {} clusters wanted;     {} clusters exist now;
        Number of words in the clusters:
        {}
        Wordcount Counters look like this:
        {}
        '''.format(self.m, len(self.clusters)-self.merge_count, str(self.tokencount), str(self.wordcount[0]))

    def print_clusters_to_files(self,): #add keyword arguments here
        print("Now printing the cluster contents to files. You should have already performed the cleanup.")
        for ci in range(len(self.clusters)):
            if self.clusters[ci] != None:
                outfile = open(self.outdir+self.prefix+str(ci)+'.txt','w', encoding='utf-8')
                for tweet in self.clusters[ci]:
                    outfile.write(tweet.current+'\n')
                outfile.close()

    def merge_two_clusters(self, aid, bid, fillfromm=False):
        '''Merge the contents of cluster a and b to cluster a.
        Cluster b is then filled with the contents of the mth cluster
        the mth cluster is deleted'''
        self.merge_count +=1

        #merge b to a
        self.clusters[aid] += self.clusters[bid]
        self.wordcount[aid] += self.wordcount[bid]
        self.tokencount[aid] += self.tokencount[bid]

        if fillfromm:
            #put m's contents in b, delete m
            #WARNING: Only use if ever a maximum of m clusters are kept
            self.clusters[bid] = self.clusters[self.m]
            self.wordcount[bid] = self.wordcount[self.m]
            self.tokencount[bid] = self.tokencount[self.m]
            #delete m
            del self.clusters[self.m]
            del self.wordcount[self.m]
            del self.tokencount[self.m]
        else:
            #set the cluster to None, preserving indexing
            self.clusters[bid] = None
            self.wordcount[bid] = Counter()
            self.tokencount[bid] = 0

            if isinstance(self, IyerOstendorfClusters ):
                self.merged_away.append(bid)

    #add and remove things from clusters
    def insert_into_blank_cluster(self, tweet):
        self.clusters.append([tweet])
        self.wordcount.append(Counter(tweet.wordlist))
        self.tokencount.append(len(tweet.wordlist))

    def remove_from_cluster(self, cluster, tweet):
        '''Delete all the contributions from a thing in a cluster.
        Ensure that the thing given is the same object as that being removed.
        Just giving an identifier will not work.'''
        try:
            cluster.remove(tweet)
        except ValueError as err:
            print(err.args, "The tweet you want to remove from a cluster is not in the cluster!")
        num = self.clusters.index(cluster)
        #update unigram LMs for the cluster
        self.wordcount[num].subtract(Counter(tweet.wordlist))
        #update the tokencount for the cluster
        self.tokencount[num] -= len(tweet.wordlist)

    def give_all_cluster_text(self, clusterno):
        '''Queries every object in the cluster, gives words in 1D list.'''
        out = []
        for tweet in self.clusters[clusterno]:
            out += tweet.wordlist
        return out

    def cleanup_empty_clusters(self):
        '''Delete any clusters that have None elements'''
        self.clusters = [x for x in self.clusters if x is not None]
        self.wordcount = [x for x in self.wordcount if x is not None]
        self.tokencount = [x for x in self.tokencount if x is not None]


'''-----------------------------------------------------------------------------------'''
class goodmanCluster(ClusterContainer):
    '''Break a collection of documents into clusters
    by maximising unigram log probability for each document.

    All cluster attributes inherited from ClusterContainer class.
    Initialisation must include a language model object used for probability.
    Note: This does not need to be build yet, rather will return probabilities.

    Methods are specific for this quick-and-dirty clustering,
    Methods:
    __init__ Y
    __str__ Y
    initialise_to_m
    merge_to_end
    iterate_merge
    find_cluster_to_merge_into
    do_clustering: Use this one to perform the clustering (holds the other methods, others for debugging.)
    '''

    def __init__(self, source_filename, numclusters, lm, iterations=3, *kwargs):
        self = ClusterContainer.__init__(self, source_filename, *kwargs)
        self.maxiter = int(iterations)
        self.languagemodelflavor = lm #
        #self.comparisons = ComparisonTable
        print("Clusters can be generated to minimize unigram perplexity over document.")

    def __str__(self):
        return "Goodman cluster process:\n" + ClusterContainer.__str__(self)

    def initialise_to_m(self):
        '''Make m clusters containing the first tweets'''
        print("Initialising Goodman clustering.")
        while len(self.clusters) < self.m and self.corpus.finished == False:
            next_tweet = self.corpus.giveTweet(usr = self.resubUser, hash=self.resubHash, url=self.resubUrl)
        if len(self.clusters) < self.m:
            print("The clustering cannot be performed because there are fewer documents than clusters.")

    def merge_to_end(self):
        '''Read the rest of the corpus, merge these into '''
        while self.corpus.finished == False: #len is m+1, can index to m-1
            next_tweet = self.corpus.giveTweet(usr = self.resubUser, hash=self.resubHash, url=self.resubUrl)
            if next_tweet == None:
                print("Clustering one iteration has finished, all the input has been read")
            else:
                ClusterContainer.insert_into_blank_cluster(next_tweet)
                a, b  = self.find_cluster_to_merge_into(next_tweet, strict_last=True)
                ClusterContainer.merge_two_clusters(self, a, b, fillfromm=True)
        self.iters +=1
        self.cleanup_empty_clusters()

    def iterate_merge(self, print_self=False):
        '''Remove tweets from clusters and replace where probability is lowest.'''
        while self.iters < self.maxiter:
            print("\nNow reassigning tweets in the {}nd iteration".format(iter), end='\n')
            if print_self:
                print(self)
            for c in self.clusters:
                for t in c:
                    if len(c)>1: #if the cluster isn't empty
                        ClusterContainer.remove_from_cluster(c, t)
                        ClusterContainer.insert_into_blank_cluster(self, t)
                        a, b = self.find_cluster_to_merge_into(t, strict_last=True)
                        ClusterContainer.merge_two_clusters(a, b , fillfromm=True)
            self.iters +=1

    def find_cluster_to_merge_into(self, tweet_object, strict_last=False):
        '''Given an unassigned tweet object, find which cluster fits it best.'''
        if strict_last == True:
            #the content of the last class in the list is being merged/evaluated
            y = len(self.clusters) - 1
            y_content = self.clusters[-1].give_all_cluster_text() #maybe this needs a prefix
            #this next line should really save the lms instead of rebuilding them. Rethink this!
            x = max( [LanguageModel(self.clusters[i].give_all_cluster_text(), lmtype = self.languagemodelflavor, smoothing=None).score(y_content) for i in range(y)], key= lambda x: x[1])[0]
        else:
            #scores = ComparisonTable #not yet implemented to save comparison scores
            print("PROBLEM! THIS IS NOT YET READY TO WORK AGGLOMMERATING TWO CLUSTERS")
        return x, y # integers representing the two classes to be merged

    def do_clustering(self):
        self.initialise_to_m()
        self.merge_to_end()
        print("Clustering appears to have been successful")

'''------------------------------------------------------------'''

class IyerOstendorfClusters(ClusterContainer):
    '''Use agglomerative clustering based on shared unigrams to build up hierarchical clusters.

    This is based on a scoring mechanism whereby
    S = sum(1 / (word_freq * word_doc1 * word_doc2)) over shared words
    Second step is EM, not yet implemented

    methods:
    __init__ Y
    __str__ Y
    initialise_to_all Y
    merge_to_m
    find_cluster_to_merge_into using a comparison table
    do_clustering Y
    '''

    def __init__(self, source_filename, numclusters, scorefunction,
    user_mask=True, hash_mask=True, url_mask=True):
        ClusterContainer.__init__(self, source_filename, numclusters, user_mask, hash_mask, url_mask )
        self.global_word_counter = Counter()
        self.document_frequency_counter = Counter()
        self.score_table = [] #nothing yet
        #ComparisonTable(numclusters, scorefunction) #ComparisonTable(numclusters) #not numclusters because that's the final amount
        self.score_function = scorefunction
        self.merged_away = []
        print("Clusters can be created with agglomeratively to over shared vocabulary")
    def __str__(self):
        '''Print like a ClusterObject, but with explanation'''
        return "Iyer and Ostendorf step 1\n"+ClusterContainer.__str__(self)

    def do_clustering(self):
        '''Perform the clustering.'''
        self.initialise_to_all()
        self.merge_to_m() #self.score_function

    def merge_to_m(self): #scorefunction
        '''Make sure that m clusters are returned'''

        print("Now calculating the initial document similarity scores.\n"
              "The function used for this is {}".format(self.score_function))
        self.score_table = ComparisonTable(len(self.clusters), self.score_function)
        self.score_table.initial_populate(self)

        print("Now merging all the single-document clusters together.")
        x, y = 0, 0 #Initialise such something arbitrary is recalculated on the first iteration
        while len(self.clusters)- self.merge_count > self.m:
            #get the clusters to merge
            x, y = self.score_table.find_to_merge()
            if (x, y) != (None, None):
                print("Merging clusters {} and {}, {} left".format(x,y, len(self.clusters)-self.merge_count))
                #update the cluster object
                self.merge_two_clusters(x, y, fillfromm=False)
                self.merged_away.append(y)
                #update the score_table
                #self.score_table.remove_row_col(y)
                self.score_table.set_to_zeros(x,y)
                #self.score_table.table[] = np.zeros(self.score_table.width)
                self.score_table.recalculate_when_growing(self, x)
            else:
                print("Clustering stopped at >m because no more scores were available. Use smoothing!")
                break


    def initialise_to_all(self):
        print("Now initialising each document into its own cluster.")
         #build the clusters initially by making a big each document its own cluster

        while self.corpus.finished == False: #len is m+1, can index to m-1
            next_tweet = self.corpus.giveTweet(usr = self.resubUser, hash=self.resubHash, url=self.resubUrl)
            if next_tweet == None:
                continue
            self.global_word_counter.update(Counter(next_tweet.splitToWords()))
            self.document_frequency_counter.update(set(next_tweet.splitToWords()))
            ClusterContainer.insert_into_blank_cluster(self, next_tweet)



        #initialise comparisontable for all the clsters
        '''
        self.comparisons = ComparisonTable(self.corpus.linesread)
        for i in range(self.m):
            for j in range(i+1, self.m):
               # print(type(self.comparisons))
                self.comparisons.table[i][j] = self.calcS(i,j)
'''
        #printing
'''
    def calcS(self, i, j , normaliser = 1):
        S = len([word for word in self.wordcount[i] & self.wordcount[j]])
        #S = sum([
        #    normaliser / (self.globalCounter[word] * self.wordcount[i][word] * self.wordcount[j][word])
        #        for word in (self.wordcount[i] & self.wordcount[j])
        #])
        return S

    def agglomerate(self):
        a, b = self.comparisons.mergeTwoClusters()
        self.comparisons.recalculateScoresRowCol(a, self.calcS)
'''

class ComparisonTable():
    '''Class handling triangular matrix that records comparison scores
    between clusters, that is, some sort of similarity score.

    Written with numpy for (or not for) ease.
    Methods:
    __int__
    __str__
    remove_row_col Y
    recalculate_when_growing Y
    find_to_merge
    initial_populate
    '''

    def __init__(self, width, scorefunction):
        import numpy as np
        '''Initialise a table width x width'''
        self.width = width
        self.table = np.zeros((width, width))
        self.score_function = scorefunction
        print("Comparison table initialised")

    def __str__(self):
        '''Print some small indication of the table'''
        return "This is part of the first row"+str(self.table[0])#+ str([self.table[x][:max(5)] for x in range(max(5, len(self.table)))])+"..."

    def initial_populate(self, cluster_input_object):
        '''Calculate the value for every combo in the table'''
        for i in range(self.width-2):
            for j in range(i+1, self.width-1):
                self.table[i][j] = self.score_function(cluster_input_object, i, j)
        print("This is the initialisation of the table\n", self.table)



    def find_to_merge(self):
        '''Query the scores table and give ids of the things to merge'''
        #TODO: Think about whether this has to be numpy
        topscore = np.max(self.table, axis=(1,0))
        if topscore <= 0:
            print("Scores are all below the threshold, no more merges made")
            return None, None

        else:
            all_maxvals = np.where(self.table == topscore)
            bestranked = list(np.column_stack(all_maxvals))
            bestfound = False
            for pair in bestranked:
                if pair[0] != pair[1]:
                    smallerid, largerid = (pair[0], pair[1])
                    bestfound = True
                    return smallerid, largerid
            if bestfound == False:
                print("The find_to_merge failed because no pair to merge could be found.")
                return None, None

    def remove_row_col(self, clusterid):
        #Working just fine
        '''When a cluster disappears, remove the scores for it'''
        try:
            #delete the row entries
            self.table = np.delete(self.table, (clusterid), axis = 0)
            #delete the column entries
            #self.table = self.table[:clusterid]+self.table[clusterid+1:]
            self.table = np.delete(self.table, (clusterid), axis = 1)
            #print(self.table)
        except TypeError as err:
            print(err, "because you had an invalid clusterid")

    def set_to_zeros(self, x, y):
        '''Put a score value of zero in all dimensions (use for any altered/merged clusters)'''
        self.table[x, :] = np.zeros(self.width)
        self.table[y, :] = np.zeros(self.width)
        self.table[:, x] = np.zeros(self.width)
        self.table[:, y] = np.zeros(self.width)


    def recalculate_when_growing(self, cluster_object, clusterid ): #scorefunction
        '''When a cluster gains more info, reset its scores combined with everything else
        and replace with the values from scorefunction'''
        #recalculate row values from 0
        self.table[clusterid,:] = [ cluster_object.score_function(cluster_object, clusterid, j)
                                    if j > clusterid and j not in cluster_object.merged_away
                                    else 0
                                    for j in range(len(self.table))]

        #recalclate column values from 0
        self.table[:,clusterid] = [ cluster_object.score_function(cluster_object, i, clusterid)
                                    if i < clusterid and i not in cluster_object.merged_away
                                    else 0
                                    for i in range(len(self.table))]

        #print(self.table)

'''-------------------------------------------------------------------------------------------------'''

#scorefunction(cluster_object, clusterid, j)
class LanguageModel:
    '''
    Makes the language models that you might want to use when evaluating which clusters to group or not.
    Use the parameters to choose
    '''
    def __init__(self, heapofsentences):
        pass

#these so far are rough and mean nothing
'''

def lm(wc, tc, i=0):
    # a quick and dirty unigram LM from the counts, using no smoothing
    return Counter({x: v / tc[i] for x, v in wc[i].items()})
'''
def lm(clusterobject, clusternumber, smoothing = False):
    #TODO: language model items need to carry a vocabsize element that can be used for smoothing
    '''Build a quick and dirty unigram LM from the counts, using no smoothing'''
    if smoothing == False:
        return Counter({x : v / clusterobject.tokencount[clusternumber]
                        for x, v in clusterobject.wordcount[clusternumber].items()})
    else:
        vocabsize = len(set(clusterobject.wordcount[clusternumber].items()))
        if vocabsize == 0:
            return None
        global notpresent
        notpresent = 1 / vocabsize
        return Counter({x : (z + 1) / (clusterobject.tokencount[clusternumber]+ vocabsize)
                        for x, z in clusterobject.wordcount[clusternumber].items()})


def lmscore(text, unigrammodel, smoothing = False):
    '''Take a list of words, multiply probability as in model'''
    #if unigrammodel ==0: #there is nothing in this cluster anymore
    prob = 0
    for word in text:
        if smoothing == False:
            print("Why are you not using smoothing? Idiot...")
            exit()
            try:
                prob += math.log(unigrammodel[word])
            except:
                print(".", end=' ')
        else:
            #print("Smoothing level is {}".format(notpresent))
            try:
                prob += math.log(unigrammodel[word])
            except:
            #    print("O", end=' ')
                prob += math.log(notpresent)
    return prob