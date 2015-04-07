__author__ = 'kristy'
from lib.ClusterContainer import *
from lib.ClusterContainer import IyerOstendorfClusters
def main():
    def testScoreFunction(cluster_object, i, j):
        S = len([word for word in cluster_object.wordcount[i] & cluster_object.wordcount[j]])
        return S

    def originalS(cluster_object, i, j , ):
        def normaliser(x, y): return math.sqrt((x+y)/(x*y))
        def documentcount(n): return len(cluster_object.clusters[n])

        S = sum([
            normaliser(documentcount(i), documentcount(j))  # normaliser (numerator)
            / (cluster_object.document_frequency_counter[word]   #number of docs with word (document frequency)
               * len(cluster_object.wordcount[i])  #unique words in doc i
               * len(cluster_object.wordcount[j])) #unique words in doc j
                for word in (cluster_object.wordcount[i] & cluster_object.wordcount[j])])
        return S


    c = IyerOstendorfClusters('../text.only', 30, originalS)
    print("initialising")
    c.initialise_to_all()
    c.merge_to_m()
    print(c)
    c.print_clusters_to_files()
    for eachcluster in c.clusters:
        if eachcluster != None:
            print("**********************************************")
            for tweet in eachcluster: #things in cluster 1
                #print(len(tweet))
                print(tweet.current)
#    print(c.score_table.find_to_merge()
#     print("These are the first two clusters to merge")
#     for i in range(3):
#         print("*" * 40)
#         print("Iteration {}".format(str(i+1)))
#         x, y = c.score_table.find_to_merge()
#         c.merge_two_clusters(x, y)
#         print("About to remove the higher row")
#         c.score_table.remove_row_col(y)
#         print(c.score_table)
#         print("About to recalcualte the lower column")
#         c.score_table.recalculate_when_growing(c, x)
#         print(c.score_table)

    #c.initialise_to_all()
    #c.do_clustering()
#    b = testScoreFunction(c, 1, 2)
    #print(b)



def notmain():
    '''
    foo = InputDocument('text.only')
    foo.giveTweet()
    foo.giveTweet()
    foo.giveTweet()
    foo.giveTweet()
    foo.giveTweet()
    foo.giveTweet()
    b = foo.giveTweet()
    print(b.autoList())
    a = foo.giveTweet(usr=True, hash=True, url=False)
    print(a.current)
    print(a)

    exit()
    for i in range(15):
        a = foo.giveTweet()
        print(a.maskAll(user=False, hashtag=False).splitToWords())


    c = ClusterObject('./text.only.train', 30, hash_mask=False)
    c.makeGoodmanClusters(iters = 50)
    print(c.clusters)

    for eachcluster in c.clusters:
        print("**********************************************")
        for tweet in eachcluster: #things in cluster 1
            #print(len(tweet))
            print(tweet.current)
    print(len(c.clusters))
    print(c)
'''

    #text.only'
    c = IyerOstendorfClusters('./text.only.train', 5)
    c.initialise()
    c.do_clustering()

    # for i in range(4):
    #     print("iteration {}".format(i))
    #     print(c.tokencount)
    #     print(c.comparisons)
    #     c.agglomerate()
    #     print(c.tokencount)
    #     print(len(c.clusters), " clusters formed")


if __name__=="__main__":
    main()
