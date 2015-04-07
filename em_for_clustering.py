__author__ = 'Kristy'

#methods for em clustering on a clustercontainer object

class EMTopicClass(object):
    def __init__(self, totalclasses, tweetlist):
        self.prior = 1/totalclasses #set at start, update at each iteration
        #posterior is a list of the posterior for each tweet in order
        self.posteriors  = [] #updated for each iteration
        self.lm = LanguageModel(tweetlist, smoothing='witten-bell')
        self.temp_sent_prob = float


class ThetaParams(object):
    def __init__(self, totalclasses):
        self.m = totalclasses
        self.topics = [EMTopicClass(self.m) for x in range(self.m)]
        self.normaliser = 1

    def calc_sentence_prob(self, sentencelist):
        self.normaliser = 0
        for topic in self.topics:
            topic.temp_sent_prob = topic.lm.give_sentence_probability(sentencelist) * topic.prior
            self.normaliser += topic.temp_sent_prob

        for topic in self.topics:
            zij = topic.temp_sent_prob / self.normaliser
            topic.temp_sent_prob = float
            topic.posteriors.append(zij)

    def reset_posteriors(self):
        for topic in self.topics:
            topic.posteriors = []

def em_pre_initialised(clusterobject, iters=3):
    #the clusterobject has been pre-initialised (either randomly, agglomeratively, or kMeans)
    m = clusterobject.m

    #set parameters for iterations or precision

    #initialise the parameters as theta
    theta = ThetaParams(m) #initialise parameters for all topics #TODO: Priors are now all equal, not reflecting clustering
    alltweets = []
    tweetassociations = []
    for c in clusterobject.clusters:
        alltweets += c
        tweetassociations.append([1/m] * m)


    #the assignment of each tweet to a topic is found in  theta.topics.posteriors
    for i in range(iters):
        #expectation step -------------------------------------------------------
        for tweet in alltweets:
            theta.calc_sentence_prob(tweet) #now theta.topic.posteriors contains scores

        #maximisation step ------------------------------------------------------
        #adjust word to topic association (w|t) based on posteriors
        for topic in theta.topics:
            topic.lm.adjust() #TODO
        #adjust topic to document associaton (t|d) based the score each document got on the topic's LM
        for tweet in list(zip(alltweets, tweetassociations)):
            #tweet[0] is tweet object
            #tweet[1] is tweet topic associations



        theta.reset_posteriors()


    #compute likelihood of sentence given every topic-mix so far
    for
    #(sum these so normalisation can take place)
    #do this over every sentence. Keep the total presumably


    #maximisation step



