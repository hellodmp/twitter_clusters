__author__ = 'Kristy'



import random as rnd
from numpy import *
from collections import Counter
from itertools import chain

class LanguageModel(object):
    '''Given a list of words (already pre-split into a stream to build a LM over),
        make an object that contains a self.grams[int] dictionary that can be accessed by tuples.
        These don't represent the joint probability, rather the conditional probability of the last word in the tuple.
        This dictionary is accessed by a tuple of words, using .get method, else risks no value being returned.
        Use the give_sentence_probability() method to do this.
        Smoothing is applied, the types available are:
        -None
        -Add-one
        -Witten-Bell
        (More to come?)
        For several sentences, send as lists in a list (2D). These will have start/end tokens added accordingly.

        Initialise and train the LM, reading from the tweetlist given.
        The first step represents hard clustering, where each topic|document = 1'''
    startsymbol="<s>"
    endsymbol="</s>"

    @classmethod
    def duplicate_to_n_grams(cls, n, sentencelist, delete_sss=True):  #works
        '''Take a list of text, turn into a list of n-grams.
        delete_sss excludes tokens that are only start/end of sentence symbols.'''
        def make_tup(x,m):
            return tuple(sentencelist[x : x+m])
        def check_not_end(x): #works
            if x.count(LanguageModel.startsymbol) == len(x) or x.count(LanguageModel.endsymbol) == len(x):
                return False #give false if all sss
            else:
                return True #give true if its not sss
        if delete_sss:
            #print(range(len(sentencelist[n:])+1))
            a = [tuple(sentencelist[x : x+n]) for x in range(len(sentencelist[n:])+1)]
            return [y for y in a if check_not_end(y)==True]
        else:
            return [tuple(sentencelist[x : x+n]) for x in range(len(sentencelist[n-1:]))]

    @staticmethod
    def flatten_list(in_list): #works
        def unpack(x):
            #if list of 2+ items
            if type(x) == list and len(x) > 1:
                #if depth ==1 or more:
                return [unpack(y) for y in x]
            elif type(x)  == list and type(x[0])==list:
                return [unpack(y) for y in x[0]]
            #if already a list of strings, or a string
            else:
                return x
        out = []
        for item in in_list:
            out += unpack(item)
        return out

    @classmethod
    def bookend_and_flatten(cls, in_list, order): #works enough
        o = order
        def unpack(x):
            #if list of 2+ items
            if type(x) == list and len(x) > 1:
                #if depth ==1 or more:
                return [cls.startsymbol] * (o - 1) + [unpack(y) for y in x] + [cls.endsymbol] * (o - 1)
            #if just double bracketed
            elif type(x)  == list and type(x[0])==list:
                return [cls.startsymbol]*(o-1) + [unpack(y) for y in x[0]] + [cls.endsymbol] * (o - 1)
            #if already a string
            else:
                return x
        out = []
        for item in in_list:
            out += unpack(item)
        return [cls.startsymbol] * (o - 1) + out + [cls.endsymbol] * (o - 1)



        #return [unpack(item) for item in in_list]




    # @staticmethod
    # def flatten_old(any_d_list):
    #     '''Flattens lists up to some length'''
    #     def get_out(x):
    #         if type(x) == list:
    #             return [val for sublist in x for val in sublist]
    #         else:
    #             return x
    #     #return list(chain.from_iterable(any_d_list))
    #     out = get_out(any_d_list)
    #    # for item in any_d_list: #
    #    #    item = get_out(item)
    #     if type(out[0]) == str:
    #         return out
    #     else:
    #         return get_out(out)

    @staticmethod
    def bookend_list(outerlist, order=2):
        '''Take a list with any dimensions, where the lowest level is a sentence. Bookend with start and end tokens.
        Adds multiple tokens to well and truly split dispirate documents apart even if long n-grams used'''
        def bookend(endlist):
            return [[LanguageModel.startsymbol]] * (order - 1) + endlist + [[LanguageModel.endsymbol]] * (order - 1)

        for miditem in outerlist:
            if type(miditem) == str:
                outerlist = bookend(outerlist)
                break
            else:
                miditem = LanguageModel.bookend_list(miditem)
        return outerlist


        # out = []
        # for item in any_d_list:
        #     if type(item) == list:
        #         out.append( [[LanguageModel.startsymbol]] * (order - 1) + LanguageModel.bookend_list(item, order) + [[LanguageModel.endsymbol]] * (order - 1) )
        #     else:
        #         out.append([item])
        # return out

    def __init__(self, tweetlist, order, lmtype = 'simple', smoothing=None, by_document=False):
        '''Initialise with words already split(but not sentence-tagged). Give as lists.
        In case of multiple sentences, give as nested lists.
        In case of multiple documents, give as nested lists of sentences.
        This will add sentence start/end tokens appropriate to the order.

        If by_document makes a simple counter for each document, saved in list/
        Else builds a series of backing off dictionaries.'''

        self.order = int(order)
        self.smoothing = smoothing
        self.lm_type = lmtype
        self.by_doc = by_document

        if by_document:  #build term-document matrix, best with order=1
            self.tdm = [self.build_counts(self.order, self.flatten_list(x)) for x in tweetlist]
        else:  #most basic normal initialisation, not really memory-efficient yet
            self.grams = {} #set up counts of ngrams
            self.totals ={} #set up total vocab counts as integer in dictionary
            self.probs = {}  #set up probabilities of phrases using counts, smoothing and total count

            for i in range(self.order, 0, -1):
                self.grams[i] = self.build_counts(i, LanguageModel.bookend_and_flatten(tweetlist, i))
                self.totals[i] = sum(self.grams[i])
                self.probs[i] = LanguageModel.counts_to_smoothed_prob(self.grams[i], self.totals[i], smoothing="nul")
                self.grams.clear() #do we need the counts anymore?


    def __str__(self):
        return "{}-gram Language Model with {} smoothing, text or by_document: {}".format(self.order, self.smoothing,
                                                                                       'text' if self.by_doc==False else "bydoc")

    def build_counts(self, n, textstream):
        print("This is the textstream build_counts gets", textstream)
        '''Make counters from the raw input for self.grams or self.tfm'''
        return Counter(LanguageModel.duplicate_to_n_grams(n, textstream))

    @staticmethod
    def counts_to_smoothed_prob(count_dict, total_tokens, smoothing="nul"):
        '''Apply smoothing to this dictionary of counts (1 level, eg just bigrams, or just unigrams'''
        if smoothing=="nul":
            return Counter({x : v / total_tokens
                        for x, v in count_dict.items()})

        elif smoothing=="add-one-unigram":
            vocabsize = len(set(count_dict))
            notpresent = 1 / vocabsize
            return Counter({x : (z + 1) / (total_tokens+ vocabsize)
                        for x, z in count_dict.items()})
        '''More complex smoothing that considers backoff is changed from these most basic counts during testing'''
        #TODO: Figure out which other smoothing methods are needed - Witten Bell and linear interpolation at the very least

    def change_bigram_probability(self):
        #TODO - this should update the probability of a bigram to a value (value found in a child class)
        pass




#------------------------------------------------------------------------------------


    def witten_bell(self, gram, lamb=0.5):
        if len(gram) == 0:
            pass
        else:
            return (1- lamb) * self.grams[self.order].get(gram, 0) + lamb * self.witten_bell(gram[1:])




    def give_gram_probability(self, gram, lamb=0.5):
        #smoothing takes place at this level
        if self.smoothing == 'witten-bell':
            return self.witten_bell(gram, lamb)
        pass #TODO

    def give_sentence_probability(self, sentencelist):
        sentence_probability = 0
        sentencelist = [LanguageModel.startsymbol]+ sentencelist + [LanguageModel.endsymbol]
        order_grams = [tuple(sentencelist[x-self.order : x+1]) for x in range(len(sentencelist[self.order:]))]
        for g in order_grams:
            sentence_probability += self.give_gram_probability(g)
        return sentence_probability



