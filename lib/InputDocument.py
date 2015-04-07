__author__ = 'kristy'

from lib.TweetText import *


class InputDocument:
    '''Takes the file with the tweets and ids.
    No blank lines allowed.
    Methods to read a line, which returns the tweet ID string and the text,
    saves text into the dictionary file'''
    def __init__(self, filename):
        self.file = open(filename, 'r', encoding='utf-8')
        self.finished = False
        self.linesread = 0
    def __str__(self):
        return "Document object from the file {}".format(self.file)

    def giveTweet(self, usr = None, hash = None, url = None):
        line = self.file.readline()
        if line == '':
            print("End of sentence file reached")
            self.finished = True
            self.file.close()
            return None
        else:
            self.linesread +=1
            sent_id, raw_text = line.split('\t')
            current_tweet = TweetText(raw_text)
            current_tweet.maskAll(user=usr, url = url, hashtag=hash).splitToWords()
        return current_tweet
