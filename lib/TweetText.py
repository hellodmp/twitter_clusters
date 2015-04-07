__author__="kristy"
'''Class to manipulate the text of a tweet object.
Initialise by giving it a string, it saves the raw thing you gave it as
self.raw
Can create a quick and dirty mask/break operation with
self.autolist()
But is best used as '''

import re

class TweetText:
    #TODO: Break punctuation away from words
    #TODO: replace numbers with holder character
    '''String class with methods for transformation of tweet text,
    Useful for hiding user information'''
    usrString = "@usr"
    urlString = "/url/"
    hashString = "#tag"
    numString = '5'

    def __init__(self, origString): ## id = "a"+str(random.randint(0,10000))):
        #self.id = id
        self.raw = origString.strip()
        self.current = self.raw
       # self.wordlist = self.autoList()
        self.wordlist = []

    def __str__(self):
        return "RAWINPUT: {}\n=>\nCURRENTFORM: {}\n=>\nCURRENTWORDLIST: {}".format(self.raw, self.current, self.wordlist)

    def autoList(self):
        '''Without changing self.current, return a list of the words'''
        return self.maskAll(change_currentform=False).splitToWords()

    #methods that change the self.current string form to anonymise user/links/hashtags
    def maskUser(self):
        self.current = re.sub(r"(?<!\w)\@[^\s]+",TweetText.usrString, self.current)
        return self
    def maskURL(self):
        self.current = re.sub(r"http:\S+", TweetText.urlString, self.current)
        return self
    def maskHashtag(self):
        self.current = re.sub(r"(?<!\w)\#[^\s]+", TweetText.hashString, self.current)
        return self

    def maskAll(self, user=True, url=True, hashtag=True, number=True, change_currentform=True):
        '''Method that either changes the self. crrent string if change_currentform is true,
        otherwise just returns a copy of the object that can be further manipulated without damaging this string
        Changes only those attributes chosen and generates the full split to words'''
        if change_currentform == True:
            work_on_object = self
        else:
            self_copy = self; work_on_object = self_copy

        if user:
            work_on_object.maskUser()
        if url:
            work_on_object.maskURL()
        if hashtag:
            work_on_object.maskHashtag()
        if number:
            work_on_object = re.sub(r'[0-9]', '5', work_on_object)
        work_on_object.wordlist = work_on_object.splitToWords()
        return work_on_object

    def splitToWords(self, split_punct=True):
        '''Split the current form of the tweet into a wordlist
        If split_punct is chosen then full-stops etc (trailing a word, not in urls) treated as separate words'''
        temp = self.current
        if split_punct:
            temp = re.sub(r"(?<=\w)([^\d\s\w])(?=\s)",r" \1", temp)

        #do any other splitting operations on the string here
        temp = temp.lower()
        self.wordlist = re.split(r'\s+', temp) #splits at all whitespace
        return self.wordlist

    def giveId(self):
        return self.id
    def giveWords(self):
        return self.wordlist
