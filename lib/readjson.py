#!/usr/bin/env python

__author__ = 'Kristy'

'''Script to read the original JSON file format of twitter dump and extract info.
Creates tweet objects that can be (but are not yet) stored in a dictionary.
'''

import json
import subprocess
import re
from pprint import pprint

class JSONTweet:
    def __init__(self, json_text):
        td = json.loads(json_text) #LOAD THE TWEET
        #pprint(td)  #sanity check
        self.id = td['id']
        self.text = re.sub('[\n\t]', ' ', td['text'])  #remove newlines in the tweet
        self.created=td['created_at']
        self.timestamp = td['timestamp']
        self.reply = td['in_reply_to_status_id']
        self.retweetcount = td['retweet_count']

    def __str__(self):
        return("id: {}, text: {}, reply: {}, retweeted: {}"\
              .format(str(self.id), self.text, self.reply, self.retweetcount))


def loadtweets(filename, textout, linedivision=(80,10,10)):
    '''TODO: textout should be a prefix, not a filename'''
   # o = open(textout, 'w+', encoding='utf-8')
    #init_holder = dict()
    counter = 0
    with open(filename, 'r') as f:
        for line in f:
            #control which file to print to
            if counter ==0:
                o = open(textout+'.train', 'w', encoding='utf-8')
            elif counter == linedivision[0]: #end of training
                o.close()
                counter = 0
                o = open(textout+'.dev', 'w', encoding='utf-8')
            elif counter == linedivision[1]: #end of dev
                o.close()
                counter = 0
                o = open(textout+'.test', 'w', encoding='utf-8') #open test til end
            counter +=1
            tweet_obj = JSONTweet(line)
            print(tweet_obj) #EXTRACT IDENTIFIER etc, rest of json discarded here
            #init_holder[tweet_obj.id] = tweet_obj#SAVE IN DICTIONARY BY id_str as object [is this large enough?]
            o.write(str(tweet_obj.id)+"\t"+tweet_obj.text+"\n")
    o.close()
    return #init_holder


def main():
    '''
    import argparse
    parser = argparse.ArgumentParser(description="Read tweets from a json file and save in utf-8 with an identifier")
    parser.add_argument('input_json_file', help='input json file from the RUG twitter dump')
    parser.add_argument('output_prefix', help='output prefix')
    parser.add_argument('--batch_dictionary', help='If saving the tweets as objects in dict, choose dict location')
    parser.add_argument('--batch size', help='Max no of tweets before saving in the one dict')
    parser.add_argument('--percent_train', help='Percentage (as ratio relative to other params) of corpus into train file directory,t if blank then all to train')
    parser.add_argument('--percent_dev', help='Percent to dev file')
    parser.add_argument('--percent_test', help='Percent to test file')
    args =parser.parse_args()

    #TODO break this off into a separate function
    division = (args.percent_train, args.percent_dev, args.percent_test)
    if sum(division) == 0:
        division = (80, 10, 10)
    elif sum(division) != 100:
        division = [x * 100/sum(division) for x in division]
    #foo = loadtweets('medium.example', 'text.only')

    #TODO break this off into a separate function
    totaltweets = subprocess.call(['wc', '-l', args.input_json_file])
    this_split = (totaltweets // d for d in division)
'''
    foo = loadtweets('./small.example', 'text.only')
#    foo = loadtweets(args.input_json_file, args.output_path, linedivision=this_split)
    print("Read tweets successfully from {} and printed them to prefix= {}".format('something', 'something'))
    print(foo)

if __name__=="__main__":
    main()