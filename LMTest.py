__author__ = 'Kristy'

from lib.LanguageModel import *
sent = ["This is sentence one.".split(" "), ["This is sentence 2".split(" ")], ["Lastword"]]
#print(LanguageModel.bookend_list(sent))
b =  LanguageModel.flatten_list(sent)
print("output:", b)

print(b)


c = LanguageModel.duplicate_to_n_grams(2, b, delete_sss=True)
print("duplicated", c)

d = LanguageModel.bookend_and_flatten(sent,2)
print("bookend", d)

a = LanguageModel(sent, 3)
print(a)
print(a.grams)

