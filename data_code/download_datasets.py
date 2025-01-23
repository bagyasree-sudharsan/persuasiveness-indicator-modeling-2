from convokit import Corpus, download

scoa_corpus = Corpus(filename=download("supreme-corpus"))
# cmv_corpus = Corpus(filename=download("winning-args-corpus"))

print('Downloaded SCOA and CMV corpora.')