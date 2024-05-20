import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt

text="Quaid-e-Azam is the founder of Pakistan resting in Karachi and Allama Iqbal is the father of the Ideology of Pakistan resting in Lahore"

text_tokens = word_tokenize(text)
print("Tokens from Word Tokenize are: ",text_tokens)

fq = FreqDist(token.lower() for token in text_tokens)
print("Frequency Distributions: ", fq, fq.most_common(10))

print("The plot of distribution")
fq.plot()
plt.show()


