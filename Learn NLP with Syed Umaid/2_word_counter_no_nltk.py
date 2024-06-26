
wordstring = 'it was the best of times it was the worst of times'

wordstring += 'it was the age of wisdom it was the age of foolishness'

wordlist = wordstring.split()

# wordfreq = []
# for w in wordlist:
#     wordfreq.append(wordlist.count(w))

# If we want it without the need of for loop using List Comprehension
wordfreq = [wordlist.count(w) for w in wordlist]




print("String\n" + wordstring +"\n")
print("List\n" + str(wordlist) + "\n")
print("Frequencies\n" + str(wordfreq) + "\n")
print("Pairs\n" + str(list(zip(wordlist, wordfreq))))