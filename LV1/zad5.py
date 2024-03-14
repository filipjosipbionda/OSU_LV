def countWords(messages):
    wordCount=0
    for message in messages:
        for word in message:
            wordCount+=1

    return wordCount


def endsWithExMark(message):
    return message[-1]=='!'


def getWordsEndingWithExMark(messages):
    wordsCount=0
    for message in messages:
        if endsWithExMark(message):
            wordsCount+=1

    return wordsCount


ham=[]
spam=[]

smsFile=open("SMSSpamCollection.txt")

for line in smsFile:
    line=line.rstrip()
    parts=line.split("\t")

    if parts[0]=="ham":
        ham.append(parts[1])
    elif parts[0]=="spam":
        spam.append(parts[1])

smsFile.close()
    
print(f"Average spam messages: {countWords(spam)/len(spam)}")
print(f"Average ham messages: {countWords(ham)/len(ham)}")
print(f"Number of spam messages ending with !: {getWordsEndingWithExMark(spam)}")

