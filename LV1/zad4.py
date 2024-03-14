wordsInSong={}

songFile=open('song.txt')

for line in songFile:
    line=line.rstrip()
    words=line.split(" ")
    for word in words:
        if word not in wordsInSong:
            wordsInSong[word]=1
            continue
        wordsInSong[word]+=1

uniqueWords=0
for word in wordsInSong:
    if wordsInSong[word]==1:
        uniqueWords+=1
        print(f"{word}:{wordsInSong[word]}")

print(uniqueWords)

