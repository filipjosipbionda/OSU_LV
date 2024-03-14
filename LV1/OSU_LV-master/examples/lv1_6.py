fruit = 'banana'
index = 0
count = 0
while index < len ( fruit ):
    letter = fruit [ index ]
    if letter == 'a':
        count = count + 1

    print ( letter )
    index = index + 1

print ( count )
print ( fruit [0:3])
print ( fruit [0: ])
print ( fruit [2:6:1])
print ( fruit [0:-1])


line = 'Dobrodosli u nas grad'

if(line.startswith('Dobrodosli')):
    print('Prva rijec je Dobrodosli')
elif ( line . startswith('dobrodosli')):
    print ('Prva rijec je dobrodosli')

line=line.lower()
print ( line )
data = 'From : pero@yahoo . com '
atpos = data . find ('@')
print ( atpos )