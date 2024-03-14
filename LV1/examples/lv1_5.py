lstEmpty = [ ]
lstFriend = ['Marko','Luka', 'Pero']
lstFriend.append('Ivan')
print(lstFriend[0])
print(lstFriend[0:1:2])
print(lstFriend[ :2])
print(lstFriend[1: ])
print(lstFriend[1:3])

a = [1 , 2 , 3]
b = [4 , 5 , 6]
c = a + b
print ( c )
print(c[0:-1])

print ( max ( c) )
c[0] = 7
c . pop ()
for number in c:
    print('List number' , number )
print ('Done !')