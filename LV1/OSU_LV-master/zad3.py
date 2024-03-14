numbers=[]

while(True):
    try:
        number=(input("Enter a number: "))
        if number=="Done":
            break
        number=float(number)
        numbers.append(number)
    except:
        print("A number must be entered!")


numbers.sort()
print(numbers)
print(len(numbers))
print(min(numbers))
print(max(numbers))


