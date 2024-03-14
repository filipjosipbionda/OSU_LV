try:
    number=(float(input("Enter a number: ")))

    if number<0 or number>1:
       raise ValueError("Number must be in range between 0 and 1.0")
    
    if number<0.6:
        print('F')
    elif number<0.7:
        print('D')
    elif number<0.8:
        print('C')
    elif number<0.9:
        print('B')
    else:
        print('A')

except ValueError as error:
    print(error)
    
