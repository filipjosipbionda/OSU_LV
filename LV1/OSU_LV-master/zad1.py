def total_euro(hours,payRate):
    return hours*payRate

hours=int(input("Working hours: "))
payRate=float(input("Pay rate: "))

print(total_euro(hours,payRate))