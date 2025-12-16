import re

message1 = "Ksh500 paid. Balance Ksh2000. Limit 150000"
# TODO: Find all amounts
# Hint: Use re.findall(r'Ksh(\d+)', message1)

amounts = re.findall(r'Ksh(\d+)',message1)


message2 = "on 15/12/24 at 2.30 PM"
# TODO: Find the date
# Hint: Pattern is 

date = re.findall(r'\d{1,2}/\d{1,2}/\d{2}',message2)

print("amounts",amounts)
print("date",date)