#step 1: Imports
import pandas as pd

#step 2: load one message
df = pd.read_csv('data/raw/mpesa_sms_messages.csv')
first_message = df.iloc[0]

print ("message sender:", first_message['sender_id'])

#step 3: Check if the sender is valid
#Real senders always use MPESA as sender

valid_senders = ['MPESA']

if first_message['sender_id'] in valid_senders:
    is_valid = 1
else:
    is_valid = 0

print ("Is valid sender?", is_valid)