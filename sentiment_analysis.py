import sentiment_mod as s

test_data = open("testdata.txt","r").read()
test_data = test_data.split("\n")
for tweet in test_data:
    sentiment_value, confidence = s.sentiment(tweet)
    print(tweet, sentiment_value, confidence)





    
