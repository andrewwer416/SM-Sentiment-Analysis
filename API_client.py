import tweepy

def download():
    api_key = "zbVpRP8CxXjTYbxw3hWmiATSO"
    api_secret = "D9loTi3q9eKvowZ1aVlwb6reaclGUBi61yw2kQpWjXA7yFK7r0"
    bearer_token = r"AAAAAAAAAAAAAAAAAAAAABrVhgEAAAAAGUbFhNGVS59BDN9XoWP95dICDes%3DuBdMY0YJrlsv3qNGKa41PlQbITyXGtfeBh0rgGCvFkkVk7KRsQ"
    access_token = "MEpham9NWkFTbldWRHNRa1BOSzk6MTpjaQ"
    access_token_secret = "kLwtDUkVUNqbYFPv6fOQVTc15Hbt6hxRrIWNJyhNSF2VnJbE_W"
    client = tweepy.Client(bearer_token, api_key, api_secret, access_token, access_token_secret)

    query = "covid"
    auth = tweepy.OAuth1UserHandler(api_key, api_secret, access_token, access_token_secret)
    api = tweepy.API(auth)

    response = client.search_recent_tweets(query=query, max_results=100)
    print(response)