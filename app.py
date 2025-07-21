#just a simple code to clear redis memory


import redis

r = redis.Redis(host='localhost', port=6379, db=0)
r.flushall()  # or r.flushdb()
