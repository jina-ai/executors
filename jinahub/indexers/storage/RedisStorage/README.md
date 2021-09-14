# RedisStorage

**RedisStorage** is Indexer wrapper around the redis server. Redis is an open source (BSD licensed), in-memory data structure store, used as a database, cache, and message broker. You can read more about it here: https://redis.io




## Prerequisites

Additionally, you will need a running redis server. This can be a local instance, a Docker image, or a virtual machine in the cloud. To connect to redis, you need the hostname and the port. 

You can start one in a Docker container, like so: 

```bash
docker run -p 127.0.0.1:6379:6379/tcp -d redis
```

📕 **Note on docker network for macOS users**:  
If you run both the database and the `RedisStorage` docker container on the same machine 
localhost in the `RedisStorage` resolves to a separate network created by Docker which cannot see the database running on the host network.  
Use `host.docker.internal` to access localhost on the host machine.  
You can pass this parameter to the `RedisStorage` storage by using `uses_with={'hostname': 'host.docker.internal''}` when
calling the `flow.add(...)` function.

This indexer does not allow indexing two documents with the same `ID` and will issue a warning. It also does not allow updating a document by a non-existing ID and will issue a warning.

## Reference

- https://redis.io
