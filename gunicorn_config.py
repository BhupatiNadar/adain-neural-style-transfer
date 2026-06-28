import multiprocessing
     
workers = 1
worker_class = "uvicorn.workers.UvicornWorker"  # lighter than sync
worker_connections = 10
max_requests = 100
max_requests_jitter = 10
timeout = 30