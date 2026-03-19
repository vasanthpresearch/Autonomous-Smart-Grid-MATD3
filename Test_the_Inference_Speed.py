import time
start_time = time.time()
with torch.no_grad():
    action = actor(torch.randn(1, 24)) # Dummy state input
end_time = time.time()
print(f"Inference Latency: {(end_time - start_time)*1000:.4f} ms")