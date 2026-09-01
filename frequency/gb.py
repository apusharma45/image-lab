p = height // 2
q = width // 2

D0 = 50
W = 10

H = np.ones((height, width), dtype=np.float32)

for u in range(height):
    for v in range(width):
        D = np.sqrt((u - p)**2 + (v - q)**2)

        if D != 0:
            H[u, v] = 1 - np.exp(-((D**2 - D0**2) / (D * W))**2)