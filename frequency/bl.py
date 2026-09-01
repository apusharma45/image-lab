p = height // 2
q = width // 2

D0 = 30
n = 2

H = np.zeros((height, width), dtype=np.float32)

for u in range(height):
    for v in range(width):
        D = np.sqrt((u - p)**2 + (v - q)**2)
        H[u, v] = 1 / (1 + (D / D0)**(2 * n))