p = height // 2
q = width // 2

D0 = 30
n = 2

H = np.zeros((height, width), dtype=np.float32)

for u in range(height):
    for v in range(width):
        D = np.sqrt((u - p)**2 + (v - q)**2)

        if D == 0:
            H[u, v] = 0
        else:
            H[u, v] = 1 / (1 + (D0 / D)**(2 * n))