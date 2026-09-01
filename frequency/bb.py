p = height // 2
q = width // 2

D0 = 50
W = 10
n = 2

H = np.ones((height, width), dtype=np.float32)

for u in range(height):
    for v in range(width):
        D = np.sqrt((u - p)**2 + (v - q)**2)

        if D**2 == D0**2:
            H[u, v] = 0
        else:
            H[u, v] = 1 / (1 + ((D * W) / (D**2 - D0**2))**(2 * n))