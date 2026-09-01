p = height // 2
q = width // 2

uk = 261
vk = 261

D0 = 5
n = 2

us = 2 * p - uk
vs = 2 * q - vk

H = np.ones((height, width), dtype=np.float32)

for u in range(height):
    for v in range(width):
        D1 = np.sqrt((u - uk)**2 + (v - vk)**2)
        D2 = np.sqrt((u - us)**2 + (v - vs)**2)

        H1 = 0 if D1 == 0 else 1 / (1 + (D0 / D1)**(2 * n))
        H2 = 0 if D2 == 0 else 1 / (1 + (D0 / D2)**(2 * n))

        H[u, v] = H1 * H2