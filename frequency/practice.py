def histogram_match(source, reference):

    hist_s = histogram(source)
    hist_r = histogram(reference)

    cdf_s = get_cdf(hist_s)
    cdf_r = get_cdf(hist_r)

    mapping = np.zeros(256, dtype=np.uint8)

    for i in range(256):

        min_diff = abs(cdf_s[i] - cdf_r[0])
        best_j = 0

        for j in range(1, 256):

            diff = abs(cdf_s[i] - cdf_r[j])

            if diff < min_diff:
                min_diff = diff
                best_j = j

        mapping[i] = best_j

    h, w = source.shape
    output = np.zeros_like(source)

    for i in range(h):
        for j in range(w):
            output[i, j] = mapping[source[i, j]]

    return output

def equalize(image):
    height, width = image.shape
    histogram = np.zeros(255)
    pdf = np.zeros(255)
    cdf = np.zeros(255)
    mapping = np.zeros(255)

    for i in range(height):
        for j in range(width):
            histogram[image[i,j]]+=1

    total = 0
    for i in range(255):
        pdf[i] = histogram[i]/(height*width)
        total+=pdf[i]
        cdf[i] = total
        mapping[i] = int(cdf[i]*255)

    equalized = np.zeros_like(image, dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            old = image[i,j]
            equalized[i,j] = mapping[old]
    return equalized

def match(image, reference):
    #hist1, hist2 from function
    cdf1 = get_cdf(image)
    cdf2 = get_cdf(reference)
    mapping = np.zeros(256, dtype=np.uint8)
    height1, width1 = image.shape

    for r in range(256):
        best_z = 0
        min_dist = abs(cdf1[r]-cdf2[0])
        for z in range(1,256):
            dist = abs(cdf1[r]-cdf2[z])
            if dist < min_dist:
                min_dist = dist
                best_z = z
        mapping[r] = best_z
    matched = np.zeros_like(image, dtype=np.uint8)
    for i in range(height1):
        for j in range(width1):
            old = image[i,j]
            matched[i,j] = mapping[old]
    return matched



    
