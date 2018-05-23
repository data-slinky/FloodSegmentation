filepath = "Test_Data/test_image23.jpg"
name = os.path.splitext(os.path.basename(filepath))[0]
image = io.imread(filepath)
image = color.rgb2grey(image)


h, theta, d = hough_line(image)

fig = plt.subplots()

plt.imshow(image, cmap=plt.cm.gray)
rows, cols = image.shape
for _, angle, dist in zip(*hough_line_peaks(h, theta, d, num_peaks=1)):
    y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
    y1 = (dist - cols * np.cos(angle)) / np.sin(angle)
    plt.plot((0, cols), (y0, y1), '-r')
plt.axis((0, cols, rows, 0))