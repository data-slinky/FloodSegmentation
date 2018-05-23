import Segmentation as seg
from skimage import io
import pandas as pd

# Use for quick testing
# filepath = "Test_Data/test_image23.jpg"
# name = os.path.splitext(os.path.basename(filepath))[0]
# image = io.imread(filepath)
# max_peak, edges = seg.Hough(image)
# Hough_plot('Combine/test.png', max_peak, image, edges, draw=False)
#seg = seg.SLIC(image, k=2, c=1)
#seg.plot_segments('SLIC_figures/test.png', image, seg, draw=True)

# plt.subplot(1, 3, 1)
# plt.imshow(image)
# plt.axis('off')
# plt.subplot(1, 3, 2)
# plt.imshow(seg, cmap=two_map)
# plt.axis('off')
# plt.subplot(1, 3, 3)
# plt.imshow(mark_boundaries(image, seg, color=(1, 0, 0), mode='thick'))
# plt.axis('off')

# # Process data using SLIC
# for filename in os.listdir('Test_Data2/'):
#     if filename.endswith(".jpg"):
#         print('Processing file: ' + filename)
#         filepath = os.path.join('Test_Data2/', filename)
#         image = io.imread(filepath)
#         # image = transform.resize(image, (500,400)) # Resize the image, note this screws up results
#         seg = seg.SLIC(image, k=2, c=1, slico=False)
#         # df = pd.DataFrame(seg)
#         # df.to_csv('SLIC_segments/' + filename.replace(".jpg", "") + '.csv', sep=',', header= False, index=False)
#         seg.plot_segments('SLIC_figures/' + filename.replace("jpg", "png"), image, seg)


# ## Process data using Hough
# for filename in os.listdir('Test_Data2/'):
#     if filename.endswith(".jpg"):
#         print('Processing file: ' + filename)
#         filepath = os.path.join('Test_Data2/', filename)
#         image = io.imread(filepath)
#         max_peak, edges = seg.Hough(image)
#         seg.Hough_plot('Hough_figures/' + filename.replace("jpg", "png"), max_peak, image, edges)

# Process data using FCN
filepath = "Test_Data/test_image15.jpg"
image = io.imread(filepath)
seg.processPrototxt(image, 8)
net = seg.load_FCN(layer=8)
# net = load_FCN(layer=16)
# net = load_FCN(layer=32)
transformer = seg.get_transformer(net)
result8 = seg.process_FCN(filepath, net, transformer)
# result16 = process_FCN(filepath, net16, transformer)
# result32 = process_FCN(filepath, net32, transformer)
# image = transform.resize(image, (500,400))
seg.plot_FCN_water('Combine/' + "water_seg.png", image, result8[0,57], net, transformer, p=0, normalize=False)
seg.plot_FCN_water('Combine/' + "car_seg.png", image, result8[0,2], net, transformer, p=0, normalize=False)

df = pd.DataFrame(result8[0,55])
df.to_csv('Combine/result8.csv', sep=',', header=False, index=False)

# for filename in os.listdir('Test_Data2/'):
#     if filename.endswith(".jpg"):
#         print('Processing file: ' + filename)
#         filepath = os.path.join('Test_Data2/', filename)
#         image = io.imread(filepath)
#         seg.processPrototxt(image, 8)
#         net = seg.load_FCN(layer=8)
#         transformer = seg.get_transformer(net)
#         # image = transform.resize(image, (500,400)) # Resize the image, note this screws up results
#         result8 = seg.process_FCN(filepath, net, transformer)
#         seg.plot_FCN_water('FCN8_figures/' + filename.replace("jpg", "png"), image, result8[0,57], net, transformer, p=0, normalize=False)

visualize_weights(net, 'conv1_1', filename='conv1.png')
visualize_weights(net, 'conv2', filename='conv2.png')