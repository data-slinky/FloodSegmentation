'''
Performs segmentation of images
'''

from skimage.transform import hough_line, hough_line_peaks
from skimage import color, feature, filters, segmentation
from skimage.segmentation import mark_boundaries
import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib import colors

# Caffe is not in the system path so we have add it everytime
caffe_root = '/Users/johnkimnguyen/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe

# Use CPU only more for Caffe
caffe.set_mode_cpu()

def SLIC(image, k=2, c=0.1, sigma=0, slico=False):
    '''
    Wrapper to perform SLIC

    Parameters:
        image: The image to be read
        k: Number of clusters
        c: Set compactness, the weighing between color and space
        sigma: Smoothing value for Gaussian
        show: boolean to show the plot or not

    Return:
        segments: The
    '''
    # Segment the image
    segments = segmentation.slic(image, n_segments=k, compactness=c, multichannel=True, slic_zero=slico, sigma=sigma)

    return segments

def plot_segments(path, image, segments, draw=False):
    '''
    Utility function to generate the plots

    Parameters:
        path: Path to be saved
        image: The image to be processed
        segments: Results of SLIC()
        draw: Whether to plot the boundaries on the original image
    '''
    two_map = colors.ListedColormap(['black', (0.98836199999999996, 0.99836400000000003, 0.64492400000000005)])
    bounds = [0, 0.1, 1]
    norm = colors.BoundaryNorm(bounds, two_map.N)

    height = float(image.shape[0])
    width = float(image.shape[1])
    if draw:
        fig = plt.figure()
        fig.set_size_inches(width / height, 1, forward=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        plt.imshow(mark_boundaries(image, seg, color=(1, 0, 0), mode='thick'))
        fig.savefig(path, format='png', dpi=height)
        plt.close(fig)
    else:
        fig = plt.figure()
        fig.set_size_inches(width / height, 1, forward=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        plt.imshow(segments, cmap=two_map, vmin=0, vmax=1)
        plt.savefig(path, format='png', dpi=height)
        plt.close()

def Quickshift(image, ratio=1, kernel_size=1, convert2lab=True, show=False):
    '''
    Read in an image and get the peak Hough line

    Parameters:
        image: The image to be read
        k: Number of clusters
        c: Set compactness, the weighing between color and space
        sigma: Smoothing value for Gaussian
        show: boolean to show the plot or not

    Return:
        segments: The segments of Quickshift
    '''
    # Segment the image
    segments = segmentation.quickshift(image, ratio=ratio, kernel_size=kernel_size)

    if (show):
        #plt.imshow(segments)
        plt.imshow(mark_boundaries(image, segments, color=(1, 0, 0), mode='thick'))
        plt.draw()
        plt.savefig("%d quickshift" % image, format='eps', dpi=300, bbox_inches='tight')
    return segments

def processPrototxt(image, layer=8):
    '''
    Edit the prototxt file to change the image dimension. This requires to load the FCN every step.
    There is a smarter way to do with with reshape (look into later)

    Parameters:
        image: The image to be read
        layer: select fcn8s, fcn16s or fcn32s
    '''
    if layer == 8:
        deploy = 'Models/test8.prototxt'
    elif layer == 16:
        deploy = 'Models/test16.prototxt'
    elif layer == 32:
        deploy = 'Models/test32.prototxt'
    else:
        raise 'pick either 8, 16 or 32!'

    print 'processing ' + deploy + ' before loading!'

    with open(deploy, "r") as f:
        data = f.readlines()
        data[4] = 'input_dim: ' + str(image.shape[0]) + '\n'
        data[5] = 'input_dim: ' + str(image.shape[1]) + '\n'
    with open(deploy, 'w') as f:
        f.writelines(data)

def load_FCN(layer=8):
    '''
    Wrapper to load the pre-trained FCN on Pascal Context

    Parameters:
        layer: select fcn8s, fcn16s or fcn32s

    Return:
        net: the pre-trained FCN with weights
    '''
    if layer == 8:
        weights = 'Models/pascalcontext-fcn8s-heavy.caffemodel'
        deploy = 'Models/test8.prototxt'
    elif layer == 16:
        weights = 'Models/pascalcontext-fcn16s-heavy.caffemodel'
        deploy = 'Models/test16.prototxt'
    elif layer == 32:
        weights = 'Models/pascalcontext-fcn32s-heavy.caffemodel'
        deploy = 'Models/test32.prototxt'
    else:
        raise 'pick either 8, 16 or 32!'
    net = caffe.Net(deploy, weights, caffe.TEST)
    print 'finished loading CNN!'
    return net

def get_transformer(net):
    '''
    Obtain a transformer instance to get the image mean
        Parameters:
            net: the FCN network

        Return:
            transformer: caffe transformer instance
    '''

    # Load the image and obtain the mean image
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_mean('data',
                         np.load('/Users/johnkimnguyen/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(
                             1))
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_channel_swap('data', (2, 1, 0))
    transformer.set_raw_scale('data', 255.0)
    mean = np.load('/Users/johnkimnguyen/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)
    transformer.set_mean('data', mean)
    print 'obtained transformer!'

    return transformer

def process_FCN(image, net, transformer):
    '''
    Wrapper to load the pre-trained FCN on Pascal Context

    Parameters:
        image: path to the image
        n: label number corresponding to the object
        net: the model chosen from load_FCN()
        transf: caffe transformer instance

    Return:
        result: A 2D numpy array consisting of the scores for the water label 57
        transf: transformer of the image
    '''
    img = caffe.io.load_image(image)
    print 'loaded the image!'

    # net.blobs['data'].reshape(1,int(img.shape[2]),int(img.shape[1]),int(img.shape[0]))
    #net.blobs['label'].reshape(1, )
    # net.reshape()
    # print 'changed blob data shape with image shape!'

    # Push the image through the CNN
    out = net.forward_all(data=np.asarray([transformer.preprocess('data', img)]))
    print 'finished forwarding!'

    # Obtain the water score (57 is water and 7 is car)
    # result = out['score'][0, n]
    # print 'finished obtaining water score!'

    return out['score']

def plot_FCN_water(path, image, score, net, transformer, p=0, normalize=False, draw=False):
    '''
    Utility function to generate the plots

    Parameters:
        score: score obtained from process_FCN()
        net: the model chosen from load_FCN()
        transformer: Caffe transformer instance
        p: percentile [0,100] if not normalize | if p = 0, use mean
        normalize: boolean to normalize or not
        draw: Whether to plot the boundaries on the original image

    Return:
        water_score: segments
    '''
    two_map = colors.ListedColormap(['black', (0.98836199999999996, 0.99836400000000003, 0.64492400000000005)])
    bounds = [0, 0.1, 1]
    norm = colors.BoundaryNorm(bounds, two_map.N)

    if normalize:
        from sklearn.preprocessing import normalize
        water_score = normalize(score, axis=0, norm='l1')

    from copy import deepcopy
    water_score = deepcopy(score)

    if p == 0:
        water_score[water_score < water_score.mean()] = 0
        water_score[water_score >= water_score.mean()] = 1
    else:
        water_score[water_score < np.percentile(water_score, p, axis=0)] = 0
        water_score[water_score >= np.percentile(water_score, p, axis=0)] = 1

    if draw:
        plt.ioff()
        fig = plt.figure()
        plt.subplot(1, 3, 1)
        plt.axis('off')
        plt.imshow(transformer.deprocess('data', net.blobs['data'].data[0]))
        plt.subplot(1, 3, 2)
        plt.axis('off')
        plt.imshow(water_score, cmap='rainbow', vmin=0, vmax=1)  # label 57 is water, label 8 is cat
        plt.subplot(1, 3, 3)
        plt.axis('off')
        plt.imshow(mark_boundaries(image, water_score, color=(1, 0, 0), mode='thick'))
        plt.axis('off')
        plt.savefig(path, format='png', dpi=200, bbox_inches='tight')
        plt.close(fig)
    else:
        height = float(image.shape[0])
        width = float(image.shape[1])
        fig = plt.figure()
        fig.set_size_inches(width / height, 1, forward=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        plt.imshow(water_score, cmap='rainbow', vmin=0, vmax=1)
        plt.savefig(path, format='png', dpi=height)
        plt.close()

def plot_FCN_car(path, image, score, net, transformer, p=0, normalize=False, draw=False):
    '''
    Utility function to generate the plots

    Parameters:
        score: score obtained from process_FCN()
        net: the model chosen from load_FCN()
        transformer: Caffe transformer instance
        p: percentile [0,100] if not normalize | if p = 0, use mean
        normalize: boolean to normalize or not
        draw: Whether to plot the boundaries on the original image

    Return:
        water_score: segments
    '''
    two_map = colors.ListedColormap(['black', 'purple'])
    bounds = [0, 0.1, 1]
    norm = colors.BoundaryNorm(bounds, two_map.N)

    from copy import deepcopy
    car_score = deepcopy(score)

    if normalize:
        from sklearn.preprocessing import normalize
        car_score = normalize(score, axis=0, norm='l1')

    if p == 0:
        car_score[car_score < car_score.max()] = 0
        car_score[car_score >= car_score.max()] = 1
    else:
        car_score[car_score < np.percentile(car_score, p, axis=0)] = 0
        car_score[car_score >= np.percentile(car_score, p, axis=0)] = 1

    if draw:
        plt.ioff()
        fig = plt.figure()
        plt.subplot(1, 3, 1)
        plt.axis('off')
        plt.imshow(transformer.deprocess('data', net.blobs['data'].data[0]))
        plt.subplot(1, 3, 2)
        plt.axis('off')
        plt.imshow(car_score, cmap=two_map, vmin=0, vmax=1)  # label 57 is water, label 8 is cat
        plt.subplot(1, 3, 3)
        plt.axis('off')
        plt.imshow(mark_boundaries(image, car_score, color=(1, 0, 0), mode='thick'))
        plt.axis('off')
        plt.savefig(path, format='png', dpi=200, bbox_inches='tight')
        plt.close(fig)
    else:
        height = float(image.shape[0])
        width = float(image.shape[1])
        fig = plt.figure()
        fig.set_size_inches(width / height, 1, forward=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        plt.imshow(car_score, cmap='rainbow')
        plt.savefig(path, format='png', dpi=height)
        plt.close()

def Hough(image, filter='Canny', sigma=3, show=False):
    '''
    Read in an image and get the peak Hough line

    Parameters:
        image: The image to be read
        filter: Set whether to use Canny (default) or Sobel edge detection
        sigma: Sensitivity for Canny
        show: boolean to show the plot Hough plot

    Return:
        max_peak: The horizon line corresponding to the max peak
        edges: the edges
    '''
    img_gray = color.rgb2gray(image) # Grayscale

    if (filter == 'Canny'):
        # Edge detection
        edges = feature.canny(img_gray, sigma=3) # Canny
    if (filter == 'Sobel'):
        edges = filters.sobel(img_gray) # Use Sobel

    # Hough transformation
    h, theta, d = hough_line(edges)

    # Get the max_peak
    max_peak = hough_line_peaks(h, theta, d, num_peaks=1)

    if(show):
        fig = plt.figure()
        plt.imshow(np.log(1 + h), extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), d[-1], d[0]],
                   cmap=plt.cm.gray, aspect=1 / 1.5)

    return max_peak, edges

def Hough_plot(path,max_peak, image, edges, draw=False):
    height = float(image.shape[0])
    width = float(image.shape[1])
    if draw:
        fig = plt.figure()
        fig.set_size_inches(width / height, 1, forward=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        plt.imshow(image)
        for _, angle, dist in zip(*max_peak):
            y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
            y1 = (dist - width * np.cos(angle)) / np.sin(angle)
            plt.plot((0, width), (y0, y1), '-r')
        plt.axis((0, width, height, 0))
        plt.savefig(path, format='png', dpi=height)
        plt.close()
    else:
        blank_img = np.zeros([int(height), int(width), 3], dtype=np.uint8)
        fig = plt.figure()
        fig.set_size_inches(width / height, 1, forward=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        plt.imshow(blank_img)
        for _, angle, dist in zip(*max_peak):
            y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
            y1 = (dist - width * np.cos(angle)) / np.sin(angle)
            plt.fill_between(x=(0, width), y1=(y0,y1), y2=(height, height), color=(0.98836199999999996, 0.99836400000000003, 0.64492400000000005))
        plt.axis((0, width, height, 0))
        plt.savefig(path, format='png', dpi=height)
        plt.close()


def visualize_weights(net, layer_name, padding=4, filename=''):
    # The parameters are a list of [weights, biases]
    data = np.copy(net.params[layer_name][0].data)
    # N is the total number of convolutions
    N = data.shape[0] * data.shape[1]
    # Ensure the resulting image is square
    filters_per_row = int(np.ceil(np.sqrt(N)))
    # Assume the filters are square
    filter_size = data.shape[2]
    # Size of the result image including padding
    result_size = filters_per_row * (filter_size + padding) - padding
    # Initialize result image to all zeros
    result = np.zeros((result_size, result_size))

    # Tile the filters into the result image
    filter_x = 0
    filter_y = 0
    for n in range(data.shape[0]):
        for c in range(data.shape[1]):
            if filter_x == filters_per_row:
                filter_y += 1
                filter_x = 0
            for i in range(filter_size):
                for j in range(filter_size):
                    result[filter_y * (filter_size + padding) + i, filter_x * (filter_size + padding) + j] = data[
                        n, c, i, j]
            filter_x += 1

    # Normalize image to 0-1
    min = result.min()
    max = result.max()
    result = (result - min) / (max - min)

    # Plot figure
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(result, cmap='gray', interpolation='nearest')

    # Save plot if filename is set
    if filename != '':
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.show()