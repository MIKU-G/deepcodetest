import numpy as np
import cv2
from skimage.measure import block_reduce


class GaborFeatureExtractor:
# initiate Gabor feature extractor
# how many rotations are taken into consider
# ksize = the largest kernel size of Gabor filter
# feat_type: 'mean' or 'max'; the method to compress feature from the response map after Gabor filtering
# feat_ksize: size of a square kernel used to compress the feature
    def __init__(self, ksize, numTheta, feat_type, feat_ksize=3):
        self.ksize = ksize
        self.numTheta = numTheta
        self.feat_type = feat_type
        self.feat_ksize = feat_ksize

    def build_filters(self):
        filters = []
        for ksize in np.arange(5, self.ksize+4, 4):
            for theta in np.arange(0, np.pi, np.pi /self.numTheta):
                kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
                kern /= 1.5 * kern.sum()
                filters.append(kern)
        return filters

    # filter a single
    def process(self, img, filters):
        accum = np.zeros_like(img)
        for kern in filters:
            fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
            np.maximum(accum, fimg, accum)
        return accum

    # extract feature
    # Note: the length of feature vector is equal to the number of filters
    def getGabor(self, img, filters):
        res = []
        for i in range(len(filters)):
            res1 = self.process(img, filters[i])
            res.append(np.asarray(res1))
        return np.array(res)


    # compress the feature tensor according to the given argument
    def getFeature(self, res):
        if self.feat_type == 'mean':
            re_img = block_reduce(res, block_size=(self.feat_ksize, self.feat_ksize, 1), func=np.mean)
        elif self.feat_type == 'max':
            re_img = block_reduce(res, block_size=(self.feat_ksize, self.feat_ksize, 1), func=np.max)
        else:
            print('ERROR: Invalid feature type input!')
        feature = np.reshape(re_img, (1, -1))
        return feature


    # Input: a list containing images
    # Output: a 3D numpy array containing filtered feature images
    def batchProcess(self, imgs, filters):
        features = []
        for idx in range(imgs.shape[0]):
            img = imgs[idx, :, :, :]
