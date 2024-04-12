from osgeo import gdal
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np


class GFFBTreeSpecies():
    site3_filename = './data/sg_quac_test2.tif'
    site3_gr_filename = './data/sg_quac_test2_gt.tif'
    site3_gr_classes_filename = './palette/GFFB_palette.csv'

    def __init__(self, width):
        self.raster_data = gdal.Open(GFFBTreeSpecies.site3_filename)
        self.image = self.scale()
        self.ground_truth = gdal.Open(GFFBTreeSpecies.site3_gr_filename)
        self.class_df = pd.read_csv(GFFBTreeSpecies.site3_gr_classes_filename)
        margin = int((width - 1) / 2)
        self.padded_data = self.padWithZeros(margin)
        self.colorList = self.get_colors()

    def scale(self):
        scaler = StandardScaler()
        raster_array = self.raster_data.ReadAsArray()
        b, h, w = raster_array.shape
        raster_array = scaler.fit_transform(raster_array.reshape((b, h * w)).T)
        return raster_array.reshape(h, w, b)

    def padWithZeros(self, margin=2):
        newX = np.zeros((self.image.shape[0] + 2 * margin, self.image.shape[1] + 2 * margin, self.image.shape[2]))
        x_offset = margin
        y_offset = margin
        newX[x_offset:self.image.shape[0] + x_offset, y_offset:self.image.shape[1] + y_offset, :] = np.copy(self.image)
        return newX

    def getPatches(self, indexes, windowSize=5):
        patchData = np.zeros([len(indexes), windowSize, windowSize, self.image.shape[2]], dtype=np.float32)
        for i, index in enumerate(indexes):
            row = int(index / self.image.shape[1])
            col = int(index - row * self.image.shape[1])
            patchData[i] = self.padded_data[row:row + windowSize, col:col + windowSize, :]
        return patchData

    def get_image_size(self):
        return self.image.shape[0], self.image.shape[1], self.image.shape[2]

    def get_colors(self):
        numberList = self.class_df['Number'].values
        colorList = np.zeros([len(numberList), 3], dtype=int)
        for i in numberList:
            row = self.class_df[self.class_df['Number'] == i]
            colorList[i, 0] = row['R'].values[0]
            colorList[i, 1] = row['G'].values[0]
            colorList[i, 2] = row['B'].values[0]
        return colorList

    def get_color_by_index(self, index):
        index = int(index) + 1
        return self.colorList[index, 0], self.colorList[index, 1], self.colorList[index, 2]