"""
poverty product from NOAA is horrible (https://ngdc.noaa.gov/eog/dmsp/download_poverty.html)

Use this script in GEE to get nightlights:
var popImgSet = ee.ImageCollection('NOAA/DMSP-OLS/NIGHTTIME_LIGHTS').select('stable_lights').filterDate('2010-01-01', '2017-01-01')

var popImgmask = ee.Image(popImgSet.sort('system:index', false).first())

// Select VIIRS Nightlights images ----------------------------------
var NLImgSet = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG').select('avg_rad').filterDate('2017-01-01', '2018-10-01')

// Mask each NL image first by DMSP-OLS then to only select values greater than 0.3
function maskImage(img){
    return img.updateMask(popImgmask).updateMask(img.gte(0.3))
}
NLImgSet = NLImgSet.map(maskImage)

// from collection to 1 image
var first = ee.Image(NLImgSet.first())

function appendBand(img, previous){
        return ee.Image(previous).addBands(img)
}
var img = ee.Image(NLImgSet.iterate(appendBand, first))

// reduce (mean) pixel wise
img = img.reduce(ee.Reducer.mean())

Map.addLayer(img, {min: 0.01, max: 100, palette: ['blue', 'green', 'red']}, 'nightlights');


// EXPORT results
Export.image.toDrive({
  image: img,
  description: 'nightlights_africa_HRM',
  scale: 1000,
  region: geometry
});


this script bins them into classes for training.
"""

from osgeo import gdal
import numpy as np

raster = gdal.Open('data/nightlights_africa_HRM.tif')
nightlights = np.array(raster.GetRasterBand(1).ReadAsArray())

# nightlights = np.round(np.float64(nightlights), 2)
intervals = [1, 5, 20]
print('0: ', len(nightlights[(nightlights <= intervals[0]) | (np.isnan(nightlights))]) / len(nightlights.ravel()))
print('poor: ', len(nightlights[(nightlights > intervals[0]) & (nightlights <= intervals[1])]) )
print('medium: ', len(nightlights[(nightlights > intervals[1]) & (nightlights <= intervals[2])]))
print('rich: ', len(nightlights[nightlights > intervals[2]]))

# if need to aggregate georasters.aggregate(data,NDV,(10,10))

# Bin
nightlights[(nightlights <= intervals[0]) | (np.isnan(nightlights))] = np.nan
nightlights[(nightlights > intervals[0]) & (nightlights <= intervals[1])] = 1
nightlights[(nightlights > intervals[1]) & (nightlights <= intervals[2])] = 2
nightlights[nightlights > intervals[2]] = 3

# write out
[cols, rows] = nightlights.shape

output_raster = gdal.GetDriverByName('GTiff').Create('data/nightlights_africa_HRM_bin.tif', rows, cols, 1, gdal.GDT_Float32)  # Open the file
output_raster.SetGeoTransform(raster.GetGeoTransform())  # Specify its coordinates
output_raster.SetProjection(raster.GetProjection())
output_raster.GetRasterBand(1).SetNoDataValue(-99)

output_raster.GetRasterBand(1).WriteArray(nightlights)  # Writes my array to the raster
output_raster.FlushCache()  # saves to disk!!