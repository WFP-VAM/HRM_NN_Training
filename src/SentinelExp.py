# contains the routines for downlaoding images from Sentinel 2 through the earth engine API


def sentinel_downlaoder(lat, lon, start_date, end_date):

    def squaretogeojson(lon,lat,d):
        """
        Given a lat lon it returns a polygon.
        :param lon: longitude, float
        :param lat: latitude, float
        :param d: distance around point (in meters)
        :return: Polygon in Geojson fromat
        """
        from math import pi,cos
        from geojson import Polygon
        r_earth=6378000
        minx = lon - ((d/2) / r_earth) * (180 / pi)
        miny = lat - ((d/2) / r_earth) * (180 / pi) / cos(lon * pi/180)
        maxx = lon + ((d/2) / r_earth) * (180 / pi)
        maxy = lat + ((d/2) / r_earth) * (180 / pi) / cos(lon * pi/180)
        #return minx,miny,maxx,maxy
        square = Polygon([[(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)]])
        return square

    def gee_url(geojson,start_date,end_date):
        """
        gets the url for a specific sentinel image given the polygon.
        :param geojson: polygon
        :param start_date: str, date from for satellite capture
        :param end_date: str, date enf for satellite capture
        :return: url
        """
        import ee
        ee.Initialize()

        lock = 0  # if it doesnt find image, increase cloud cover while loop.
        cloud_cover = 10
        while lock == 0:  # band B8 is the NIR
            sentinel = ee.ImageCollection('COPERNICUS/S2') \
                    .filterDate(start_date, end_date) \
                    .select('B2', 'B3', 'B4') \
                    .filterBounds(geojson) \
                    .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', cloud_cover) \
                    .sort('GENERATION_TIME') \
                    .sort('CLOUDY_PIXEL_PERCENTAGE', False) #also sort by date

            # check if there are images, otherwise increase accepteable cloud cover
            collectionList = sentinel.toList(sentinel.size())

            try:  # if it has zero images this line will return an EEException
                collectionList.size().getInfo()
                lock = 1
            except ee.ee_exception.EEException:
                print('found no images with {} cloud cover. Going to 10+'.format(cloud_cover))
                cloud_cover = cloud_cover + 10

        image1 = sentinel.mosaic()
        path = image1.getDownloadUrl({
            'scale': 10,
            'crs': 'EPSG:4326',
            'region':geojson
        })
        return path

    d = 5000
    geojson = squaretogeojson(lon, lat, d)
    return gee_url(geojson, str(start_date), str(end_date))


def download_and_unzip(buffer, path):
    """
    given what is returned from the request to ee api, saves the tifs.
    """
    unzipped=[]
    from zipfile import ZipFile

    zip_file = ZipFile(buffer)
    files = zip_file.namelist()
    for i in range(3, 6):
        try:
            zip_file.extract(files[i], path+"/tiff/")
        except IndexError:
            raise 'the url from ee returned no image!'

        unzipped.append(files[i])

    return unzipped


def rgbtiffstojpg(files, path, name):
    '''
    files: a list of files ordered as follow. 0: Blue Band 1: Green Band 2: Red Band
    path: the path to look for the tiff files and to save the jpg
    '''
    import scipy.misc as sm
    import gdal
    import numpy as np
    b2_link = gdal.Open(path+"/tiff/"+files[0])
    b3_link = gdal.Open(path+"/tiff/"+files[1])
    b4_link = gdal.Open(path+"/tiff/"+files[2])

    # call the norm function on each band as array converted to float
    def norm(band):
        band_min, band_max = band.min(), band.max()
        return ((band - band_min) / (band_max - band_min))

    b2 = norm(b2_link.ReadAsArray().astype(np.float))
    b3 = norm(b3_link.ReadAsArray().astype(np.float))
    b4 = norm(b4_link.ReadAsArray().astype(np.float))

    # Create RGB
    rgb = np.dstack((b4, b3, b2))
    del b2, b3, b4
    sm.toimage(rgb, cmin=np.percentile(rgb,2), cmax=np.percentile(rgb,98)).save(path+'/images/'+name)