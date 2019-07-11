// settlements layer from JRC: https://developers.google.com/earth-engine/datasets/catalog/JRC_GHSL_P2016_SMOD_POP_GLOBE_V1
var settlements = ee.ImageCollection("JRC/GHSL/P2016/SMOD_POP_GLOBE_V1").
                  filter(ee.Filter.date('2015-01-01', '2016-12-31')).select("smod_code");

settlements = settlements.reduce(ee.Reducer.median());

var settlements_mask = settlements.gte(2)

Map.addLayer(settlements_mask, {min: 0, max: 3, palette: ['blue', 'green', 'red']}, 'Degree of Urbanization');

// Select VIIRS Nightlights images
var NLImgSet = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG').select('avg_rad').filterDate('2017-01-01', '2018-12-31')

// Mask each nighlights image first by settlements then to only select values greater than 0.3
function maskImage(img){
    return img.updateMask(settlements_mask)//.updateMask(img.gte(0.3))
}
NLImgSet = NLImgSet.map(maskImage)
var NLImg = NLImgSet.reduce(ee.Reducer.median())

NLImg = NLImg.clip(geometry)
Map.addLayer(NLImg, {min: 0, max: 100, palette: ['blue', 'green', 'red']}, 'Masked Nightlights');


// EXPORT results
Export.image.toDrive({
  image: NLImg,
  description: 'nightlights_africa_settlements',
  scale: 1000,
  region: geometry
});