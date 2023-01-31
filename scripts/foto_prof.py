import matplotlib.pyplot as plt
import numpy as np
from astropy.modeling.models import Gaussian2D
from photutils.datasets import make_noise_image
from photutils.isophote import EllipseGeometry, Ellipse
from photutils.isophote import build_ellipse_model
from photutils import EllipticalAperture
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord
import numpy.ma as ma
from astropy.nddata import Cutout2D
'''
name = 'test'
coord  = SkyCoord('23h20m14.5s', '8d12m22.5s', frame='icrs')
data_file = 'data/SPITZER_I2_49619200_0000_2_E10643313_maic.fits'
'''


#name='587726015625429219'
z=0.042
#coord = SkyCoord('14h59m39.805s', '2d18m4.31s', frame='icrs')

#data_file = 'data/frame-g-001458-4-0668.fits.bz2'


name='587748929773240371'
coord = SkyCoord('11h20m48.996s', '0d27m33.33s', frame='icrs')
data_file = 'data/frame-g-006793-5-0064.fits.bz2'


#name='587732482211184777'
#coord = SkyCoord('11h56m28.921s', '48d55m41.71s', frame='icrs')
#data_file = 'data/frame-g-002964-1-0296.fits.bz2'


#name='587735429080547544'
#coord = SkyCoord('13h12m4.604s', '48d27m36.66s', frame='icrs')
#data_file = 'data/frame-g-003650-2-0066.fits.bz2'

#==========================================================================
#READ_DATA-----------------------------------------------------------------

image = fits.open(data_file,memmap=True)
img = image['PRIMARY'].data
img_header = image['PRIMARY'].header
wcs = WCS(image[0].header)
cd1=img_header['CD1_1']
cd2=img_header['CD1_2']

cx, cy = wcs.wcs_world2pix(coord.ra, coord.dec, 1)
cx=int((cx))
cy=int((cy))
print(cx,cy)

c=299792.458
H0=66.93 

#----------------------------------------------------------------------------
#============================================================================
#CUTOUT----------------------------------------------------------------------

data = img

cutout_data = Cutout2D(data, coord, (50, 50)*u.arcsec, wcs=wcs, mode='strict')

hdu_out = fits.PrimaryHDU(data=cutout_data.data, header=cutout_data.wcs.to_header())
cutout_data.data = hdu_out.data
cutout_data.wcs = WCS(hdu_out.header)

cx, cy = cutout_data.wcs.wcs_world2pix(coord.ra, coord.dec, 1)
cx=int((cx))
cy=int((cy))
print('...cutout_wcs - ok, coord: '+repr(cx)+' '+repr(cy))

#----------------------------------------------------------------------------
#============================================================================
#CRATE_DATA_WIDE-------------------------------------------------------------

data_wide = cutout_data.data

#hdu2 = fits.PrimaryHDU()
#hdu2.data = data_wide
#hdu2.writeto('data_wide.fits', overwrite=True)

data_wide = ma.masked_equal(data_wide, np.zeros(shape=data_wide.shape))

#----------------------------------------------------------------------------
#============================================================================
#CREATE_ELLIPSE--------------------------------------------------------------

geometry = EllipseGeometry(x0=cx-0.5, y0=cy, sma=4, eps=0.1,
                           pa=60.*np.pi/180.)

#show_ellipse----------------------------------------------------------

aper = EllipticalAperture((geometry.x0, geometry.y0), geometry.sma,
                          geometry.sma*(1 - geometry.eps), geometry.pa)
plt.imshow(data_wide, origin='lower')
aper.plot(color='white')
plt.show()

#----------------------------------------------------------------------------
#=============================================================================
#FIT_ELLIPSE------------------------------------------------------------------
print('fitting ellipse...')
ellipse = Ellipse(data_wide, geometry)
#isolist = ellipse.fit_image()
isolist = ellipse.fit_image(sma0=None, minsma=0.01, maxsma=60, step=0.01, conver=0.5, minit=10, maxit=50, fflag=0.7, maxgerr=0.5, sclip=3.0, nclip=0, integrmode=u'bilinear', linear=True, maxrit=None)

print(isolist.to_table())

hdu = fits.PrimaryHDU(data=isolist.sma)
hdu.writeto('isolist/'+name+'_sma.fits', overwrite=True)

hdu2 = fits.PrimaryHDU(data=isolist.intens)
hdu2.writeto('isolist/'+name+'_intens.fits', overwrite=True)

#----------------------------------------------------------------------------
#=============================================================================
#CREATE_PROF------------------------------------------------------------------

sma=isolist.sma
intens=isolist.intens
zeropoint=22.5
m = -2.5*np.log10(intens)+zeropoint
dist=10**6*c*z/H0
m_abs = m - 5*np.log10((dist)/10)-0.1

deg_pix = (cd1**2+cd2**2)**(0.5)
#print('deg_pix = '+repr(deg_pix))
r=sma*deg_pix*0.0174533*dist
r_kpc = r/1000

plt.plot(r_kpc,m_abs)
plt.ylim(max(m_abs), min(m_abs)) 
plt.xlabel('r, kpc')
plt.ylabel('M, mag')
plt.title('M(r)')
plt.savefig('results/'+name+'.png')
plt.show()


print('...done')

