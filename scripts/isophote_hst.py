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

coord  = SkyCoord('12h48m15.230s', '17d46m26.45s', frame='icrs')
name='587742903401119752'
file_name='hst_13026_04_wfc3_uvis_f621m_drz.fits'

#==========================================================================
#READ_DATA-----------------------------------------------------------------

image_hst = fits.open('data/'+file_name,memmap=True)
img_hst = image_hst['SCI'].data
wcs_hst = WCS(image_hst['SCI'].header)
cx_hst, cy_hst = wcs_hst.wcs_world2pix(coord.ra, coord.dec, 1)
cx_hst=int((cx_hst))
cy_hst=int((cy_hst))
print(cx_hst,cy_hst)


#----------------------------------------------------------------------------
#============================================================================
#CREATE_MASK-----------------------------------------------------------------
'''
mask_hst = fits.open('mask_hst.fits', memmap=True)
mask_data_hst = mask_hst[0].data
mask_data_hst[mask_data_hst>0]=-1
mask_data_hst[mask_data_hst==0]=1
mask_data_hst[mask_data_hst==-1]=0

hdu1 = fits.PrimaryHDU()
hdu1.data = mask_data_hst
hdu1.writeto('mask_isoph_hst.fits', overwrite=True)

mask_sp = fits.open('mask_spitzer.fits', memmap=True)
mask_data_sp = mask_sp[0].data
mask_data_sp[mask_data_sp>0]=-1
mask_data_sp[mask_data_sp==0]=1
mask_data_sp[mask_data_sp==-1]=0

hdu2 = fits.PrimaryHDU()
hdu2.data = mask_data_sp
hdu2.writeto('mask_isoph_sp.fits', overwrite=True)
'''
#----------------------------------------------------------------------------
#============================================================================
#CUTOUT----------------------------------------------------------------------

data_hst = img_hst

cutout_data_hst = Cutout2D(data_hst, coord, (25, 25)*u.arcsec, wcs=wcs_hst, mode='strict')
#cutout_mask_hst = Cutout2D(mask_data_hst, coord, (120, 120)*u.arcsec, wcs=wcs_hst, mode='strict')

hdu_out_hst = fits.PrimaryHDU(data=cutout_data_hst.data, header=cutout_data_hst.wcs.to_header())
cutout_data_hst.data = hdu_out_hst.data
cutout_data_hst.wcs = WCS(hdu_out_hst.header)

cx_hst, cy_hst = cutout_data_hst.wcs.wcs_world2pix(coord.ra, coord.dec, 1)
cx_hst=int((cx_hst))
cy_hst=int((cy_hst))
print('...cutout_wcs__hst - ok, coord: '+repr(cx_hst)+' '+repr(cy_hst))


#----------------------------------------------------------------------------
#============================================================================
#CRATE_DATA_WIDE-------------------------------------------------------------
data_plt_hst = cutout_data_hst.data
#data_wide_hst = cutout_data_hst.data*cutout_mask_hst.data
data_wide_hst = cutout_data_hst.data

hdu3 = fits.PrimaryHDU()
hdu3.data = data_wide_hst
hdu3.writeto('work/data_wide_'+name+'.fits', overwrite=True)

data_wide_hst = ma.masked_equal(data_wide_hst, np.zeros(shape=data_wide_hst.shape))

#----------------------------------------------------------------------------
#============================================================================
#CREATE_ELLIPSE--------------------------------------------------------------

geometry_hst = EllipseGeometry(x0=cx_hst+4, y0=cy_hst-6, sma=4, eps=0.1,
                           pa=125.*np.pi/180.)

#show_ellipse----------------------------------------------------------
'''
aper = EllipticalAperture((geometry_hst.x0, geometry_hst.y0), geometry_hst.sma,
                          geometry_hst.sma*(1 - geometry_hst.eps), geometry_hst.pa)
plt.imshow(data_wide_hst, origin='lower')
aper.plot(color='white')
plt.show()
'''

#-----------------------------------------------------------------------------
#=============================================================================
#FIT_ELLIPSE------------------------------------------------------------------

print('fitting ellipse_hst...')
ellipse_hst = Ellipse(data_wide_hst, geometry_hst)
isolist_hst = ellipse_hst.fit_image(sma0=None, minsma=0.01, maxsma=100, step=0.1, conver=0.5, minit=10, maxit=50, fflag=0.7, maxgerr=0.5, sclip=3.0, nclip=0, integrmode=u'bilinear', linear=True, maxrit=None)

print(isolist_hst.to_table())

print('...plot ellipse model_hst...')
model_image_hst = build_ellipse_model(data_plt_hst.shape, isolist_hst, high_harmonics=False)
residual_hst = data_plt_hst - model_image_hst

hdu_hst1 = fits.PrimaryHDU(data=data_plt_hst, header=cutout_data_hst.wcs.to_header())
hdu_hst1.writeto('work/data_'+name+'.fits', overwrite=True)

hdu_hst2 = fits.PrimaryHDU(data=model_image_hst, header=cutout_data_hst.wcs.to_header())
hdu_hst2.writeto('work/model_'+name+'.fits', overwrite=True)

hdu_hst3 = fits.PrimaryHDU(data=residual_hst, header=cutout_data_hst.wcs.to_header())
hdu_hst3.writeto('work/residual_'+name+'.fits', overwrite=True)

hdu_hst4 = fits.PrimaryHDU(data=isolist_hst.sma)
hdu_hst4.writeto('isolist/'+name+'_sma.fits', overwrite=True)

hdu_hst5 = fits.PrimaryHDU(data=isolist_hst.intens)
hdu_hst5.writeto('isolist/'+name+'_intens.fits', overwrite=True)

hdu_hst6 = fits.PrimaryHDU(data=isolist_hst.int_err)
hdu_hst6.writeto('isolist/'+name+'_intens_err.fits', overwrite=True)

print('...done_hst')

#==============================================================================
#------------------------------------------------------------------------------
#==============================================================================
#PLOT_FIG----------------------------------------------------------------------
fig, axes = plt.subplots(figsize=(12, 4), nrows=1, ncols=3)
fig.subplots_adjust(left=0.04, right=0.96, bottom=0.04, top=0.96, wspace=0.01, hspace=0.01)
#data_hst---------------------------------------
ax11 = axes[0]
ax11.imshow(np.log(data_plt_hst), cmap='gray')
ax11.text(0.1, 0.97, 'HST', horizontalalignment='center', verticalalignment='center', transform=ax11.transAxes, color='white', fontsize=11)
l1=ax11.axhline(365,color='white',ls='-', linewidth=3, xmin=0.78, xmax=0.98)
ax11.text(0.93, 0.94, '1"', horizontalalignment='right', verticalalignment='center', transform=ax11.transAxes, color='white', fontsize=12)
ax11.set_xticks([])
ax11.set_yticks([])
ax11.set_xticklabels([])
ax11.set_yticklabels([])
ax11.set_xlim(250,380)
ax11.set_ylim(250,380)
#model_hst--------------------------------------
ax31 = axes[1]
ax31.imshow(np.log(model_image_hst), cmap='gray')
#l1=ax31.axhline(370,color='black',ls='-', linewidth=3, xmin=0.78, xmax=0.98)
#ax31.text(0.93, 0.94, '1"', horizontalalignment='right', verticalalignment='center', transform=ax31.transAxes, color='white', fontsize=12)
ax31.set_xticks([])
ax31.set_yticks([])
ax31.set_xticklabels([])
ax31.set_yticklabels([])
ax31.set_xlim(250,380)
ax31.set_ylim(250,380)
#resid_hst--------------------------------------
ax41 = axes[2]
ax41.imshow(residual_hst, cmap='gray', vmin=-1, vmax=1)
l1=ax41.axhline(365,color='white',ls='-', linewidth=3, xmin=0.78, xmax=0.98)
ax41.text(0.93, 0.94, '1"', horizontalalignment='right', verticalalignment='center', transform=ax41.transAxes, color='white', fontsize=12)
ax41.set_xticks([])
ax41.set_yticks([])
ax41.set_xticklabels([])
ax41.set_yticklabels([])
ax41.set_xlim(250,380)
ax41.set_ylim(250,380)
plt.savefig('result/ellips_'+name+'.png')
plt.show()
#END============================================================================

