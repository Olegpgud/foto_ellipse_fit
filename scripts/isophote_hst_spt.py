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

ngc7619  = SkyCoord('23h20m14.5s', '8d12m22.5s', frame='icrs')

#==========================================================================
#READ_DATA-----------------------------------------------------------------

image_hst = fits.open('hst_14219_34_wfc3_ir_f110w_drz.fits',memmap=True)
img_hst = image_hst['SCI'].data
wcs_hst = WCS(image_hst['SCI'].header)
cx_hst, cy_hst = wcs_hst.wcs_world2pix(ngc7619.ra, ngc7619.dec, 1)
cx_hst=int((cx_hst))
cy_hst=int((cy_hst))
print(cx_hst,cy_hst)

image_sp = fits.open('SPITZER_I2_49619200_0000_2_E10643313_maic.fits',memmap=True)
img_sp = image_sp[0].data
wcs_sp = WCS(image_sp[0].header)
cx_sp, cy_sp = wcs_sp.wcs_world2pix(ngc7619.ra, ngc7619.dec, 1)
cx_sp=int((cx_sp))
cy_sp=int((cy_sp))
print(cx_sp,cy_sp)

#----------------------------------------------------------------------------
#============================================================================
#CREATE_MASK-----------------------------------------------------------------

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

#----------------------------------------------------------------------------
#============================================================================
#CUTOUT----------------------------------------------------------------------

data_hst = img_hst

cutout_data_hst = Cutout2D(data_hst, ngc7619, (120, 120)*u.arcsec, wcs=wcs_hst, mode='strict')
cutout_mask_hst = Cutout2D(mask_data_hst, ngc7619, (120, 120)*u.arcsec, wcs=wcs_hst, mode='strict')

hdu_out_hst = fits.PrimaryHDU(data=cutout_data_hst.data, header=cutout_data_hst.wcs.to_header())
cutout_data_hst.data = hdu_out_hst.data
cutout_data_hst.wcs = WCS(hdu_out_hst.header)

cx_hst, cy_hst = cutout_data_hst.wcs.wcs_world2pix(ngc7619.ra, ngc7619.dec, 1)
cx_hst=int((cx_hst))
cy_hst=int((cy_hst))
print('...cutout_wcs__hst - ok, coord: '+repr(cx_hst)+' '+repr(cy_hst))


data_sp = img_sp

cutout_data_sp = Cutout2D(data_sp, ngc7619, (300, 300)*u.arcsec, wcs=wcs_sp, mode='strict')

hdu_out_sp = fits.PrimaryHDU(data=cutout_data_sp.data, header=cutout_data_sp.wcs.to_header())
cutout_data_sp.data = hdu_out_sp.data
cutout_data_sp.wcs = WCS(hdu_out_sp.header)

cx_sp, cy_sp = cutout_data_sp.wcs.wcs_world2pix(ngc7619.ra, ngc7619.dec, 1)
cx_sp=int((cx_sp))
cy_sp=int((cy_sp))
print('...cutout_wcs__sp - ok, coord: '+repr(cx_sp)+' '+repr(cy_sp))

#----------------------------------------------------------------------------
#============================================================================
#CRATE_DATA_WIDE-------------------------------------------------------------
data_plt_hst = cutout_data_hst.data
data_wide_hst = cutout_data_hst.data*cutout_mask_hst.data

hdu3 = fits.PrimaryHDU()
hdu3.data = data_wide_hst
hdu3.writeto('data_wide_hst.fits', overwrite=True)

data_wide_hst = ma.masked_equal(data_wide_hst, np.zeros(shape=data_wide_hst.shape))


data_plt_sp = cutout_data_sp.data
data_wide_sp = cutout_data_sp.data*mask_data_sp

hdu4 = fits.PrimaryHDU()
hdu4.data = data_wide_sp
hdu4.writeto('data_wide_sp.fits', overwrite=True)

data_wide_sp = ma.masked_equal(data_wide_sp, np.zeros(shape=data_wide_sp.shape))

#----------------------------------------------------------------------------
#============================================================================
#CREATE_ELLIPSE--------------------------------------------------------------

geometry_hst = EllipseGeometry(x0=cx_hst, y0=cy_hst, sma=10, eps=0.3,
                           pa=125.*np.pi/180.)

geometry_sp = EllipseGeometry(x0=cx_sp, y0=cy_sp, sma=10, eps=0.3,
                           pa=55.*np.pi/180.)

#-----------------------------------------------------------------------------
#=============================================================================
#FIT_ELLIPSE------------------------------------------------------------------

print('fitting ellipse_hst...')
ellipse_hst = Ellipse(data_wide_hst, geometry_hst)
isolist_hst = ellipse_hst.fit_image(sma0=None, minsma=0.05, maxsma=700, step=0.1, conver=0.5, minit=10, maxit=50, fflag=0.7, maxgerr=0.5, sclip=3.0, nclip=0, integrmode=u'bilinear', linear=True, maxrit=None)

print(isolist_hst.to_table())

print('...plot ellipse model_hst...')
model_image_hst = build_ellipse_model(data_plt_hst.shape, isolist_hst, high_harmonics=False)
residual_hst = data_plt_hst - model_image_hst

hdu_hst = fits.PrimaryHDU(data=residual_hst, header=cutout_data_hst.wcs.to_header())
hdu_hst.writeto('residual_isoph/residual_hst_test.fits', overwrite=True)
print('...done_hst')



print('fitting ellipse_sp...')
ellipse_sp = Ellipse(data_wide_sp, geometry_sp)
isolist_sp = ellipse_sp.fit_image(sma0=None, minsma=0.05, maxsma=200, step=0.01, conver=0.5, minit=10, maxit=50, fflag=0.7, maxgerr=0.5, sclip=3.0, nclip=0, integrmode=u'bilinear', linear=True, maxrit=None)

print(isolist_sp.to_table())

print('...plot ellipse model_sp...')
model_image_sp = build_ellipse_model(data_plt_sp.shape, isolist_sp, high_harmonics=False)
residual_sp = data_plt_sp - model_image_sp


hdu_sp = fits.PrimaryHDU(data=residual_sp, header=cutout_data_sp.wcs.to_header())
hdu_sp.writeto('residual_isoph/residual__sp.fits', overwrite=True)
print('...done_sp')

#==============================================================================
#------------------------------------------------------------------------------
#==============================================================================
#PLOT_FIG----------------------------------------------------------------------
fig, axes = plt.subplots(figsize=(12, 8), nrows=2, ncols=3)
fig.subplots_adjust(left=0.04, right=0.96, bottom=0.04, top=0.96, wspace=0.01, hspace=0.01)
#data_sp----------------------------------------
ax1 = axes[0,0]
data_plt_sp=np.rot90(data_plt_sp)
ax1.imshow(np.log(data_plt_sp), cmap='gray')
ax1.text(0.5, 0.97, 'NGC7619', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes, color='white', fontsize=11)
ax1.text(0.1, 0.97, 'Spitzer', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes, color='white', fontsize=11)
l1=ax1.axhline(347,color='white',ls='-', linewidth=3, xmin=0.78, xmax=0.98)
ax1.text(0.93, 0.94, '24"', horizontalalignment='right', verticalalignment='center', transform=ax1.transAxes, color='white', fontsize=12)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax1.set_xlim(150,350)
ax1.set_ylim(150,350)
#model_sp---------------------------------------
ax3 = axes[0,1]
model_image_sp=np.rot90(model_image_sp)
ax3.imshow(np.log(model_image_sp), cmap='gray')
#ax3.text(0.5, 0.97, 'NGC7619', horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes, color='white', fontsize=11)
#ax3.text(0.1, 0.97, 'Spitzer', horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes, color='white', fontsize=11)
l1=ax3.axhline(347,color='white',ls='-', linewidth=3, xmin=0.78, xmax=0.98)
ax3.text(0.93, 0.94, '24"', horizontalalignment='right', verticalalignment='center', transform=ax3.transAxes, color='white', fontsize=12)
ax3.set_xticks([])
ax3.set_yticks([])
ax3.set_xticklabels([])
ax3.set_yticklabels([])
ax3.set_xlim(150,350)
ax3.set_ylim(150,350)
#resid_sp---------------------------------------
ax4 = axes[0,2]
residual_sp=np.rot90(residual_sp)
ax4.imshow(residual_sp, cmap='gray', vmin=-0.2, vmax=0.25)
#ax4.text(0.5, 0.97, 'NGC7619', horizontalalignment='center', verticalalignment='center', transform=ax4.transAxes, color='white', fontsize=11)
#ax4.text(0.1, 0.97, 'Spitzer', horizontalalignment='center', verticalalignment='center', transform=ax4.transAxes, color='white', fontsize=11)
l1=ax4.axhline(347,color='white',ls='-', linewidth=3, xmin=0.78, xmax=0.98)
ax4.text(0.93, 0.94, '24"', horizontalalignment='right', verticalalignment='center', transform=ax4.transAxes, color='white', fontsize=12)
circle = plt.Circle((270, 232.5), 11, color='r', fill=False)
ax4.add_artist(circle)
ax4.set_xticks([])
ax4.set_yticks([])
ax4.set_xticklabels([])
ax4.set_yticklabels([])
ax4.set_xlim(150,350)
ax4.set_ylim(150,350)
#data_hst---------------------------------------
ax11 = axes[1,0]
ax11.imshow(np.log(data_plt_hst), cmap='gray')
ax11.text(0.5, 0.97, 'NGC7619', horizontalalignment='center', verticalalignment='center', transform=ax11.transAxes, color='white', fontsize=11)
ax11.text(0.1, 0.97, 'HST', horizontalalignment='center', verticalalignment='center', transform=ax11.transAxes, color='white', fontsize=11)
l1=ax11.axhline(990,color='white',ls='-', linewidth=3, xmin=0.78, xmax=0.98)
ax11.text(0.93, 0.94, '12"', horizontalalignment='right', verticalalignment='center', transform=ax11.transAxes, color='white', fontsize=12)
ax11.set_xticks([])
ax11.set_yticks([])
ax11.set_xticklabels([])
ax11.set_yticklabels([])
ax11.set_xlim(330,1000)
ax11.set_ylim(330,1000)
#model_hst--------------------------------------
ax31 = axes[1,1]
ax31.imshow(np.log(model_image_hst), cmap='gray')
#ax31.text(0.5, 0.97, 'NGC7619', horizontalalignment='center', verticalalignment='center', transform=ax31.transAxes, color='white', fontsize=11)
#ax31.text(0.1, 0.97, 'HST', horizontalalignment='center', verticalalignment='center', transform=ax31.transAxes, color='white', fontsize=11)
l1=ax31.axhline(990,color='white',ls='-', linewidth=3, xmin=0.78, xmax=0.98)
ax31.text(0.93, 0.94, '12"', horizontalalignment='right', verticalalignment='center', transform=ax31.transAxes, color='white', fontsize=12)
ax31.set_xticks([])
ax31.set_yticks([])
ax31.set_xticklabels([])
ax31.set_yticklabels([])
ax31.set_xlim(330,1000)
ax31.set_ylim(330,1000)
#resid_hst--------------------------------------
ax41 = axes[1,2]
ax41.imshow(residual_hst, cmap='gray', vmin=-1, vmax=1)
#ax41.text(0.5, 0.97, 'NGC7619', horizontalalignment='center', verticalalignment='center', transform=ax41.transAxes, color='white', fontsize=11)
#ax41.text(0.1, 0.97, 'HST', horizontalalignment='center', verticalalignment='center', transform=ax41.transAxes, color='white', fontsize=11)
l1=ax41.axhline(990,color='white',ls='-', linewidth=3, xmin=0.78, xmax=0.98)
ax41.text(0.93, 0.94, '12"', horizontalalignment='right', verticalalignment='center', transform=ax41.transAxes, color='white', fontsize=12)
circle2 = plt.Circle((845, 500), 60, color='r', fill=False)
ax41.add_artist(circle2)
ax41.set_xticks([])
ax41.set_yticks([])
ax41.set_xticklabels([])
ax41.set_yticklabels([])
ax41.set_xlim(330,1000)
ax41.set_ylim(330,1000)
plt.savefig('residual_isoph/data/ellips_NGC7619.png')
plt.show()
#END============================================================================
