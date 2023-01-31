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
coord  = SkyCoord('11h20m48.996s', '0d27m33.33s', frame='icrs')
name='587748929773240371'
file_name='cutout-HSC-R-9584-pdr2_wide-210216-133255.fits'
'''
'''
coord  = SkyCoord('12h24m56.506s', '2d48m3.92s', frame='icrs')
name='587726015608520857'
file_name='cutout-HSC-R-9837-pdr2_wide-210216-132516.fits'
'''
coord  = SkyCoord('14h59m39.805s', '2d18m4.31s', frame='icrs')
name='587726015625429219'
file_name='cutout-HSC-Y-9863-pdr2_wide-210216-140135.fits'

#==========================================================================
#READ_DATA-----------------------------------------------------------------

image_sub = fits.open('data/'+file_name,memmap=True)
img_sub = image_sub[1].data
wcs_sub = WCS(image_sub[1].header)
cx_sub, cy_sub = wcs_sub.wcs_world2pix(coord.ra, coord.dec, 1)
cx_sub=int((cx_sub))
cy_sub=int((cy_sub))
print(cx_sub,cy_sub)


#----------------------------------------------------------------------------
#============================================================================
#CREATE_MASK-----------------------------------------------------------------
'''
mask_sub = fits.open('mask_sub.fits', memmap=True)
mask_data_sub = mask_sub[0].data
mask_data_sub[mask_data_sub>0]=-1
mask_data_sub[mask_data_sub==0]=1
mask_data_sub[mask_data_sub==-1]=0

hdu1 = fits.PrimaryHDU()
hdu1.data = mask_data_sub
hdu1.writeto('mask_isoph_sub.fits', overwrite=True)
'''
#----------------------------------------------------------------------------
#============================================================================
#CUTOUT----------------------------------------------------------------------

data_sub = img_sub

cutout_data_sub = Cutout2D(data_sub, coord, (25, 25)*u.arcsec, wcs=wcs_sub, mode='strict')
#cutout_mask_sub = Cutout2D(mask_data_sub, coord, (120, 120)*u.arcsec, wcs=wcs_sub, mode='strict')

hdu_out_sub = fits.PrimaryHDU(data=cutout_data_sub.data, header=cutout_data_sub.wcs.to_header())
cutout_data_sub.data = hdu_out_sub.data
cutout_data_sub.wcs = WCS(hdu_out_sub.header)

cx_sub, cy_sub = cutout_data_sub.wcs.wcs_world2pix(coord.ra, coord.dec, 1)
cx_sub=int((cx_sub))
cy_sub=int((cy_sub))
print('...cutout_wcs__sub - ok, coord: '+repr(cx_sub)+' '+repr(cy_sub))


#----------------------------------------------------------------------------
#============================================================================
#CRATE_DATA_WIDE-------------------------------------------------------------
data_plt_sub = cutout_data_sub.data
#data_wide_sub = cutout_data_sub.data*cutout_mask_sub.data
data_wide_sub = cutout_data_sub.data

hdu3 = fits.PrimaryHDU()
hdu3.data = data_wide_sub
hdu3.writeto('work/data_wide_'+name+'.fits', overwrite=True)

data_wide_sub = ma.masked_equal(data_wide_sub, np.zeros(shape=data_wide_sub.shape))

#----------------------------------------------------------------------------
#============================================================================
#CREATE_ELLIPSE--------------------------------------------------------------
'''
geometry_sub = EllipseGeometry(x0=cx_sub-0.5, y0=cy_sub, sma=6, eps=0.1,
                           pa=100.*np.pi/180.)
'''
'''
geometry_sub = EllipseGeometry(x0=cx_sub-1.0, y0=cy_sub, sma=4, eps=0.1,
                           pa=100.*np.pi/180.)
'''
geometry_sub = EllipseGeometry(x0=cx_sub-0.5, y0=cy_sub, sma=4, eps=0.2,
                           pa=-3.0*np.pi/180.)
'''
#show_ellipse----------------------------------------------------------

aper = EllipticalAperture((geometry_sub.x0, geometry_sub.y0), geometry_sub.sma,
                          geometry_sub.sma*(1 - geometry_sub.eps), geometry_sub.pa)
plt.imshow(data_wide_sub, origin='lower')
aper.plot(color='white')
plt.show()
'''

#-----------------------------------------------------------------------------
#=============================================================================
#FIT_ELLIPSE------------------------------------------------------------------

print('fitting ellipse_sub...')
ellipse_sub = Ellipse(data_wide_sub, geometry_sub)
#isolist_sub = ellipse_sub.fit_image(sma0=None, minsma=0.01, maxsma=60, step=0.1, conver=0.5, minit=10, maxit=50, fflag=0.7, maxgerr=0.5, sclip=3.0, nclip=0, integrmode=u'bilinear', linear=True, maxrit=None)

#isolist_sub = ellipse_sub.fit_image(sma0=None, minsma=0.01, maxsma=40, step=0.1, conver=0.5, minit=10, maxit=50, fflag=0.7, maxgerr=0.5, sclip=3.0, nclip=0, integrmode=u'bilinear', linear=True, maxrit=None)

isolist_sub = ellipse_sub.fit_image(sma0=None, minsma=0.01, maxsma=70, step=0.1, conver=0.5, minit=10, maxit=50, fflag=0.7, maxgerr=0.5, sclip=3.0, nclip=0, integrmode=u'bilinear', linear=True, maxrit=None)

print(isolist_sub.to_table())

print('...plot ellipse model_sub...')
model_image_sub = build_ellipse_model(data_plt_sub.shape, isolist_sub, high_harmonics=False)
residual_sub = data_plt_sub - model_image_sub

hdu_sub1 = fits.PrimaryHDU(data=data_plt_sub, header=cutout_data_sub.wcs.to_header())
hdu_sub1.writeto('work/data_'+name+'.fits', overwrite=True)

hdu_sub2 = fits.PrimaryHDU(data=model_image_sub, header=cutout_data_sub.wcs.to_header())
hdu_sub2.writeto('work/model_'+name+'.fits', overwrite=True)

hdu_sub3 = fits.PrimaryHDU(data=residual_sub, header=cutout_data_sub.wcs.to_header())
hdu_sub3.writeto('work/residual_'+name+'.fits', overwrite=True)

hdu_sub4 = fits.PrimaryHDU(data=isolist_sub.sma)
hdu_sub4.writeto('isolist/'+name+'_sma.fits', overwrite=True)

hdu_sub5 = fits.PrimaryHDU(data=isolist_sub.intens)
hdu_sub5.writeto('isolist/'+name+'_intens.fits', overwrite=True)

hdu_sub6 = fits.PrimaryHDU(data=isolist_sub.int_err)
hdu_sub6.writeto('isolist/'+name+'_intens_err.fits', overwrite=True)

print('...done_sub')

#==============================================================================
#------------------------------------------------------------------------------
#==============================================================================
#PLOT_FIG----------------------------------------------------------------------
fig, axes = plt.subplots(figsize=(12, 4), nrows=1, ncols=3)
fig.subplots_adjust(left=0.04, right=0.96, bottom=0.04, top=0.96, wspace=0.01, hspace=0.01)
#data_sub---------------------------------------
ax11 = axes[0]
ax11.imshow(np.log(data_plt_sub), cmap='gray')
ax11.text(0.1, 0.97, 'sub', horizontalalignment='center', verticalalignment='center', transform=ax11.transAxes, color='white', fontsize=11)
l1=ax11.axhline(365,color='white',ls='-', linewidth=3, xmin=0.78, xmax=0.98)
ax11.text(0.93, 0.94, '1"', horizontalalignment='right', verticalalignment='center', transform=ax11.transAxes, color='white', fontsize=12)
ax11.set_xticks([])
ax11.set_yticks([])
ax11.set_xticklabels([])
ax11.set_yticklabels([])
ax11.set_xlim(30,120)
ax11.set_ylim(30,120)
#model_sub--------------------------------------
ax31 = axes[1]
ax31.imshow(np.log(model_image_sub), cmap='gray')
#l1=ax31.axhline(370,color='black',ls='-', linewidth=3, xmin=0.78, xmax=0.98)
#ax31.text(0.93, 0.94, '1"', horizontalalignment='right', verticalalignment='center', transform=ax31.transAxes, color='white', fontsize=12)
ax31.set_xticks([])
ax31.set_yticks([])
ax31.set_xticklabels([])
ax31.set_yticklabels([])
ax31.set_xlim(30,120)
ax31.set_ylim(30,120)
#resid_sub--------------------------------------
ax41 = axes[2]
ax41.imshow(residual_sub, cmap='gray', vmin=-1, vmax=1)
l1=ax41.axhline(365,color='white',ls='-', linewidth=3, xmin=0.78, xmax=0.98)
ax41.text(0.93, 0.94, '1"', horizontalalignment='right', verticalalignment='center', transform=ax41.transAxes, color='white', fontsize=12)
ax41.set_xticks([])
ax41.set_yticks([])
ax41.set_xticklabels([])
ax41.set_yticklabels([])
ax41.set_xlim(30,120)
ax41.set_ylim(30,120)
plt.savefig('result/ellips_'+name+'.png')
plt.show()
#END============================================================================

