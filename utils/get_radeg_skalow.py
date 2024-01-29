import astropy.units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time

loc = EarthLocation(lon=116.7644482*u.deg, lat=-26.82472208*u.deg)
t = Time('2021-09-21 14:12:40.1', scale='utc', location=loc)
sc = SkyCoord(ra=t.sidereal_time('apparent'), dec=loc.lat)
ra, dec = sc.ra.deg, sc.dec.deg
print(t, ':\t', 'RA = %.3f\tDEC = %.3f' %(ra, dec))

t1 = Time('2021-09-21 14:12:40.1', scale='utc', location=loc)
t2 = Time('2021-09-21 18:12:40.1', scale='utc', location=loc)

sc1 = SkyCoord(ra=t1.sidereal_time('apparent'), dec=loc.lat)
sc2 = SkyCoord(ra=t2.sidereal_time('apparent'), dec=loc.lat)
print('RA = [%f - %f] deg' %(sc1.ra.deg, sc2.ra.deg))

FoV_ra = (360 - sc1.ra.deg + sc2.ra.deg) * u.deg
FoV_dec = (16.*u.arcsec*2048).to('deg')

print('FoV (RA, Dec) : %.3f - %.3f %s' %(FoV_ra.value, FoV_dec.value, FoV_ra.unit))
