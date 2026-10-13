"""WGS84 UTM MGRS encoder. Refuse polar inputs rather than fabricate UPS."""
import math

def utm(latitude, longitude):
    if not (-80 <= latitude < 84 and -180 <= longitude <= 180):
        raise ValueError('Use GeographicLib/GEOTRANS for polar or invalid coordinates')
    z = min(60, int((longitude + 180) // 6) + 1)
    if 56 <= latitude < 64 and 3 <= longitude < 12:
        z = 32
    if 72 <= latitude < 84 and 0 <= longitude < 42:
        z = 31 if longitude < 9 else 33 if longitude < 21 else 35 if longitude < 33 else 37
    a, e2, k = 6378137., 0.0066943799901413165, .9996
    phi = math.radians(latitude)
    dl = math.radians(longitude - (z * 6 - 183))
    ep = e2 / (1-e2)
    n = a / math.sqrt(1-e2*math.sin(phi)**2)
    t, c, aa = math.tan(phi)**2, ep*math.cos(phi)**2, math.cos(phi)*dl
    m = a*((1-e2/4-3*e2**2/64-5*e2**3/256)*phi
           -(3*e2/8+3*e2**2/32+45*e2**3/1024)*math.sin(2*phi)
           +(15*e2**2/256+45*e2**3/1024)*math.sin(4*phi)
           -35*e2**3/3072*math.sin(6*phi))
    x = 500000+k*n*(aa+(1-t+c)*aa**3/6+(5-18*t+t*t+72*c-58*ep)*aa**5/120)
    y = k*(m+n*math.tan(phi)*(aa*aa/2+(5-t+9*c+4*c*c)*aa**4/24
                            +(61-58*t+t*t+600*c-330*ep)*aa**6/720))
    if latitude < 0:
        y += 10000000
    return z,x,y

def mgrs(latitude, longitude, digits=3):
    if not 0 <= digits <= 5:
        raise ValueError('MGRS digits must be 0..5 per axis')
    z,x,y = utm(latitude,longitude)
    band = 'CDEFGHJKLMNPQRSTUVWX'[min(19,int((latitude+80)//8))]
    cols = ('ABCDEFGH','JKLMNPQR','STUVWXYZ')[(z-1)%3]
    col = cols[int(x//100000)-1]
    rows = 'ABCDEFGHJKLMNPQRSTUV'
    row = rows[(int(y//100000)+(5 if z%2==0 else 0))%20]
    scale = 10**(5-digits)
    tail = '' if not digits else f'{int(x%100000//scale):0{digits}d}{int(y%100000//scale):0{digits}d}'
    return f'{z:02d}{band}{col}{row}{tail}'
