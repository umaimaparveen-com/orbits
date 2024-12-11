import numpy as np

def eph(el, t, rho=None, rv=None):
    """
    Function to compute orbital elements, relative coordinates, or radial velocities (RV).
    
    Parameters:
    - el: Orbital elements [PTEaWwiK1K1V0], should be a list or numpy array.
    - t: Time vector (array-like).
    - rho: Optional; if set, will return theta and rho instead of coordinates.
    - rv: Optional; if set, will return radial velocities (RV) for primary and secondary.
    
    Returns:
    - res: Relative coordinates (or radial velocities) as a 2D numpy array.
    """
    
    # Initialize variables
    n = len(t)
    res = np.zeros((n, 2))  # result array for relative coordinates (or RV)
    pi2 = 2 * np.pi
    gr = 180 / np.pi  # conversion factor for radians to degrees
    
    # Extract orbital elements
    P = el[0]
    SF = el[2]
    CF2 = 1 - SF * SF
    CF = np.sqrt(CF2)
    CF3 = CF * CF2
    EC = np.sqrt((1 + SF) / (1 - SF))
    CWW = np.cos(el[4] / gr)
    SWW = np.sin(el[4] / gr)
    W = el[5] / gr
    CW = np.cos(el[5] / gr)
    SW = np.sin(el[5] / gr)
    SI = np.sin(el[6] / gr)
    CI = np.cos(el[6] / gr)

    # If RV (radial velocity) is requested, initialize K1, K2, and V0
    if rv is not None:
        K1 = el[7]
        K2 = el[8]
        V0 = el[9]
    else:
        # Thiele-van-den-Bos elements
        AA = el[3] * (CW * CWW - SW * SWW * CI)
        BB = el[3] * (CW * SWW + SW * CWW * CI)
        FF = el[3] * (-SW * CWW - CW * SWW * CI)
        GG = el[3] * (-SW * SWW + CW * CWW * CI)

    # Loop through each time value to calculate positions
    for i in range(n):
        # Solve Kepler equation
        DT = t[i] - el[1]
        PHASE = DT / el[0] % 1
        if PHASE < 0:
            PHASE += 1.0

        ANM = PHASE * 2 * np.pi
        E = ANM
        E1 = E + (ANM + SF * np.sin(E) - E) / (1.0 - SF * np.cos(E))
        
        # Iterate to solve for E (Kepler's equation)
        while np.abs(E1 - E) > 1.E-5:
            E = E1
            E1 = E + (ANM + SF * np.sin(E) - E) / (1.0 - SF * np.cos(E))
        
        V = 2 * np.arctan(EC * np.tan(E1 / 2.))

        if rv is not None:
            # Radial Velocity ephemerides
            U = V + W
            CU = np.cos(U)
            A1 = SF * CW + CU
            res[i, 0] = V0 + K1 * A1
            res[i, 1] = V0 - K2 * A1
        else:
            # Visual orbit
            CV = np.cos(V)
            R = CF2 / (1 + SF * CV)
            X = R * CV
            Y = R * np.sin(V)
            res[i, 0] = AA * X + FF * Y
            res[i, 1] = BB * X + GG * Y
    
    # If rho is requested, calculate theta and rho
    if rho is not None:
        rho = np.sqrt(res[:, 0]**2 + res[:, 1]**2)
        theta = np.degrees(np.arctan2(res[:, 1], res[:, 0]))
        theta = (theta + 360) % 360
        res[:, 0] = theta
        res[:, 1] = rho

    return res

def getcoord(s):
    """
    Function to convert a coordinate string in the format 'DDDMMSS' or 'DDDMM.MMMM' 
    to a decimal degree value.

    Parameters:
    - s: Coordinate string (e.g., "123.4567", "-45.1234").

    Returns:
    - res: Decimal degree value.
    """
    
    # Find the position of the decimal point in the string
    l = s.find('.')
    
    # Extract the degree, minute, and second parts
    deg = int(s[:l])  # Degrees part before the decimal point
    min = int(s[l+1:l+3])  # Minutes part (next 2 characters)
    sec = float(s[l+3:])  # Seconds part after the decimal point

    # Calculate the decimal degree
    res = abs(deg) + min / 60.0 + sec / 3600.0

    # If the degree is negative, apply the negative sign
    if deg < 0:
        res = -res
    
    return res


def correct(data, t0):
    """
    Function to convert between Julian days and years based on the provided input.
    
    Parameters:
    - data: A 2D array (or list of lists) where the first column represents time (either Julian years or days).
    - t0: A reference time that helps determine if the data is in Julian years or days.
    
    Modifies the input `data` in-place to convert the time values.
    """
    
    # Extract the time values (assuming data is a numpy array or a list of lists)
    time = data[:, 0]

    for i in range(len(time)):
        # If time is in years and we need to convert to Julian days (JD-240000)
        if time[i] < 3e3 and t0 > 3e3:
            data[i, 0] = 365.242198781 * (time[i] - 1900.0) + 15020.31352

        # If time is in Julian days (JD-240000) and we need to convert to Besselian years
        elif time[i] > 3e3 and t0 < 3e3:
            data[i, 0] = 1900.0 + (time[i] - 15020.31352) / 365.242198781

    return data

def sixty(scalar):
    """
    Function to convert a scalar value in degrees to a 3-element vector [deg, min, sec].
    
    Parameters:
    - scalar: The scalar value in degrees (can be positive or negative).
    
    Returns:
    - result: A list containing [deg, min, sec].
    """
    # Take the absolute value of scalar in degrees and convert to minutes and seconds
    ss = abs(3600.0 * scalar)
    mm = abs(60.0 * scalar)
    dd = abs(scalar)
    
    # Initialize the result list
    result = [0.0, 0.0, 0.0]
    
    # Calculate degrees, minutes, and seconds
    result[0] = int(dd)
    result[1] = int(mm - 60.0 * result[0])
    result[2] = ss - 3600.0 * result[0] - 60.0 * result[1]
    
    # If scalar is negative, adjust the signs of the components
    if scalar < 0.0:
        if result[0] != 0:
            result[0] = -result[0]
        elif result[1] != 0:
            result[1] = -result[1]
        else:
            result[2] = -result[2]
    
    return result

import numpy as np

# Define the constants and arrays that were declared in the IDL code
xorb, xbase, obj, el, elerr, fixel, elname, pos, rv1, rv2, graph, editel, x, y, x2, y2 = [None]*16

# Initialize necessary arrays
el = np.zeros(10)
fixel = np.ones(10)  # 1 if the element is fitted
nmax = 500  # max number of observations or RV
pos = np.zeros((nmax, 6))  # time, PA, theta, err, O-C, O-C
rv1 = np.zeros((nmax, 3))  # time, Va, eVa
rv2 = np.zeros((nmax, 3))  # time, Vb, eVb

# Initialize object structure
obj = {
    'name': '',
    'radeg': 0.0,
    'dedeg': 0.0,
    'npos': 0,
    'nrv1': 0,
    'nrv2': 0,
    'rms': np.zeros(4),
    'chi2n': np.zeros(4),
    'chi2': 0.0,
    'fname': ''
}

def readinp(fname):
    try:
        with open(fname, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f'File {fname} is not found, exiting.')
        obj['fname'] = ''
        return

    # Reset the element array and flags
    global el, fixel
    el = np.zeros(10)
    fixel = np.ones(10)

    # Reading the file
    for line in lines[:13]:
        a = line.strip()
        
        if a.startswith('*'):
            fix = 0
            a = a[1:]
        else:
            fix = 1
            
        if a.startswith('C'):
            continue

        t = a.split()
        
        ind = -1
        if t[0] == 'Object:':
            obj['name'] = t[1].strip()
        elif t[0] == 'RA:':
            obj['radeg'] = 15.0 * getcoord(t[1])
        elif t[0] == 'Dec:':
            obj['dedeg'] = getcoord(t[1])
        elif t[0] == 'P':
            ind = 0
        elif t[0] == 'T':
            ind = 1
        elif t[0] == 'e':
            ind = 2
        elif t[0] == 'a':
            ind = 3
        elif t[0] == 'W':
            ind = 4
        elif t[0] == 'w':
            ind = 5
        elif t[0] == 'i':
            ind = 6
        elif t[0] == 'K1':
            ind = 7
        elif t[0] == 'K2':
            ind = 8
        elif t[0] == 'V0':
            ind = 9
        else:
            print(f'Unknown tag <{t[0]}>')

        if ind >= 0:
            el[ind] = float(t[1])
            fixel[ind] = fix

    kpos = 0
    krv1 = 0
    krv2 = 0
    for line in lines[13:]:
        a = line.strip()

        if a.startswith('C'):
            continue
        
        t = a.split()
        
        for item in t:
            if item.startswith('I1'):
                # Position measure
                pos[kpos, 0:3] = np.array([float(i) for i in t[:3]])
                kpos += 1
                continue
            
            if item.startswith('V'):
                type_ = item[1]
                if type_ == ':':
                    continue  # Skip this measure
                
                elif type_ == 'a':
                    rv1[krv1, 0:2] = np.array([float(i) for i in t[:2]])
                    krv1 += 1
                elif type_ == 'b':
                    rv2[krv2, 0:2] = np.array([float(i) for i in t[:2]])
                    krv2 += 1
                elif type_ == '2':
                    rv1[krv1, 0:2] = np.array([float(i) for i in t[:2]])
                    rv2[krv2, 1:2] = np.array([float(i) for i in t[3:5]])
                    rv2[krv2, 0] = float(t[0])
                    krv1 += 1
                    krv2 += 1
                else:
                    pass  # Handle other types if needed
                continue

    # Trim the arrays
    if kpos > 0:
        pos = pos[:kpos]
    else:
        pos = np.zeros((0, 6))

    if krv1 > 0:
        rv1 = rv1[:krv1]
    else:
        rv1 = np.zeros((0, 3))

    if krv2 > 0:
        rv2 = rv2[:krv2]
    else:
        rv2 = np.zeros((0, 3))

    # Correct the data
    correct(pos, el[1])
    correct(rv1, el[1])
    correct(rv2, el[1])

    # Add precession correction here

    print(f'Position measures: {kpos}')
    print(f'RV measures: {krv1}, {krv2}')
    obj['npos'] = kpos
    obj['nrv1'] = krv1
    obj['nrv2'] = krv2

    if kpos > 0 and np.max(pos[:, 3] == 0):
        print('Warning: zero errors encountered in input file! Stopping.')
        return

    elerr = np.zeros(10)

    if krv1 == 0:
        graph['mode'] = 0  # Plot visual orbit only
    if kpos == 0:
        graph['mode'] = 1  # Plot SB orbit only

# Assuming the correct and getcoord functions are defined elsewhere
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.backends.backend_pdf import PdfPages

def orbplot(ps=False, speckle=None, rverr=None):
    # Constants and setup
    wsize = 600  # 600 points in graphic window
    margin = 60  # margin, points
    npts = 100  # points in the curve
    fact = 1.3  # scale factor

    name = obj['fname'].split('.')[0]  # for plots

    if (graph_mode == 0) and (obj['npos'] > 0):  # orbit plot
        time = np.linspace(0, 1, npts) * el[0] + el[1]
        xye = eph(el, time)  # [*,2] array of orbit curve

        gr = 180. / np.pi
        xobs = -pos[:, 2] * np.sin(pos[:, 1] / gr)  # data points
        yobs = pos[:, 2] * np.cos(pos[:, 1] / gr)

        # Find the plot scale
        tmp = np.concatenate([-xye[:, 1], xobs])
        xr = [np.min(tmp), np.max(tmp)]
        tmp = np.concatenate([xye[:, 0], yobs])
        yr = [np.min(tmp), np.max(tmp)]

        range_ = np.max([xr[1] - xr[0], yr[1] - yr[0]]) * fact  # maximum range in both coordinates
        range_ = float(range_)
        scale = (wsize - margin) / range_  # from arcseconds to points
        xcent = float(np.mean(xr))
        ycent = float(np.mean(yr))

        # Transform to pixel coordinates
        x = (xobs - xcent) * scale + wsize / 2 + margin / 2
        y = (yobs - ycent) * scale + wsize / 2 + margin / 2

        xy0 = eph(el, pos[:, 0])

        if speckle is not None:
            selsp = np.where(pos[:, 3] <= speckle)[0]
            nsp = len(selsp)
        else:
            nsp = 0

        pp = float(margin) / wsize  # left margin
        pp1 = 0.995  # right margin

        # Prepare plot
        if ps:
            pdf_name = f"{name}_POS.pdf"
            with PdfPages(pdf_name) as pdf:
                fig, ax = plt.subplots(figsize=(14, 14))
                ax.set_xlim([xcent - range_ / 2, xcent + range_ / 2])
                ax.set_ylim([ycent - range_ / 2, ycent + range_ / 2])

                ax.set_xlabel('X (arcsec)')
                ax.set_ylabel('Y (arcsec)')
                ax.plot(-xye[:, 1], xye[:, 0], linestyle='None', marker='o', color='black')  # orbit curve
                ax.scatter(xobs, yobs, c='blue', marker='x')  # observed points

                for i in range(len(x)):
                    ax.plot([xobs[i], -xy0[i, 1]], [yobs[i], xy0[i, 0]], color='black')  # ephemeris points

                if nsp > 0:
                    ax.scatter(xobs[selsp], yobs[selsp], c='red', marker='+')

                pdf.savefig(fig)  # Save the plot to PDF
                print(f"Plot {pdf_name} is produced.")
        else:
            # Plot without saving to file
            fig, ax = plt.subplots(figsize=(14, 14))
            ax.set_xlim([xcent - range_ / 2, xcent + range_ / 2])
            ax.set_ylim([ycent - range_ / 2, ycent + range_ / 2])

            ax.set_xlabel('X (arcsec)')
            ax.set_ylabel('Y (arcsec)')
            ax.plot(-xye[:, 1], xye[:, 0], linestyle='None', marker='o', color='black')  # orbit curve
            ax.scatter(xobs, yobs, c='blue', marker='x')  # observed points

            for i in range(len(x)):
                ax.plot([xobs[i], -xy0[i, 1]], [yobs[i], xy0[i, 0]], color='black')  # ephemeris points

            if nsp > 0:
                ax.scatter(xobs[selsp], yobs[selsp], c='red', marker='+')

            plt.show()

import numpy as np
import matplotlib.pyplot as plt

def plot_rv_data(rv1, rv2=None, el=None, rverr=None, maxphase=1.5, wsize=8, margin=0.5, ps=False, name='RV_Plot'):
    phase = maxphase * np.arange(len(rv1)) / (len(rv1) - 1)
    time = phase * el[0] + el[1]
    curve = eph(el, time, rv=True)  # Assuming eph function is available
    ampl = (max(curve) - min(curve)) * 0.6
    ycent = np.mean(curve)
    yr = [-1, 1] * ampl + ycent

    # Set plot scales
    yscale = (wsize - margin) / (2 * ampl)
    xscale = (wsize - margin) / maxphase
    xcent = maxphase / 2.0

    # Radial velocity 1 (primary)
    ph1 = (rv1[:, 0] - el[1]) / el[0] % 1
    v1 = rv1[:, 1]
    ev1 = rv1[:, 2]

    # Plot setup
    fig, ax = plt.subplots(figsize=(wsize, wsize))
    ax.plot(phase, curve[:, 0], label="Orbit curve", linestyle='-', color='b')
    ax.set_xlim([0, maxphase])
    ax.set_ylim(yr)
    ax.set_xlabel("Phase")
    ax.set_ylabel("RV (km/s)")
    
    # RV Data Plotting (with optional error bar handling)
    if rverr is not None:
        good = ev1 < rverr[0]
        bad = ev1 >= rverr[0]
        ax.scatter(ph1[good], v1[good], color='g', label='Good Data')
        ax.scatter(ph1[bad], v1[bad], color='r', label='Bad Data')
    else:
        ax.scatter(ph1, v1, color='g', label='RV Data')

    # Plot secondary (SB2)
    if rv2 is not None:
        ph2 = (rv2[:, 0] - el[1]) / el[0] % 1
        v2 = rv2[:, 1]
        ev2 = rv2[:, 2]
        ax.plot(phase, curve[:, 1], label="Secondary orbit", linestyle='--', color='r')
        if rverr is not None:
            good = ev2 < rverr[1]
            bad = ev2 >= rverr[1]
            ax.scatter(ph2[good], v2[good], color='b', label='Secondary Good Data')
            ax.scatter(ph2[bad], v2[bad], color='y', label='Secondary Bad Data')
        else:
            ax.scatter(ph2, v2, color='b', label='Secondary RV Data')

    # Save to file if required
    if ps:
        plt.savefig(f"{name}_RV.ps", format='ps')

    plt.legend()
    plt.show()

from scipy.optimize import least_squares

def alleph(i, par, obj, el, rv1, rv2, fixel):
    # Derivative calculation step size
    e = 0.01
    deltas = np.array([e * el[0], e * el[0], e, e * el[3], 1., 1., 1., e * el[7], e * el[8], e * el[7]])

    # Select fitted elements
    selfit = np.where(fixel > 0)[0]
    el0 = np.copy(el)
    el0[selfit] = par

    result = np.zeros(10)  # Initialize results (value + derivatives)

    if i < 2 * obj['npos']:
        # rho-theta calculation (primary)
        if i >= obj['npos']:
            j = 1  # rho
        else:
            j = 0  # theta
        time = obj['pos'][i - j * obj['npos'], 0]
        res = eph(el0, [time], rho=(j == 1))  # Assuming eph function exists

        for k in range(7):
            if fixel[k] > 0:
                el1 = np.copy(el0)
                el1[k] += deltas[k]
                result[k] = (eph(el1, [time], rho=(j == 1)) - res) / deltas[k]

        return result
    else:
        # RV Calculation (primary and secondary)
        idx = [0, 1, 2, 5, 7, 8, 9]  # Indices of RV-relevant elements
        if i < 2 * obj['npos'] + obj['nrv1']:
            j = 0  # Primary
            time = rv1[i - 2 * obj['npos'], 0]
        else:
            j = 1  # Secondary
            time = rv2[i - 2 * obj['npos'] - obj['nrv1'], 0]
        
        res = eph(el0, [time], rv=(j == 1))

        for k in range(7):
            if fixel[idx[k]] > 0:
                el1 = np.copy(el0)
                el1[idx[k]] += deltas[idx[k]]
                result[idx[k]] = (eph(el1, [time], rv=(j == 1)) - res) / deltas[idx[k]]

        return result
      
from scipy.optimize import least_squares

def fit_orbital_elements(obj, el, rv1, rv2, fixel):
    def residuals(par):
        return alleph(np.arange(len(rv1)), par, obj, el, rv1, rv2, fixel)

    result = least_squares(residuals, el)
    return result.x  # Optimized orbital elements

import numpy as np

def fitorb(obj, rms=None):
    v0err = 0.3  # instrumental error

    # Calculate the observables
    npos = obj['npos']
    nrv1 = obj['nrv1']
    nrv2 = obj['nrv2']
    n = 2 * npos + nrv1 + nrv2  # total number of points

    yy = np.zeros(n)  # Observables: theta, rho, RV1, RV2
    err = np.zeros(n)  # Measurement errors

    if npos > 0:
        yy[:npos] = obj['pos'][:, 1]  # angle
        err[:npos] = obj['pos'][:, 3] / obj['pos'][:, 2] * 180 / np.pi  # error in angle
        yy[npos:2*npos] = obj['pos'][:, 2]  # separation
        err[npos:2*npos] = obj['pos'][:, 3]  # error in separation

    if nrv1 > 0:
        yy[2*npos:2*npos+nrv1] = obj['rv1'][:, 1]  # RV primary
        err[2*npos:2*npos+nrv1] = np.sqrt(obj['rv1'][:, 2]**2 + v0err**2)  # error in RV
        if nrv2 > 0:
            yy[2*npos+nrv1:] = obj['rv2'][:, 1]  # RV secondary
            err[2*npos+nrv1:] = np.sqrt(obj['rv2'][:, 2]**2 + v0err**2)  # error in RV

    # Fitting elements
    fixel = obj['fixel']  # zero if fixed
    selfit = np.where(fixel > 0)[0]
    print(f'Fitting {len(selfit)} elements')
    par = obj['el'][selfit]

    ix = np.arange(n)  # Fictitious argument
    y1 = np.zeros(n)  # Ephemeris
    for i in range(n):
        y1[i] = alleph(i, par)[0]  # Ephemeris calculation (replace with actual function)

    # Residual calculation
    nmin = np.array([0, npos, 2*npos, 2*npos+nrv1])  # Left limit
    nmax = np.array([npos, 2*npos, 2*npos+nrv1, 2*npos+nrv1+nrv2])  # Right limit
    ndat = nmax - nmin  # Data points of each type

    wt = err**(-2)  # Weight
    resid2 = (yy - y1)**2 * wt

    sd = np.zeros(4)  # res^2 / sig2
    wsum = np.zeros(4)
    normchi2 = np.zeros(4)

    for j in range(4):
        if ndat[j] > 0:
            sd[j] = np.sum(resid2[nmin[j]:nmax[j]])
            wsum[j] = np.sum(wt[nmin[j]:nmax[j]])
            normchi2[j] = sd[j] / ndat[j]

    # Weighted RMS
    wrms = np.zeros(4)
    wsel = np.where(wsum > 0)[0]
    wrms[wsel] = np.sqrt(sd[wsel] / wsum[wsel])  # Weighted RMS

    print(f'CHI2/N: {normchi2}')
    print(f'RMS in Theta, rho, RV1, RV2: {wrms}')

    if rms is not None:  # Only RMS is needed, no fit
        obj['rms'] = wrms
        obj['chi2n'] = normchi2
        return

    # Perform LM fit (using some fitting function, for example, lmfit)
    # y1, chi2, iter = lmfit(ix, yy, par, measure_errors=err, function_name='alleph', iter=iter, itmax=30)

    print(f'LM iterations: {iter}')
    print(f'CHI2, M: {chi2}, {2*npos + nrv1 + nrv2 - len(selfit)}')

    # Recalculate residuals
    resid2 = (yy - y1)**2 * wt
    for j in range(4):
        if ndat[j] > 0:
            sd[j] = np.sum(resid2[nmin[j]:nmax[j]])
            wsum[j] = np.sum(wt[nmin[j]:nmax[j]])
            normchi2[j] = sd[j] / ndat[j]

    # Weighted RMS
    wrms = np.zeros(4)
    wsel = np.where(wsum > 0)[0]
    wrms[wsel] = np.sqrt(sd[wsel] / wsum[wsel])  # Weighted RMS

    print(f'CHI2/N: {normchi2}')
    print(f'RMS in Theta, rho, RV1, RV2: {wrms}')

    obj['rms'] = wrms
    obj['chi2n'] = normchi2
    obj['chi2'] = chi2

    obj['el'][selfit] = par  # Update elements
    obj['elerr'][selfit] = sigma  # Element errors

    # You can add further functionality here for plotting, etc.
    showdat()
    orbplot()

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from datetime import datetime

def orbsave(obj, rv1, rv2):
    # Generate the filename from the object name
    fname = obj.fname.split('.')[0] + '.out'
    
    # Convert RA and DEC into hours, minutes, seconds format
    ra1 = (obj.radeg / 15.0)  # Convert RA to hours
    dec1 = np.abs(obj.dedeg)  # Convert DEC to degrees

    # Format RA and DEC for output
    ra = "{:9.6f}".format(ra1[0] + 0.01 * ra1[1] + 1e-4 * ra1[2])
    dec = "{:09.6f}".format(dec1[0] + 0.01 * dec1[1] + 1e-4 * dec1[2])
    if obj.dedeg < 0:
        dec = '-' + dec
    
    # Orbital elements (example array sizes, adjust based on the actual data)
    elfmt = ['{:12.5f}', '{:10.4f}', '{:6.4f}', '{:8.4f}', '{:8.2f}', 
             '{:8.2f}', '{:8.2f}', '{:8.2f}', '{:8.2f}', '{:8.2f}']
    elname = ['P', 'T', 'e', 'a', 'W', 'w', 'i', 'K1', 'K2', 'V0']
    
    el = np.array([obj.P, obj.T, obj.e, obj.a, obj.W, obj.w, obj.i, obj.K1, obj.K2, obj.V0])
    elerr = np.zeros_like(el)
    
    with open(fname, 'w') as f:
        f.write(f"Object: {obj.name}\n")
        f.write(f"RA:     {ra}\n")
        f.write(f"Dec:    {dec}\n")
        
        # If chi2 is 0, calculate residuals
        if obj.chi2 == 0:
            fitorb(obj)  # Call fitorb function to calculate residuals
        
        # Print orbital elements
        for i in range(10):
            if el[i] == 0:
                elerr[i] = 0.0  # Fixed element case
            f.write(f"{elname[i]} {el[i]:12.5f} {elerr[i]:10.4f}\n")
        
        # Print CHI2 and RMS
        f.write(f"C RMS: {obj.rms:10.2f} {obj.chi2:10.2f}\n")
        
        # Print period and T0 in years
        if el[1] > 10000.0:
            tmp = [el[0] / 365.2421987, 1900.0 + (el[1] - 15020.31352) / 365.242198781]
        else:
            tmp = [el[0] * 365.24219878, 365.242198781 * (el[1] - 1900.0) + 15020.31352]
        
        f.write(f"P,T= {tmp[0]:12.5f} {tmp[1]:10.4f}\n")
        print(f"P,T= {tmp[0]:12.5f} {tmp[1]:10.4f}")

        # Position measurements and residuals if available
        if obj.npos > 0:
            res = eph(el, pos[:, 0], rho=True)
            for i in range(obj.npos):
                f.write(f"{pos[i, 0]:10.3f} {pos[i, 1]:8.1f} "
                        f"{pos[i, 2]:8.3f} {res[i, 0]:8.1f} {res[i, 1]:8.3f} I1\n")

        # RV observations
        nrv = obj.nrv1 + obj.nrv2
        if nrv > 0:
            comp = np.zeros(nrv, dtype=int)
            compname = ['a', 'b']
            
            if obj.nrv2 > 0:
                comp[obj.nrv1:nrv] = 1  # secondary
                rv = np.vstack([rv1, rv2])
            else:
                rv = rv1
            
            ord = np.argsort(rv[:, 0])  # Sort by time
            rv = rv[ord]
            comp = comp[ord]
            
            res = eph(el, rv[:, 0], rv=True)
            for i in range(nrv):
                f.write(f"{rv[i, 0]:10.3f} {rv[i, 1]:10.3f} {rv[i, 2]:10.3f} "
                        f"{res[i, comp[i]]:10.3f} V{compname[comp[i]]}\n")
        
        print(f"Results saved in {fname}")

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Assuming you have the necessary functions defined elsewhere, like eph, msum, etc.

# Streamlit Sidebar for interaction
st.sidebar.title("Orbital Fitting")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload a file", type=["txt", "csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)  # Example: reading the file as a DataFrame

# Input fields for various parameters
n_points = st.sidebar.number_input("Number of points", min_value=0, value=0)
tstart = st.sidebar.number_input("Start time", min_value=0.0)
tend = st.sidebar.number_input("End time", min_value=0.0)
if n_points > 1:
    t = np.linspace(tstart, tend, n_points)
else:
    t = np.array([tstart])

# Display selected values
st.write(f"Start time: {tstart}, End time: {tend}, Number of points: {n_points}")

# Ephemerides button
if st.sidebar.button('Ephemerides'):
    if uploaded_file is not None:
        # Placeholder for your 'eph' function
        res = eph(el, t, rho=True)  # Call your 'eph' function here
        st.write("Ephemerides results:")
        st.write(res)
    
    # Optional RV button
    if st.sidebar.button('RV (Radial Velocity)'):
        res2 = eph(el, t, rv=True)  # Call the radial velocity function
        st.write("Radial Velocity Results:")
        st.write(res2)

# Plotting
fig, ax = plt.subplots()
ax.plot(t, res[:, 0], label="Primary")
ax.plot(t, res[:, 1], label="Secondary")
ax.set_xlabel("Time")
ax.set_ylabel("Separation (AU)")
ax.legend()

# Display plot
st.pyplot(fig)

# Add more features such as mass sum, spectroscopic masses, etc.
if st.sidebar.button('Mass sum from parallax'):
    plx = st.sidebar.number_input("Enter Parallax", min_value=0.0, value=0.0)
    if plx != 0:
        msum(plx)  # Call the msum function here
        st.write("Mass sum calculated from parallax")

# Handle more operations as needed

import streamlit as st
import numpy as np

# Assuming 'el' and 'elname' are lists or arrays containing the orbital elements and their names.
# Initialize orbital elements (dummy values for demonstration)
el = np.array([1.0, 0.1, 0.5, 10, 0, 0, 0, 0, 0, 0])  # Example orbital elements
elname = ['Semi-major axis', 'Eccentricity', 'Inclination', 'Longitude of Ascending Node',
          'Argument of Perihelion', 'Mean Anomaly', 'Element 7', 'Element 8', 'Element 9', 'Element 10']

# Streamlit UI to edit orbital elements
st.title("Edit Orbital Elements")

# Create editable fields for each orbital element
for i in range(10):
    el[i] = st.number_input(f"Edit {elname[i]}", value=el[i], step=0.01)

# Display current orbital elements
st.write("Current Orbital Elements:")
for i in range(10):
    st.write(f"{elname[i]}: {el[i]}")

# Option to perform actions on the updated values (optional)
if st.button("Save Changes"):
    st.write("Changes have been saved.")
    # Here you would perform the necessary updates, like recalculating ephemerides or fitting orbits based on updated elements.
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Define orbital element names and options
elname = ['P', 'T', 'e', 'a', 'W', 'w', 'i', 'K1', 'K2', 'V0']
options = ['Ephemeride', 'Mass sum', 'M sin3i', 'M2min', 'Plot separation']

# Initialize orbital elements (dummy values for demonstration)
el = np.array([1.0, 0.1, 0.5, 10, 0, 0, 0, 0, 0, 0])  # Example orbital elements

# Streamlit Title
st.title("Orbital GUI")

# File uploader (replaces 'Open' and 'Reopen' buttons)
uploaded_file = st.file_uploader("Upload Input File", type=["txt", "csv"])

# If file is uploaded, process it (for now, just print the file name)
if uploaded_file is not None:
    fname = uploaded_file.name
    st.write(f"Loaded file: {fname}")
else:
    fname = None

# Create editable fields for orbital elements
st.subheader("Edit Orbital Elements")
for i in range(10):
    el[i] = st.number_input(f"Edit {elname[i]}", value=el[i], step=0.01)

# Display current orbital elements
st.write("Current Orbital Elements:")
for i in range(10):
    st.write(f"{elname[i]}: {el[i]}")

# Dropdown menu for options
selected_option = st.selectbox("Analysis Options", options)

# Buttons for actions
if st.button("Fit"):
    st.write("Fitting model with updated elements...")
    # Add fitting logic here if needed

if st.button("Save"):
    st.write("Saving orbital data...")
    # Add save functionality here, for example, save to a file or database

if st.button("Exit"):
    st.write("Exiting application...")
    # Optionally, implement an exit functionality

# Plotting Section
st.subheader("Graphical Plot")
# Example plot (use your own plot logic here)
fig, ax = plt.subplots()
ax.plot(np.linspace(0, 10, 100), np.sin(np.linspace(0, 10, 100)))
ax.set_title("Orbital Plot")
st.pyplot(fig)

# For showing additional data (similar to the 'showdat' function in IDL)
if uploaded_file is not None:
    st.write("Displaying additional data from file...")
    # Add your data processing logic here, e.g., read and show the content of the file

# Add any further custom logic you need, for example:
# - Performing ephemerides calculations
# - Showing mass sum, M sin3i, or other calculations
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Orbital element names
elname = ['P', 'T', 'e', 'a', 'W', 'w', 'i', 'K1', 'K2', 'V0']

# Placeholder arrays for the orbital elements, errors, and fixed flags
el = np.zeros(10)  # Orbital elements (replace with actual data)
fixel = np.zeros(10)  # Fixed orbital elements (replace with actual data)
elerr = np.zeros(10)  # Orbital element errors (replace with actual data)

# Object information (replace with actual data)
obj = {
    'fname': "ObjectName",  # Example object name
    'npos': 100,  # Number of positions
    'nrv1': 10,  # First RV data points
    'nrv2': 20   # Second RV data points
}

# Streamlit app
st.title("Orbital Data Analysis")

# Function to read input data (This replaces the readinp functionality)
def readinp(fname):
    # Replace this with actual file reading logic
    # For now, we'll simulate reading a file with mock data
    obj['fname'] = fname
    obj['npos'] = 100  # Example value for number of positions
    obj['nrv1'] = 10   # Example value for RV data points
    obj['nrv2'] = 20   # Example value for RV data points

    st.write(f"Loaded object: {obj['fname']}")
    st.write(f"Number of positions (Npos): {obj['npos']}")
    st.write(f"RV data points (NRV1, NRV2): {obj['nrv1']}, {obj['nrv2']}")

# File uploader and processing
uploaded_file = st.file_uploader("Upload Input File", type=["txt", "csv"])

if uploaded_file is not None:
    fname = uploaded_file.name
    readinp(fname)

# Editable fields for orbital elements (showing 'el' values)
st.subheader("Edit Orbital Elements")
for i in range(10):
    el[i] = st.number_input(f"Edit {elname[i]}", value=el[i], step=0.01)

# Function to show orbital data (This corresponds to the showdat function)
def showdat():
    st.subheader("Orbital Elements with Errors and Fixes")
    elfmt = ['F12.5', 'F10.4', 'F8.4', 'F8.4', 'F8.2', 'F8.2', 'F8.2', 'F8.2', 'F8.2', 'F8.2']
    for i in range(10):
        st.write(f"{elname[i]}: {el[i]:{elfmt[i]}}")
        st.write(f"Fixed: {fixel[i]}")
        st.write(f"Error: {elerr[i]:{elfmt[i]}}")

    st.write(f"Object Name: {obj['fname']}")
    st.write(f"Number of Positions (Npos): {obj['npos']}")
    st.write(f"Number of RV points (NRV1, NRV2): {obj['nrv1']} , {obj['nrv2']}")

# Display orbital elements
showdat()

# Function for mass sum calculation (msum equivalent)
def msum(el, plx):
    a = el[3] / plx
    p = el[0]
    if el[1] > 10000:
        p = p / 365.25  # Convert period to days if greater than 10000
    mass_sum = (a**3) / (p**2)
    return mass_sum

# Parallax input
parallax = st.number_input("Enter Parallax (in arcseconds)", value=0.1, step=0.01)

if st.button("Calculate Mass Sum"):
    mass_sum = msum(el, parallax)
    st.write(f"Mass Sum: {mass_sum}")

# Function for mass calculation (M sin³i equivalent)
def msin3i(el):
    p = el[0]
    if el[1] < 3000:
        p *= 365.25  # Convert period to days if less than 3000
    e = el[2]

    mass = 1.035e-7 * (1 - e**2)**1.5 * (el[7] + el[8])**3 * p
    mass1
import numpy as np
import streamlit as st

# Orbital elements (el) and other variables
el = np.zeros(10)  # Orbital elements (replace with actual data)

# m1: Mass of the primary star
m1 = st.number_input("Enter Mass of Primary Star (m1)", value=1.0, step=0.1)

# Function to calculate the minimum secondary mass for SB1 (M2min equivalent)
def M2min(m1, el):
    per = el[0]
    if el[1] < 3000.:
        per *= 365.25  # Convert period to days if less than 3000
    e = el[2]  # Eccentricity
    k1 = el[7]  # K1 value

    # Handle known inclination
    if el[6] > 0.:
        sini = np.sin(np.radians(el[6]))  # Inclination, converted to radians
        st.write(f"Account for the inclination: sini = {sini}")
    else:
        sini = 1.0

    # Cubic root of the mass function = m2 * (m1 + m2)^(-2/3)
    y = k1 * per**(1./3.) * (1 - e**2)**0.5 * 4.695e-3 / sini
    m2old = 0.0
    m2new = y * m1**(-2./3.)

    # Iterative calculation of the minimum secondary mass (M2min)
    while abs(m2old - m2new) > 1e-3:
        m2old = m2new
        m2new = y * (m1 + m2old)**(2./3.)

    return m2new

# Button to calculate the minimum secondary mass
if st.button("Calculate Minimum Secondary Mass (M2min)"):
    m2min_value = M2min(m1, el)
    st.write(f"The minimum secondary mass (M2min) is: {m2min_value}")
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Orbital elements (el) and other variables
el = np.zeros(10)  # Orbital elements (replace with actual data)
pos = np.zeros((100, 3))  # Placeholder for position array (replace with actual data)

# Function to compute ephemerides (eph equivalent in Python)
def eph(el, time):
    # Placeholder for ephemerides calculation (this needs to be defined based on your model)
    # Assuming xye corresponds to [x, y, z] position components
    # Here, just an example where xye is a dummy 3D position array
    xye = np.random.rand(len(time), 3)  # Replace with actual ephemerides logic
    return xye

# Function to plot the separation (rho) and position
def plotrho(el, pos):
    time = pos[:, 0]  # Assuming time is in the first column of pos
    xye = eph(el, time)  # Get the ephemerides (e.g., [x, y, z] position components)

    # Compute rho (separation) as the Euclidean distance
    rho = np.sqrt(np.sum(xye**2, axis=1))

    # Plot position
    plt.plot(time, pos[:, 2], marker='o', label="Position", linestyle='None')
    plt.plot(time, rho, label="Separation (rho)")
    plt.xlabel("Time")
    plt.ylabel("Distance (units)")
    plt.title("Position and Separation over Time")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)  # Display the plot using Streamlit

# Button to plot rho and position
if st.button("Plot Position and Separation (rho)"):
    plotrho(el, pos)
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Orbital elements (el) and other variables
el = np.zeros(10)  # Orbital elements (replace with actual data)

# Function to plot the orbit (orbplot equivalent in Python)
def orbplot(el):
    # Placeholder for orbit plotting (this needs to be defined based on your model)
    # Example: plotting RV curve based on orbital elements
    time = np.linspace(0, 10, 1000)  # Example time array (replace with actual time data)
    rv_curve = np.sin(2 * np.pi * time / el[0])  # Example RV curve (replace with actual calculation)
    
    # Plot the RV curve
    plt.plot(time, rv_curve)
    plt.xlabel("Time")
    plt.ylabel("Radial Velocity (km/s)")  # Example unit, adjust based on your model
    plt.title(f"Radial Velocity Curve for Period P = {el[0]}")
    plt.grid(True)
    st.pyplot(plt)  # Display the plot using Streamlit

# Function to run the trials and generate plots
def try_trials(dt, n1, n2):
    for i in range(n1, n2 + 1):
        el[0] = dt / i  # Update period (el[0] is the period)
        st.write(f"P = {el[0]}")  # Display the period in Streamlit
        orbplot(el)  # Plot the orbit
        # Stopping condition can be adjusted (currently plotting for each i)
        st.stop()  # Stop the streamlit app after each plot (remove if you want continuous plotting)

# Streamlit user interface for input
dt = st.number_input("Enter value for dt", value=100.0)  # Example input for dt
n1 = st.number_input("Enter starting value for n1", value=1, min_value=1)
n2 = st.number_input("Enter ending value for n2", value=5, min_value=1)

# Button to run the trials
if st.button("Run Trials"):
    try_trials(dt, n1, n2)
