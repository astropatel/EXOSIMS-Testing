#! /usr/local/anaconda2/bin/python
print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
print 'Simulation has begun'
print 'The following text will show you EXOSIMS specific variables as well'
print 'the sequence of observations.'
print 'Most error messages are harmless'
print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n'

# ===================================================================================================
#                            IMPORT STATEMENTS
# ===================================================================================================


import sys, warnings, collections
from astropy.utils.exceptions import AstropyWarning

sys.path.insert(0, '/var/www/wfirst-dev/html/sims/tools/EXOSIMS')
# print(sys.path)
# print 'in Python: sys.argv= ', sys.argv

import numpy as np
import json, os, csv, string

import EXOSIMS.MissionSim as msim
from astropy import constants as con
import astropy
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable

# import mosaic_tools as mt

# ===================================================================================================
#                  IGNORE ALL ASTROPY WARNINGS.
# ===================================================================================================

warnings.simplefilter('ignore', category=AstropyWarning)
warnings.filterwarnings("ignore")


# ===================================================================================================
#                      PLOTTING DEFINITIONS
# ===================================================================================================

def triple_axes_dist(ylog=False, xlog=False, topax=True, rightax=True,
                     figsize=(10, 10), xlabel='x', ylabel='y'):
    """Sets up plots with 3 axes -- one in center, one on right and one above center
    plot. The purpose is to have a scatter plot or w/e in the center, and two
    distribution plots of the x and y parameters in the side plots.
    Input:
    axScatter: axis object for center scatter plot
    ylog, xlog: booleans to indicate whether x and y axes of center plot are in
                log scale (base 10)
    xlabel, ylabel : labels for x and y axes labels

    Return
    """
    axScatter = plt.figure(111, figsize=figsize).add_subplot(111)
    axScatter.set_xlabel('%s' % xlabel, fontsize=25)
    axScatter.set_ylabel('%s' % ylabel, fontsize=25)

    divider = make_axes_locatable(axScatter)
    if topax:
        axHistX = divider.append_axes("top", size=2, pad=0.2, sharex=axScatter)
        plt.setp(axHistX.get_xticklabels(), visible=False)
    else:
        axHistX = None

    if rightax:
        axHistY = divider.append_axes("right", size=2, pad=0.2, sharey=axScatter)
        plt.setp(axHistY.get_yticklabels(), visible=False)
    else:
        axHistY = None

    if xlog:
        axScatter.set_xscale('log')
        axHistX.set_xscale('log', nonposy='clip')
    if ylog:
        axScatter.set_yscale('log')
        axHistY.set_yscale('log', nonposy='clip')

    return axScatter, axHistX, axHistY


def plot_setup(axis, gridon=False, minortickson=True,
               ticklabel_fontsize=20, majortick_width=2.5,
               minortick_width=1.2, majortick_size=8,
               minortick_size=5, axes_linewidth=1.5,
               ytick_direction='in', xtick_direction='in',
               yaxis_right=False, ylog=False, xlog=False):
    """Changes the boring default matplotlib plotting canvas so that it
    looks nice and neat with thicker borders and larger tick marks as well
    as larger fontsizes for the axis labels. Options exist to include or
    exclude the plot grid and minortick mark labels -- set up as boolean
    variables"""

    if gridon:
        axis.grid()
    if minortickson:
        axis.minorticks_on()
    if yaxis_right:
        axis.yaxis.tick_right()

    for line in axis.yaxis.get_majorticklines():
        line.set_markeredgewidth(majortick_width)
    for line in axis.xaxis.get_majorticklines():
        line.set_markeredgewidth(majortick_width)

    for line in axis.xaxis.get_minorticklines():
        line.set_markeredgewidth(minortick_width)
    for line in axis.yaxis.get_minorticklines():
        line.set_markeredgewidth(minortick_width)

    if xlog:
        axis.set_xscale('log', nonposy='clip')
    if ylog:
        axis.set_yscale('log', nonposy='clip')

    plt.rc('font', **{'family': 'sans-serif',
                      'sans-serif': ['Helvetica']})
    plt.rcParams['mathtext.fontset'] = 'stixsans'
    axis.tick_params(axis='both', which='major', labelsize=ticklabel_fontsize)
    plt.rc("axes", linewidth=axes_linewidth)
    plt.rcParams['xtick.major.size'] = majortick_size
    plt.rcParams['xtick.minor.size'] = minortick_size
    plt.rcParams['ytick.major.size'] = majortick_size
    plt.rcParams['ytick.minor.size'] = minortick_size

    plt.rcParams['xtick.direction'] = xtick_direction
    plt.rcParams['ytick.direction'] = ytick_direction

    plt.subplots_adjust(left=0.13, bottom=0.13, top=0.95, right=0.97)

    return


def linePlot(xdat, ydat, ax0=None, labels=('x', 'y'), xlog=False, ylog=False, **kw):
    """
    Input:
    ------
    xdat: numpy.array, x-axis data
    ydat: numpy.array, y-axis data
    ax0: None-type or axis object (plt.figure().add_subplot(111)). If None,
         an axis object is created.
    labels: 2x1 string-tuple, x and y-axis labels
    xlog, ylog: boolean, plot in log space.
    kw: dict, key-word dictionary of matplotlib plot inputs.

    Return:
    -------
    ax: axis plot object.
    """

    if ax0 is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
    else:
        ax = ax0
    try:
        plot_setup(ax, ylog=ylog, xlog=xlog)
    except:
        pass

    ax.plot(xdat, ydat, **kw)
    ax.set_xlabel(r'%s' % labels[0], fontsize=20)
    ax.set_ylabel(r'%s' % labels[1], fontsize=20)

    return ax


def scatterPlot(xdat, ydat, ax0=None, labels=('x', 'y'), xlog=False, ylog=False,
                figsize=(10, 10), **kw):
    """
    Input:
    ------
    xdat: numpy.array, x-axis data
    ydat: numpy.array, y-axis data
    ax0: None-type or axis object (plt.figure().add_subplot(111)). If None,
         an axis object is created.
    labels: 2x1 string-tuple, x and y-axis labels
    xlog, ylog: boolean, plot in log space.
    kw: dict, key-word dictionary of matplotlib plot inputs.

    Return:
    -------
    ax: axis plot object.
    """

    if ax0 is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        ax = ax0

    try:
        plot_setup(ax, ylog=ylog, xlog=xlog)
    except:
        pass

    ax.plot(xdat, ydat, linestyle='None', **kw)
    ax.set_xlabel(r'%s' % labels[0], fontsize=20)
    ax.set_ylabel(r'%s' % labels[1], fontsize=20)

    return ax


def histPlot(xdat, bins=15, ax0=None, labels=('x', 'y'), xlog=False, ylog=False, **kw):
    """
    Input:
    ------
    xdat: numpy.array, x-axis data
    bins: int or numpy.array, bins.
    ax0: None-type or axis object (plt.figure().add_subplot(111)). If None,
         an axis object is created.
    labels: 2x1 string-tuple, x and y-axis labels
    xlog, ylog: boolean, plot in log space.
    kw: dict, key-word dictionary of matplotlib plot inputs.

    Return:
    -------
    ax: axis plot object.
    """

    if ax0 is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
    else:
        ax = ax0

    try:
        plot_setup(ax, ylog=ylog, xlog=xlog)
    except:
        pass

    ax.hist(xdat, bins, **kw)
    ax.set_xlabel(r'%s' % labels[0], fontsize=20)
    ax.set_ylabel(r'%s' % labels[1], fontsize=20)

    return ax


def skyplot(ra, dec, figsize=(20, 20), projection='mollweide', axisbg='white',
            ticklcolor=None, ticklabels=None):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection=projection, facecolor=axisbg)

    plt.setp(ax.spines.values(), linewidth=10)
    if ticklcolor is not None:
        ax.set_xticklabels(tick_labels, color=ticklcolor)  # we add the scale on the x axis
    else:
        ax.set_xticklabels(tick_labels)

    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
        tick.label1.set_fontsize(20)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
        tick.label1.set_fontsize(20)

    return ax


# ===================================================================================================
#                      FORMAT OUTPUT DEFINITIONS
# ===================================================================================================


def reformat_DRM(drm):
    """
    PROCESS THE EXOSIMS RESULTS DRM OBJECT IN A MORE PALLATABLE FORMAT.
    DRM IS CURRENTLY A LIST OF DICTIONARY THAT CONTAINS INFORMATION FOR EVERY VISIT.
    THIS REFORMATTER TURNS IT INTO A DICTIONARY OF RESULTS

    """
    # ========================================================================
    # WHAT KEYS WILL BE ORGANIZED HOW - UPDATED AS NEEDED
    # ------------------------------------------------------------------------
    # Don't do anything with these initially- these are dictionaries in the DRM.
    dontdoanything = ['det_params', 'char_params']

    # Keys whose array elements need to be repeated based on number of planets
    # detected per star.
    expand_keys = ['star_name', 'star_ind', 'arrival_time', 'OB_nb',
                   'det_mode', 'char_mode', 'det_time', 'char_time',
                   'det_fZ', 'char_fZ',
                   'FA_det_status', 'FA_char_status', 'FA_char_SNR', 'FA_char_fEZ',
                   'FA_char_dMag', 'FA_char_WA',
                   'slew_time', 'slew_dV', 'slew_mass_used',
                   'sc_mass', 'det_dV', 'det_mass_used',
                   'det_dF_lateral', 'det_dF_axial', 'char_dV', 'char_mass_used',
                   'char_dF_lateral', 'char_dF_axial']

    # Keys whose arrays need to be concatendated because each is an array
    # e.g. - plan_inds: [ [0],[0,21,5],...,[51,20] ] <-- where each sub-array
    # corresponds to n planets around one star.
    # In the end, each element in these arrays will have their
    # own individually mapped index in arrays.

    concatenate_keys = ['plan_inds',
                        'det_status', 'char_status',
                        'det_SNR', 'char_SNR']

    # Keys whose arrays will be expanded from 'char_params', and 'det_params'
    # UPDATE AS NEDED.

    det_expandkeys = ['WA', 'd', 'dMag', 'fEZ', 'phi']
    char_expandkeys = ['WA', 'd', 'dMag', 'fEZ', 'phi']
    # ========================================================================

    # CHECK IF DRM IS EMPTY
    if drm:

        # ====================================================================
        # COLLECT ALL KEYS IN DRM AND UNIQUE-IFY THE ARRAY
        kys = np.array([dt.keys() for dt in drm])
        kys = np.unique(np.concatenate(kys))
        # ====================================================================

        # --------------------------------------------
        # create dictionary of drm based on keywords
        ddrm = {}
        # --------------------------------------------

        # ====================================================================
        #        REARRANGE THE DRM
        # ----------------------------------------------------
        # change DRM from array of diciontaries to dictionary
        # where the values of each key is an array. Each entry
        # in the arrays represents the meta data for a single planet.
        # This section will do the above, and replace bad value stuff
        # with nan's. Keys reported in DRM not in the ICD are not used.
        for ky in kys:
            arr = []
            for dt in drm:

                if ky not in dt:
                    arr.append([np.nan])
                elif isinstance(dt[ky], np.ndarray):
                    if np.size(dt[ky]) == 0:
                        arr.append([np.nan])
                    else:
                        arr.append(dt[ky])
                elif not isinstance(dt[ky], (int, long, float)) and not dt[ky]:
                    arr.append([np.nan])

                else:
                    arr.append(dt.get(ky, [np.nan]))

            ddrm[ky] = np.array(arr)
        # ====================================================================

        # ====================================================================
        #  EXPAND ARRAYS TO GET SINGLE PLANET MAPPING
        # --------------------------------------------------------------------
        plan_raw_inds = ddrm['plan_inds']

        # Here, clone the values for each "expand_keys" key to the number of
        # planets per star.
        for ky in expand_keys:
            try:
                vals = ddrm[ky]
                # ADD SAME # AS PLANETS DETECTED.
                # AT LEAST ONE ADDED EVEN WITH NO PLANETS DETECTED
                temp_val = [[vals[i]] * len(plan_raw_inds[i])
                            for i in xrange(len(plan_raw_inds))]

                ddrm[ky] = np.concatenate(temp_val)
            except KeyError:
                pass
        # ====================================================================

        # ====================================================================
        #  CONCATENATE ARRAYS TO GET SINGLE PLANET MAPPING
        # --------------------------------------------------------------------
        for ky in concatenate_keys:
            try:
                ddrm[ky] = np.concatenate(ddrm[ky])
            except KeyError:
                pass
        # ====================================================================

        # ====================================================================
        # UNZIP THE VALUES IN 'det_params" and "char_params" to a one-to-one
        # mapping with each planet.
        # remove_astropy strips each value of its astropy quantity... cause why the
        # hell is that in the output?!!

        if ddrm.has_key('det_params'):
            det_DRM = list(ddrm['det_params'])
            det_DRM = remove_astropy(det_DRM)

            for ky in det_expandkeys:
                arr = np.array([dt['WA'] for dt in det_DRM])
                arr = np.concatenate(arr)
                ddrm['det_' + ky] = arr
            # REMOVE Zipped dicitonary
            ddrm.pop('det_params');

        if ddrm.has_key('char_params'):
            char_DRM = list(ddrm['char_params'])
            char_DRM = remove_astropy(char_DRM)

            for ky in char_expandkeys:
                arr = np.array([dt['WA'] for dt in char_DRM])
                arr = np.concatenate(arr)
                ddrm['char_' + ky] = arr
            # REMOVE Zipped dicitonary
            ddrm.pop('char_params');

    else:
        ddrm = None

    return ddrm


def remove_astropy(DRM):
    """
    Given the DRM from an EXOSIMS simulation, strip the top level
    astropy quantity to bare numbers, because why the hell is an
    astropy quant IN THE DAMN OUTPUT?!!
    """

    kys = np.array([dt.keys() for dt in DRM])
    kys = np.unique(np.concatenate(kys))

    if DRM:
        for i in xrange(len(DRM)):

            for ky in kys:
                if isinstance(DRM[i][ky], astropy.units.quantity.Quantity):
                    DRM[i][ky] = DRM[i][ky].value

    return DRM


def varcsvOut(data):
    with open('Input_leftover_params.csv', 'wb') as myfile:
        w = csv.writer(myfile, delimiter=',')

        for key, val in data.items():

            if type(val) == list:
                i = 1
                with open('Input_{}.csv'.format(key), 'wb') as myfile_list:
                    wl = csv.writer(myfile_list)
                    for aitems in data[key]:
                        wl.writerow(['{}'.format(key), '{}'.format(i)])
                        for keyl, vall in aitems.items():
                            wl.writerow([keyl, vall])
                            myfile_list.flush()
                        i += 1

            elif type(val) == dict:
                with open('Input_{}.csv'.format(key), 'wb') as myfile_dict:
                    wd = csv.writer(myfile_dict)
                    for keyd, vald in data[key].items():
                        wd.writerow([keyd, vald])
                        myfile_dict.flush()

            else:
                w.writerow([key, val])
                myfile.flush()


def varcsvOut2(data):
    """
    Formats the genoutspec from the simulation into a 2 column string 
     of keys and values to be saved in the full csv output file
    
    
    Input:
        data: 

    Returns:

    """

    st_leftover = ['# ----- Other_Inputs---------\n']
    st_list = []
    st_dict = []
    for key, val in data.iteritems():

        if type(val) == list:
            i = 1
            st_list.append('# ----- Input_{}---------\n'.format(str(key)))
            for aitems in data[key]:
                st_list.append('#!! {},{}!!\n'.format(key, i))
                for keyl, vall in aitems.iteritems():
                    st_list.append('# {},{}\n'.format(keyl, str(vall)))
                i += 1

        elif type(val) == dict:
            st_dict.append('## ----- Input_{}---------\n'.format(key))

            for keyd, vald in data[key].items():
                st_dict.append('# {}, {}\n'.format(keyd, str(vald)))


        else:
            st_leftover.append('# {}, {}\n'.format(key, str(val)))

    return st_leftover, st_list, st_dict


def get_specs(jfile):
    script = open(jfile).read()
    specs = json.loads(script)
    return specs


def format_args(args):
    # KEEP ALL BUT FIRST ELEMENT OF ARGV.
    invar = np.array(args[1:])
    # REMOVES ALL DASHES
    invar2 = np.array([term.strip('-') for term in invar])
    invar3 = dict(np.resize(invar2, (invar2.size / 2, 2)))
    return invar3


# ================================================================================

# PT = mt.PlottingTools() RP -- DON'T NEED THIS ANYMORE.

# ==============================================================================
#                     FOLDERS
# ==============================================================================
# baseFolder = '/var/www/wfirst-dev/html/sims/tools/EXOSIMS-Testing/'
baseFolder = os.path.join(os.path.expanduser('~'), 'Dropbox', 'Research', 'WFIRST', 'EXOSIMSTesting')
resultFolder = os.path.join(baseFolder, 'SimResults')
# resultFolder = ''
scriptFolder = os.path.join(baseFolder, 'scripts')
# JX compFolder = os.path.join(baseFolder, 'Completeness')
# ==============================================================================

# ==============================================================================
#                 INPUT VARIABLE DICTIONARY
# ==============================================================================

# USE THIS FOR LOCAL TESTING
args = ['/var/www/wfirst-dev-15-08-07/html/sims/tools/exosimsCGI/bin/EXOSIMS_end2end+outputs.py',
        '-missionLife', '5.5', '-missionPortion', '0.3', '-extendedLife', '0.0', '-missionStart', '60676.0', '-dMagLim',
        '20', '-settlingTime', '0.03', '-minComp', '0.1', '-telescopeKeepout', '10.0', '-intCutoff', '50.0', '-eta',
        '0.1', '-ppFact', '0.3', '-FAP', '0.0000003', '-MDP', '0.001', '-TargetType', 'KnownRV', '-Dlam', '550.0',
        '-DBW', '0.1', '-DSNR', '5.0', '-DohTime', '0.1', '-Clam', '600.0', '-CBW', '0.1', '-CSNR', '5.0', '-CohTime',
        '0.1', '-CatType', 'EXOCAT1']  # ,'-seed','898150027']
inputVars = format_args(args)

# USE THIS FOR UI
# args = sys.argv
# inputVars = format_args(sys.argv)

# ==============================================================================
#                   JSON INPUTS
# ==============================================================================

# BASE PARAMETERS
jfileBase = os.path.join(scriptFolder, 'baseParams.json')
# OBSERVING MODE FILE
jfileObsMode = os.path.join(scriptFolder, 'baseObservingModes.json')
# MODULES FILE
jfileModules = os.path.join(scriptFolder, 'baseModules.json')

# LOAD THE SPECS FROM EACH FILE
modSpecs = get_specs(jfileModules)
obsSpecs = get_specs(jfileObsMode)
baseParams = get_specs(jfileBase)
useSpecs = baseParams.copy()

# ===================================================================================================
#                            SEED TEMPLATE JSON FILE WITH UI INPUT PARAMETERS
# ===================================================================================================

# THIS PIECE UPDATES THE INPUT VARIABLE FILE WITH INPUTS FROM THE USER INTERFACE
osmodekeys = ['lam', 'BW', 'SNR']

# REPLACE UNIVERAL PARAMS WITH INPUT VARS
for ikey, ivar in inputVars.iteritems():

    try:
        useSpecs[ikey] = float(ivar)
        if ikey == 'seed':
            useSpecs[ikey] = int(ivar)
    except ValueError:
        pass  # print '%s=%s cant be converted to float' % (ikey, ivar)
        # if ikey in useSpecs:
        #     try:
        #         useSpecs[ikey] = float(ivar)
        #     except ValueError:
        #         pass #print '%s=%s cant be converted to float' % (ikey, ivar)
# REPLACE INPUT OF BAND INFO AND CREATE OBSERVING MODE ARRAY
for okey in osmodekeys:
    try:
        obsSpecs['HLC_Detection'][okey] = float(inputVars['D' + okey])
    except ValueError:
        pass  # print '%s=%s cant be converted to float' % (okey, inputVars['D' + okey])
    try:
        obsSpecs['SPC_Characterization'][okey] = float(inputVars['C' + okey])
    except ValueError:
        pass  # print '%s=%s cant be converted to float' % (okey, inputVars['D' + okey])

# CREATES LIST OF 2 OBSERVING MODES
useSpecs['observingModes'] = [obsSpecs['HLC_Detection'],
                              obsSpecs['SPC_Characterization']]

useSpecs['modules'] = modSpecs[inputVars['TargetType']]

if inputVars['TargetType'] is not 'KnownRV':
    useSpecs['modules']['StarCatalog'] = inputVars['CatType']

# ===================================================================================================
#                            RUN SIMULATION
# ===================================================================================================

try:
    sim = msim.MissionSim(**useSpecs)
    sim.SurveySimulation.run_sim()
except IndexError:
    print 'Target List was filtered out to nothingness. Try again please.'
    raise SystemExit

# Stored detection information -- if any -- in DRM Dictionary
DRM = sim.SurveySimulation.DRM

# Calculate number of stars observed with one pl
# anet.
Nstar_obs_Wplanet = len([DRM[x]['star_ind'] for x in range(len(DRM)) if 1 in DRM[x]['det_status']])
print 'Number of Stars Observed with at least 1 planet detected: ', Nstar_obs_Wplanet

# Check to see if any planets/DRM is empty
if not DRM:
    ANYVISITS = False
else:
    ANYVISITS = True

# ***************************
# REFORMATTED DDRM
if ANYVISITS:
    DRM = remove_astropy(DRM)
    DDRM = reformat_DRM(DRM)
# ***************************
# Simulation specifications ; i.e., all parameters used in simulation
AllSpecs = sim.genOutSpec()

TL = sim.TargetList
SC = sim.StarCatalog
SU = sim.SimulatedUniverse
SSim = sim.SurveySimulation
OS = sim.OpticalSystem
ZL = sim.ZodiacalLight
BS = sim.BackgroundSources
CP = sim.Completeness
PP = sim.PlanetPopulation
PM = sim.PlanetPhysicalModel

Name = TL.Name
Spec = TL.Spec
parx = TL.parx
Umag = TL.Umag
Bmag = TL.Bmag
Vmag = TL.Vmag
Rmag = TL.Rmag
Imag = TL.Imag
Jmag = TL.Jmag
Hmag = TL.Hmag
Kmag = TL.Kmag
dist = TL.dist
BV = TL.BV
MV = TL.MV
BC = TL.BC
L = TL.L
coords = TL.coords
pmra = TL.pmra
pmdec = TL.pmdec
rv = TL.rv
Binary_Cut = TL.Binary_Cut
# maxintTime = OS.maxintTime
comp0 = TL.comp0
MsEst = TL.MsEst
MsTrue = TL.MsTrue
nStars = TL.nStars

star_prop = {'Name': Name, 'Spec': Spec, 'parx': parx, 'Umag': Umag, 'Bmag': Bmag, 'Vmag': Vmag,
             'Imag': Imag, 'Jmag': Jmag, 'Hmag': Hmag, 'Kmag': Kmag, 'dist': dist, 'BV': BV,
             'MV': MV, 'Lum': L, 'coords': coords, 'pmra': pmra, 'pmdec': pmdec, 'rv': rv,
             'Binary_Cut': Binary_Cut, 'comp0': comp0, 'MsEst': MsEst,
             'MsTrue': MsTrue, 'nStars': nStars}  # ,'maxintTime':maxintTime}

try:
    arange = TL.arange
    erange = TL.erange
    wrange = TL.wrange
    Orange = TL.Orange
    prange = TL.prange
    Irange = TL.Irange
    Rrange, Mprange = TL.Rrange, TL.Mprange
    rrange = TL.rrange

except AttributeError:
    pass

try:
    nPlans, plan2star = SU.nPlans, SU.plan2star
    sInds = SU.sInds
    # ORBTIAL PARAMETERS
    sma, ecc, wangle, Oangle, Iangle = SU.a, SU.e, SU.w, SU.O, SU.I
    # PLANET PROPERTIES
    Mp, Rp = SU.Mp, SU.Rp
    # POSITION AND VELOCITY VECTOR OF PLANET
    r, v = SU.r, SU.v
    # ALBEDO
    palbedo = SU.p
    fEZ = SU.fEZ


except AttributeError:
    pass

Mprange = AllSpecs['Mprange']
arange = AllSpecs['arange']

# CHECK IF DDRM IS EMTPY = NO PLANETS DETECTED.
if ANYVISITS:
    # indices of targets observed in order including repeats
    target_obsind = DDRM['star_ind']

    # indices of all planets detected for each star.
    try:
        detind_pl = DDRM['plan_inds']
    # detind_pl = np.concatenate(detind_pl).astype('int32')
    except:
        NOPLANETS = True
        pass  # print 'No planets detected?'

    # arrival time array
    arrival_time = DDRM['arrival_time']

    # angular distance of each detection ? mas\
    det_wa = DDRM['det_WA']

    # status of any detections
    det_status = DDRM['det_status']

    # detection of planets
    target_observed = Name[target_obsind]
    detstatus_array = np.array([dt['det_status'] for dt in DRM])
else:
    pass  # print 'No Planets detected.'

# =========================================================================================
#             OUTPUT CSV FILES
# =========================================================================================
leftover, slist, sdict = varcsvOut2(AllSpecs)
comments = [string.join(sdict), string.join(slist), string.join(leftover)]
comments = string.join(comments)
hdr = 'st_name,st_spt,st_parallax,st_Umag,st_Bmag,st_Vmag,st_Rmag,st_Imag,st_Jmag,st_Hmag,st_Kmag,st_distance,st_B-V,st_M_v,st_BC,st_Lum'
stardat = np.array([Name, Spec, parx,
                    Umag, Bmag, Vmag, Rmag, Imag, Jmag, Hmag, Kmag,
                    dist, BV, MV, BC, L])

if ANYVISITS:
    hdr += ',pl_mass,pl_rad,pl_sma,pl_ecc,pl_wangle,pl_albedo,pl_phi,pl_incl,pl_fEZ,pl_dMag,pl_WA'
    planetdat = np.array([Mp.to(con.M_jup).value, Rp.to(con.R_jup).value, sma.value, ecc, wangle.value,
                          palbedo, SU.phi, Iangle.value, fEZ, SU.dMag, SU.WA.value])
    write_array = stardat[:, DDRM['star_ind']]
    write_array = np.vstack((write_array, planetdat[:, DDRM['plan_inds']]))

    for key, val in DDRM.iteritems():
        if key not in ['plan_inds', 'star_ind', 'char_mode']:
            hdr += ',{}'.format(key)
            dat = DDRM[key]
            write_array = np.vstack((write_array, dat))

else:
    write_array = stardat

csvsavef = os.path.join(resultFolder, 'simResults.csv')

np.savetxt(csvsavef, write_array.T, fmt='%s',
           header=hdr, delimiter=',', comments=comments)
# ===================================================================================================


# ===================================================================================================
#                 HEADER INFORMATION - NEEDS TO BE UPDATED
# ===================================================================================================
hdr_info = [('st_name', 'Star ID'),
            ('st_spt', 'Stellar Spectral type'),
            ('st_parallax', 'Stellar parallax (mas)'),
            ('st_Umag', 'Stellar U mag (mag)'),
            ('st_Bmag', 'Stellar B mag (mag)'),
            ('st_Vmag', 'Stellar V mag (mag)'),
            ('st_Rmag', 'Stellar R mag (mag)'),
            ('st_Imag', 'Stellar I mag (mag)'),
            ('st_Jmag', 'Stellar J mag (mag)'),
            ('st_Hmag', 'Stellar H mag (mag)'),
            ('st_Kmag', 'Stellar K mag (mag)'),
            ('st_distance', 'Stellar Distance (pc)'),
            ('st_B-V', 'Stellar B-V color (mag)'),
            ('st_M_v', 'Stellar absolute V mag (mag)'),
            ('st_BC', 'Stellar bolometric correction'),
            ('pl_mass', 'Planet mass (M_jupiter)'),
            ('pl_rad', 'Planet radius (R_jupiter)'),
            ('pl_sma', 'Planet semi-major axis (AU)'),
            ('pl_ecc', 'Planet eccentricity'),
            ('pl_wangle', 'Planet star-separation [working angle] (arcsec)'),
            ('pl_albedo', 'Planet albedo'),
            ('pl_phi', 'Planet phase angle'),
            ('pl_incl', 'Planet inclination'),
            ('pl_fEZ', 'Surface brightness o fexozodicacal light (1/arcsec^2)'),
            ('pl_dMag', 'planet to star magnitude difference (mag)'),
            ('pl_WA', 'planet working angle at detection (mas)'),
            ('char_status',
             ' characterization status for each planet orbiting the observed target star where 1 is full spectrum -1 partial spectrum and 0 not characterized'),
            (
            'det_WA', ' working angles at detection in units of mas for each planet orbiting the observed target star'),
            ('char_time', 'Integration time for characterization in units of day'),
            ('char_fEZ',
             ' exo-zodi surface brightnesses at characterization in units of 1/arcsec2 for each planet orbiting the observed target star'),
            ('det_fEZ',
             'exo-zodi surface brightnesses at detection in units of 1/arcsec2 for each planet orbiting the observed target star.'),
            ('det_time', 'Integration time for detection in units of day'),
            ('arrival_time', 'Elapsed time since mission start when observation begins in units of day'),
            ('det_status',
             ' detection status for each planet orbiting the observed target star where 1 is detection 0 missed detection -1 below IWA and -2 beyond OWA'),
            ('char_dMag', ' delta magnitudes at characterization for each planet orbiting the observed target star'),
            ('det_SNR',
             ' detection SNR values for each planet orbiting the observed target star. Non-observable planets have their SNR set to 0.'),
            ('char_SNR',
             'characterization SNR values for each planet orbiting the observed target star. Non-observable planets have their SNR set to 0.'),
            ('char_WA',
             'working angles at characterization in units of mas for each planet orbiting the observed target star'),
            ('det_dMag', ' delta magnitudes at detection for each planet orbiting the observed target star')]

hdr_info2 = collections.OrderedDict(hdr_info)

hdrinfo_file = os.path.join(resultFolder, 'headerinfo.csv')

f = open(hdrinfo_file, 'w')
f.write('header,description\n')
for key, val in hdr_info2.iteritems():
    f.write('{},{}\n'.format(key, val))
f.close()

# ===================================================================================================
#                 PLOTS
# ===================================================================================================
save_plots = True

# Set up Data
dirsave = ''
# JX dirsave = os.path.join(baseFolder,'SimPlots')

ra = coords.ra.deg
ra[ra > 180] -= 360
ra = -1 * ra
ra_rad, dec_rad = np.radians(ra), coords.dec.radian
tick_labels = np.array([150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210])
tick_labels = np.remainder(tick_labels + 360, 360)

Rj = 69.911e6  # meters
Mj = 1.898e27  # Kg

# BINS & DATA
ebins = np.linspace(0, 1, 7)
logarange = np.log10(arange)
smabins = np.logspace(logarange[0], logarange[1], 10)
massbins = np.linspace(.001, 50, 20)
# ===================================================================================================
#                 Sky Plots
# ===================================================================================================


# if FOUND_PLANETS:
# SKY DISTRIBUTION OF TARGETS
plot_num = 1  # RP : PLOT NUMBER

axt = skyplot(ra_rad, dec_rad, axisbg='#E8E8E8', ticklabels=tick_labels)

plt.show()
try:
    plot_setup(axt)
except:
    pass
plt.grid(color='black', lw=2.5)
axt.plot(ra_rad, dec_rad, 'o', color='#00cc00', ms=10, mec='none', label='Stars in survey')

if ANYVISITS:
    print 'yes'
    axt.plot(ra_rad[target_obsind], dec_rad[target_obsind], 'ms', ms=15, mec='m', mew=3, mfc='none')
else:
    print 'NO'
    axt.text(-2, 3, r'NO SYSTEMS WERE OBSERVED', fontsize=30)

txt = """RA vs. DEC OF ALL STARS IN THE SIMULATION (green) \n AND THOSE THAT WERE OBSERVED (red)"""
axt.set_title('%s' % txt, fontsize=30, y=1.03)
if save_plots:
    plt.savefig(os.path.join(dirsave, 'fig%i.png' % plot_num), bbox_inches='tight')
    plot_num += 1

# ================================================================================
#        OBSERVATION ORDER LINES
# ================================================================================
axt3 = skyplot(ra_rad, dec_rad, axisbg='#E8E8E8', ticklabels=tick_labels)
try:
    plot_setup(axt3)
except:
    pass
plt.grid(color='black', lw=2.5)
axt3.plot(ra_rad, dec_rad, 'o', color='#00cc00', ms=10, mec='none')

if ANYVISITS:
    alp = np.logspace(-1, np.log10(0.6), len(target_obsind))  # [::-1]
    for i in xrange(len(target_obsind)):
        if i == len(target_obsind) - 1:
            pass
        else:
            axt3.plot([ra_rad[target_obsind[i]], ra_rad[target_obsind[i + 1]]],
                      [dec_rad[target_obsind[i]], dec_rad[target_obsind[i + 1]]],
                      'm-', mew=2, lw=7, alpha=alp[i])
    for i, j in enumerate(target_obsind):
        if i == 0:
            axt3.plot(ra_rad[j], dec_rad[j], 'm*', ms=20, mec='none')

else:
    axt3.text(-2, 3, r'NO SYSTEMS WERE OBSERVED', fontsize=30)

axt3.xaxis.label.set_color('lightgray')
axt3.tick_params(axis='both', colors='lightgray')

txt = """RA vs. DEC OF ALL STARS IN THE SIMULATION (green) \n AND THOSE THAT WERE OBSERVED (red).
ORDER OF OBSERVATIONS SHOWN BY TRACKS (light to dark)."""
axt3.set_title('%s' % txt, fontsize=30, y=1.03)

axt3.tick_params(labelbottom='off', top='off', which='both', labelleft='off')

if save_plots:
    plt.savefig(os.path.join(dirsave, 'fig%i.png' % plot_num), bbox_inches='tight')
    plt.savefig(os.path.join(dirsave, 'fig%i.pdf' % plot_num), bbox_inches='tight')
    plot_num += 1

# ==================================================================================================================
#                           CMD
# ==================================================================================================================
kw = {'markersize': 15, 'color': '#00cc00', 'marker': 'o', 'alpha': .6, 'mec': 'none'}
axbv = scatterPlot(BV, MV, figsize=(15, 15), **kw)
if ANYVISITS:
    axbv.plot(BV[target_obsind], MV[target_obsind], 'ms', markersize=16, mec='m', mew=3, alpha=0.7, mfc='none')
else:
    axbv.text(np.average(BV), np.average(MV), 'NO SYSTEMS WERE OBSERVED', fontsize=30)
axbv.set_ylim(axbv.get_ylim()[::-1])
axbv.set_xlabel(r'B-V [mag]', fontsize=35)
axbv.set_ylabel(r'$M_V$ [mag]', fontsize=35)

txt = """COLOR-MAGNITUDE DIAGRAM OF ALL \nAND OBSERVED STARS IN SIMULATION."""
axbv.set_title('%s' % txt, fontsize=30, y=1.03)
if save_plots:
    plt.savefig(os.path.join(dirsave, 'fig%i.png' % plot_num), bbox_inches='tight', edgecolor='none')
    plt.savefig(os.path.join(dirsave, 'fig%i.pdf' % plot_num), bbox_inches='tight', edgecolor='none')
    plot_num += 1

# ==================================================================================================================
#            CONTRAST VS. SEPARATION PLOT
# ==================================================================================================================


# ********************************* Planet Plots ***************************



# CONTRAST CURVES
if ANYVISITS:
    WA = DDRM['det_WA']
    cntrst = 10 ** (DDRM['det_dMag'] / (-2.5))
    kw = {'markersize': 20, 'color': 'r', 'marker': 'o', 'alpha': .6, 'mec': 'none'}
    axcc = scatterPlot(WA, cntrst, figsize=(15, 15), **kw)
    WAlims = WA[~np.isnan(WA)]
    cnlims = cntrst[~np.isnan(cntrst)]
    axcc.set_xlim(WAlims.min() * .8, WAlims.max() * 1.1)
    axcc.set_ylim(cnlims.min() * .8, cnlims.max() * 1.1)
    axcc.set_xscale('log')
    axcc.set_yscale('log')


else:
    axcc = plt.figure(figsize=(15, 15)).add_subplot(111)
    axcc.text(0.1, 0.5, 'NO SYSTEMS WERE OBSERVED', fontsize=30)
axcc.set_ylabel(r'Planet-Star Flux Ratio', fontsize=35)
axcc.set_xlabel(r'Working Angle (mas)', fontsize=35)

txt = """CONTRAST VS. WORKING ANGLE OF DETECTED PLANETS.\n IWA=0.0 mas, OWA=inf"""
axcc.set_title('%s' % txt, fontsize=30, y=1.03)
axcc.grid()

if save_plots:
    plt.savefig(os.path.join(dirsave, 'fig%i.png' % plot_num), bbox_inches='tight', edgecolor='none')
    plt.savefig(os.path.join(dirsave, 'fig%i.pdf' % plot_num), bbox_inches='tight', edgecolor='none')
    plot_num += 1

else:
    print 'No planets detected, so no plots were produced. Check input parameters.'

print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
print 'Simulation Complete.'
if not ANYVISITS:
    print 'NO STARS WERE OBSERVED.'
print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
