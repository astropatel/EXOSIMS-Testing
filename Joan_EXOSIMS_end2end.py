import EXOSIMS as EX, os.path
import EXOSIMS.MissionSim as msim
import EXOSIMS.StarCatalog.EXOCAT1 as exc
import numpy as np
import math as ma
import matplotlib.pyplot as plt
import astropy.units as u
import pickle, json, warnings, astropy
#%matplotlib

# LOAD JSON SCRIPT FILE
#jfile = 'template_WFIRST_KeplerLike.json'
#jfile = 'template_WFIRST_EarthTwinHabZone.json'
#jfile = 'template_WFIRST_KnownRV.json'
#jfile = 'template_rpateltest_KnownRV.json'
jfile = 'template_rpateltest_KnownRV_2years.json'
scriptfile = os.path.join(os.path.abspath(''),'scripts',jfile)
print scriptfile

script = open(scriptfile).read()
specs_from_file = json.load(script)

# QUESTION -- DO I HAVE TO RUN THIS EACH TIME OR IS THERE A WAY TO SAVE/LOAD THE OUTPUT?
#JX %time sim = msim.MissionSim(scriptfile)
sim = msim.MissionSim(scriptfile) #JX
#JX %time sim.SurveySimulation.run_sim()
res=sim.SurveySimulation.run_sim() #JX

# Stored detection information -- if any -- in DRM Dictionary
DRM = sim.SurveySimulation.DRM

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

Name   = TL.Name
Spec   = TL.Spec
parx   = TL.parx
Umag   = TL.Umag
Bmag   = TL.Bmag
Vmag   = TL.Vmag
Rmag   = TL.Rmag
Imag   = TL.Imag
Jmag   = TL.Jmag
Hmag   = TL.Hmag
Kmag   = TL.Kmag
dist   = TL.dist
BV = TL.BV
MV = TL.MV
BC = TL.BC
L  = TL.L
coords = TL.coords
pmra   = TL.pmra
pmdec  = TL.pmdec
rv = TL.rv
Binary_Cut = TL.Binary_Cut
#maxintTime = OS.maxintTime
comp0  = TL.comp0
MsEst  = TL.MsEst
MsTrue = TL.MsTrue
nStars = TL.nStars

star_prop = {'Name':Name,'Spec':Spec,'parx':parx,'Umag':Umag,'Bmag':Bmag,'Vmag':Vmag,
    'Imag':Imag,'Jmag':Jmag,'Hmag':Hmag,'Kmag':Kmag,'dist':dist,'BV':BV,
    'MV':MV,'Lum':L,'coords':coords,'pmra':pmra,'pmdec':pmdec,'rv':rv,
    'Binary_Cut':Binary_Cut,'comp0':comp0,'MsEst':MsEst,
    'MsTrue':MsTrue,'nStars':nStars}#,'maxintTime':maxintTime}


try:
    arange = TL.arange
    erange = TL.erange
    wrange = TL.wrange
    Orange = TL.Orange
    prange = TL.prange
    Irange = TL.Irange
    Rrange, Mprange = TL.Rrange, TL.Mprange
    rrange = TL.rrange
    synplanet_prop = {'arange':arange,'erange':erange,'wrange':wrange,'Orange':Orange,
        'prange':prange,'Irange':Irange,'Rrange':Rrange,'Mprange':Mprange,
        'rrange':rrange}
except AttributeError:
    print 'Sorry.. no go.'
    print 'Not simulated planets.'
    synplanet_prop = None

try:
    nPlans, plan2star = SU.nPlans, SU.plan2star
    sInds = SU.sInds
    # ORBTIAL PARAMETERS
    sma,e,w,O,I = SU.a, SU.e, SU.w, SU.O, SU.I
    # PLANET PROPERTIES
    Mp,Rp = SU.Mp, SU.Rp
    # POSITION AND VELOCITY VECTOR OF PLANET
    r, v = SU.r, SU.v
    # ALBEDO
    p = SU.p
    fEZ = SU.fEZ

    empplanet_prop = {'nplans':nPlans,'plan2star':plan2star,'sInds':sInds,'sma':sma,
    'e':e,'w':w,'O':O,'I':I,'Mp':Mp,'Rp':Rp,'r':r,'v':v,'p':p,'fEZ':fEZ}
except AttributeError:
    print 'Sorry.. no go.'
    print 'Not ``real`` planets.'
    empplanet_prop = None

mlife = AllSpecs['missionLife']
#nsimplanets = specs_from_file['Nplanets']
jfilebase = jfile.strip('.json').strip('template_')

#simresults = 'simresults_%iyrs_%.0Estars_%s.pickle' %(mlife,nsimplanets,jfilebase)
simresults = 'simresults_%.2fyrs_%s.pickle' %(mlife,jfilebase)
#simfile = os.path.join('/Users/jxie/EXOSIMS-Testing/SimResults/',simresults)
simfile = os.path.join('/SimResults',simresults)

data = {'DRM':DRM,'empplanet_prop':empplanet_prop,'synplanet_prop':synplanet_prop,
    'star_prop':star_prop,'AllSpecs':AllSpecs}#,'etc_data':etc_data}

handler = open(simfile,'wb')
pickle.dump(data,handler)
handler.close()

simresults