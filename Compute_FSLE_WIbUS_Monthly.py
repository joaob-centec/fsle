from parcels import FieldSet, ParticleSet, JITParticle, AdvectionRK4, ErrorCode
import xarray as xr
import numpy as np
import pandas as pd
#from pyproj import Geod
import datetime
#import cartopy.crs as ccrs
#import cartopy.feature as cfeature
#import matplotlib.pyplot as plt
import argparse
import os
import glob
import matplotlib.dates

## Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("fYear", type=int, help="year of computation")
parser.add_argument("fMonth", type=int, help="month of computation")
parser.add_argument("fStep", type=int, help="FSLE field time step (days)")
parser.add_argument("outStep", type=int, help="trajectory output time step (hours)")
parser.add_argument("maxD", type=float, help="distance threshold")
parser.add_argument("iniD", type=int, help="initial distance (grid subsample)")
parser.add_argument("maxT", type=int, help="max trajectory integration interval (days)")
parser.add_argument("timeStep", type=float, help="trajectory integration time step (minutes)")
parser.add_argument("westLon", type=float, help="westernmost longitude (-180º to 180º)")
parser.add_argument("eastLon", type=float, help="easternmost longitude (-180º to 180º)")
parser.add_argument("southLat", type=float, help="southermost latitude (-90º to 90º)")
parser.add_argument("northLat", type=float, help="northermost latitude (-90º to 90º)")
parser.add_argument("z", type=float, help="depth (positive downwards)")

args = parser.parse_args()

# 1. Distance threshold: 
maxD=float(args.maxD) # in m
# 2. Initial distance (as a function of velocity grid spacing): 
iniD = args.iniD
# 3. Maximum integration time interval: 
maxT=datetime.timedelta(days=args.maxT)
# 4. Integration timestep (<0 --> backward FSLE)
timeStep=-datetime.timedelta(minutes=args.timeStep)

# Initial time of the velocity field
vInitialDate=datetime.datetime(1999,6,1,12,0,0)

# FSLE grid 
lonMin= args.westLon  # ºE
lonMax= args.eastLon   # ºE
latMin= args.southLat   # ºN
latMax= args.northLat   # ºN
zMin= 0      # Surface
zMax= 300   # m

# Compute FSLE for year/month

fYear = args.fYear
fMonth = args.fMonth

# FSLE field step in days

fStep = args.fStep

# FSLE field dates


# Particle trajectory output step in hours

outStep = args.outStep

# Echo input parameters

print("FSLE computation with following parameters: ")
print("  fYear: "+str(fYear))
print("  fMOnth: "+str(fMonth))
print("  maxD:"+str(maxD))


# 3.1 Get velocity grid lon/lat
# Velocity grid file 
vGridFileName = 'IBI_Data/IBI_MULTIYEAR_PHY_005_002-TDS_199906.nc'

# Velocity grid variables names
vLat = "latitude"
vLon = "longitude"
vDepth = "depth"

# Load velocity grid data
vGridData = xr.open_dataset(vGridFileName)

# Show info of the data 
#print(vGridData)

# Velocity grid lon/lat limited to FSLE grid limits
tmpL = vGridData.latitude.sel(latitude  = slice(latMin, latMax))
fLat0 = xr.DataArray(data=tmpL.values, dims=["j"], coords=[np.arange(tmpL.size)])

tmpL = vGridData.longitude.sel(longitude  = slice(lonMin, lonMax))
fLon0 = xr.DataArray(data=tmpL.values, dims=["i"], coords=[np.arange(tmpL.size)])

fZ = vGridData.depth.sel(depth  = slice(zMin, zMax))

# Interpolate latitude and longitude to get FLSE grid nodes

fJ = np.linspace(0,fLat0.size-1,fLat0.size*iniD-1,endpoint=True)

fI = np.linspace(0,fLon0.size-1,fLon0.size*iniD-1,endpoint=True)

fLat = fLat0.interp(j=fJ)

fLon = fLon0.interp(i=fI)

nF = fLat.size * fLon.size

nLat = fLat.size

nLon = fLon.size

# Create initial condition vectors

fLonI, fLatI = np.meshgrid(fLon,fLat,sparse=False,indexing='xy')

## Compute neighbor particle index grid

# Grid with particle number 
fNeig = np.reshape(np.arange(nF),fLonI.shape).astype(int)

# Arrays with neighbour particle indices

iNeig = np.zeros((nLat, nLon, 4),dtype=np.int16)

for j in range(nLat):
       for i in range(nLon):
                                   
            iNeig[j,i,0] = fNeig[j,min(i+1,nLon-1)]
            iNeig[j,i,1] = fNeig[j,max(i-1,0)]
            iNeig[j,i,2] = fNeig[min(j+1,nLat-1),i]
            iNeig[j,i,3] = fNeig[max(j-1,0),i]

# Grid with center particle distance (greatest distance between particle (i,j) and neighbour particles (i+1,j).(i-1,j),(i,j+1),(i,j-1))
fDist = np.zeros(fNeig.shape)
fIDist = np.zeros(fNeig.shape) # Initial distances

def haversine(lonC,latC,lonN,latN):
    # Computes the great-circle distance between particles.
    # Uses the Haversine formulas (http://www.movable-type.co.uk/scripts/gis-faq-5.1.html).
    # dlon = lon2 - lon1
    # dlat = lat2 - lat1
    # a = sin^2(dlat/2) + cos(lat1) * cos(lat2) * sin^2(dlon/2)
    # c = 2 * arcsin(min(1,sqrt(a)))
    # d = R * c
    
    distC=np.zeros(lonN.shape) # A nLat x nLon x 4 array
    
    # If any particle is deleted (position is nan,nan), set position to c particle, so that distance is 0
    for i in range(4):
        nanN = np.where(np.isnan(lonN[:,:,i]),True,False)
        lonN[nanN,i]=lonC[nanN]
        latN[nanN,i]=latC[nanN]

    for i in range(4):
        dLon = (lonN[:,:,i] - lonC)*np.pi/180.
        dLat = (latN[:,:,i] - latC)*np.pi/180.
        A = np.sin(0.5*dLat)**2 + np.cos(latC*np.pi/180) * np.cos(latN[:,:,i]*np.pi/180) * np.sin(0.5*dLon)**2
        C = 2 * np.arcsin(np.fmin(1,np.sqrt(A)))
        distC[:,:,i] = 6371000 * C
        
    return np.max(distC,axis=2)

# Compute initial distances (distances with obs=0)

eLon = np.zeros((nLat,nLon,4))
eLat = np.zeros((nLat,nLon,4))

for i in range(4):
    
    eLon[:,:,i] = np.reshape(fLonI.flat[iNeig[:,:,i].flatten()],(nLat,nLon))
    eLat[:,:,i] = np.reshape(fLatI.flat[iNeig[:,:,i].flatten()],(nLat,nLon))

fIDist = haversine(fLonI,fLatI,eLon,eLat)

# Set up the velocity fields in a FieldSet object

velocityFiles=sorted(glob.glob('/home/joao/Ciencia/FSLE_WIbUS/IBI_Data/IBI_MULTIYEAR_PHY_005_002-TDS_*.nc'))# + 

#fname = '/home/joao/Ciencia/FSLE_WIbUS/IBI_Data/*.nc'
filenames = {'U': velocityFiles, 'V': velocityFiles}
variables = {'U': 'uo', 'V': 'vo'}
dimensions = {'U': {'lat': 'latitude', 'lon': 'longitude', 'time': 'time'},
              'V': {'lat': 'latitude', 'lon': 'longitude', 'time': 'time'}}
fieldset = FieldSet.from_netcdf(filenames, variables, dimensions)

# Define recovery kernel
def DeleteParticle(particle, fieldset, time):
    # Delete particles who run out of bounds.
    #print("Particle %d deleted at (%f, %f, %f)" % (particle.id, 
    #                                               particle.lon, particle.lat,
    #                                              particle.depth))
    particle.delete()
    
## Compute the FSLE fields

fDate = datetime.datetime(fYear,1,1,12,0,0) # First FSLE field for 1st of January of fYear and then every fStep days

nowDate = datetime.datetime.now()

fDates = pd.date_range(datetime.datetime(fYear,fMonth,1,12,0,0),datetime.datetime(fYear,fMonth+1,1,12,0,0)-
   datetime.timedelta(days=1), freq=str(fStep)+'D')

nDates = fDates.size
  
fField = np.zeros((nLat, nLon, nDates))

k = 0    # Loop Counter

fFileTag = str(fYear)+"{:02d}".format(fMonth)

particlesFile="fParticles"+ fFileTag +".nc"

  while k < 1:#nDates:
    
    print("compute FSLE for " + fDates[k].strftime("%Y-%m-%d %H:%M:%S"))
    countTime = datetime.datetime.now()-nowDate
    countTimeDays = countTime.days
    countTimeHours = int(countTime.seconds/3600)
    countTimeMinutes = int((countTime.seconds/3600-countTimeHours)*60)
    countTimeSeconds = countTime.seconds - countTimeHours*3600 - countTimeMinutes*60
    
    print(" at " + str(countTimeDays) + " days, " + str(countTimeHours) + " hours, " 
          + str(countTimeMinutes) + " minutes, " + str(countTimeSeconds) + " seconds. ")
          
    ## Define the particles type and initial conditions in a ParticleSet object
    
    fSetTime = fDates[k]-vInitialDate       # Release date in seconds from vInitialDate
    
    fSet = ParticleSet(fieldset=fieldset,   # the fields on which the particles are advected
                   pclass=JITParticle,  # the type of particles (JITParticle or ScipyParticle)
                   lon=fLonI.flatten(), # release longitudes 
                   lat=fLatI.flatten(), # release latitudes
                   time=fSetTime.total_seconds(), # Release time (seconds from first time of velocity field )
                   depth=np.full(nF,args.z),         # release depth
                   )   
    
    output_file = fSet.ParticleFile(name=particlesFile, outputdt=3600*outStep) # the file name and the time step of the outputs
        
    fSet.execute(AdvectionRK4,                 # the kernel (which defines how particles move)
             runtime=maxT,                 # the total length of the run
             dt=timeStep,                  # the timestep of the kernel
             recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle},
             output_file=output_file)

     # 4. Exporting the simulation output to a netcdf file

    output_file.export()
    output_file.close() 
    
    # Load trajectory data

    fData = xr.open_dataset(particlesFile)
    #print(fData)
    #print(fData.z[dict(traj=5,obs=slice(0,100))])
    
    fCalc = np.ones(fDist.shape, dtype=bool) # Flag to indicate distance has reach maxD so we don't need to keep computing distances for this particle anymore

    #fT0 = np.reshape(fData.time[:,0].values,fLonI.shape)       # Initial time (we need this to compute elapsed time until distanced threshold is reached)

    fT = np.zeros(fLonI.shape)  # Save here the time distance has reached maxD
    fDist = np.zeros(fLonI.shape)  # Save here the distance 
    
    for m in np.arange(1,fData.dims['obs']):#fData.dims['obs']): # Loop over times
            
      # Active particles at observation m
      activeParticles = np.where(fCalc,True,False)
    
      # Filter active particles
      fTraj = np.reshape(fData.trajectory[:,m].values,(nLat,nLon))
    
      # Deleted active particles at observation m 
      deletedParticles = np.where(np.logical_and(np.isnan(fTraj),activeParticles), True, False)
    
      # Compute interparticle Distance
      cLon = np.reshape(fData.lon[:,m].values,(nLat,nLon))
      cLat = np.reshape(fData.lat[:,m].values,(nLat,nLon))
    
      for i in range(4):
        eLon[:,:,i] = np.reshape(fData.lon[:,m].values[iNeig[:,:,i].flatten()],(nLat,nLon))
        eLat[:,:,i] = np.reshape(fData.lat[:,m].values[iNeig[:,:,i].flatten()],(nLat,nLon))
        
      # Distance computed at observation m for those particles that are still active and have not been deleted
      distanceParticles = np.logical_and(np.logical_not(deletedParticles),activeParticles)
    
      fDistTmp = np.where(distanceParticles,haversine(cLon,cLat,eLon,eLat),0)
    
      # Where distance is greater than threshold, set final distance
      distFlag = np.where(fDistTmp>maxD, True, False)


# Set distance and time
    
      setDist = distFlag#, np.logical_not(deletedParticles)) 
    
      fDist[setDist]=fDistTmp[setDist]
    
      obsT=np.reshape(fData.time[:,m].values-fData.time[:,0].values,(nLat,nLon))
    
      fT[setDist]=obsT[setDist]
    
      filterParticle = np.logical_or(setDist, deletedParticles)
    
      fCalc[filterParticle]=False

      if m % 500 == 0:
        print(" Computing distances at observation " + str(m))
        
   ## Compute FSLE

    cff1=np.log(fDist/fIDist)

    cff2=-fT/(1000000000*86400.) # From nanoseconds to days

    #fField = cff1 / cff2

    fField[:,:,k] = cff1 / cff2#xr.DataArray(np.where(np.isinf(fField),0,fField), coords=[fLat, fLon], dims=["latitude", "longitude"])
    
    # Do some cleanup
    
    del fData
    os.remove(particlesFile) 
        
    # Step counter    
        
    k = k + 1
    
## Save FSLE data as xarray DataSet

#fsleData = xr.DataArray(fField, coords=[fLat, fLon, fDates], dims=["latitude", "longitude", "time"])
fsleData = xr.Dataset(
                {
                    "fsle":(["latitude","longitude","time"],fField)
                },
                coords={
                    "latitude":("latitude",fLat),
                    "longitude":("longitude",fLon),
                    "time":("time",fDates)
                }
)

fsleData["fsle"].attrs['units'] = 'day-1'
fsleData["fsle"].attrs['standard_name'] = 'finite-size lyapunov exponent'

fsleData.attrs['distance_threshold'] = maxD
fsleData.attrs['initial_distance (subsampling of background velocity field grid)'] = iniD
fsleData.attrs['maximum particle integration time (T0 + seconds)'] = maxT.total_seconds()
fsleData.attrs['particle integration time step'] = timeStep.total_seconds()
fsleData.attrs['particle trajectory output time step (hours)'] = outStep
fsleData.attrs['velocity grid source']=vGridFileName
fsleData.attrs['velocity data source']=velocityFiles

fsleOutputFile = "FSLE_WIbUS_" + fFileTag + ".nc" #str(fYear) + ".nc"

fsleData.to_netcdf(fsleOutputFile,encoding={"time": {"dtype": "double"}})


