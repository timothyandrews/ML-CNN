import iris
from iris.time import PartialDateTime
import iris.coord_categorisation
import sys
import numpy as np
import os

def process_stash(infiles,xstash,xstartyr,xendyr):
  print('Retrieving monthly diagnostic '+xstash+' from '+str(xstartyr)+' to '+str(xendyr)+'')
  constraint_stash=iris.AttributeConstraint(STASH=xstash)
  cube=iris.load_cube(infiles,constraint_stash)
  cube.remove_coord("forecast_period")
  cube.remove_coord("forecast_reference_time")
  cube.coord('time').bounds=None
  constraint_time=iris.Constraint(time=lambda cell: \
     PartialDateTime(year=xstartyr,month=1,day=1) <= cell <= \
     PartialDateTime(year=xendyr,month=12,day=30))
  cube=cube.extract(constraint_time)
  iris.coord_categorisation.add_year(cube,'time',name='year')
  iris.coord_categorisation.add_month(cube,'time',name='month')
  return cube


# Define some global attributes
suite='u-cw549'
exp='amip-piForcing'
startyr=1871 # first year of data
endyr=2019 # last year of data
DATADIR=os.getenv('DATADIR')
dirout=DATADIR+'/ML/data/'
model='HadGEM3-GC31-LL'
SCRATCH=os.getenv('SCRATCH')
infiles=SCRATCH+'/ML/ModelData/'+suite+'/apm/*.pp'

dTs=process_stash(infiles,'m01s00i024',startyr,endyr)
dT=process_stash(infiles,'m01s03i236',startyr,endyr)
drlut=process_stash(infiles,'m01s02i205',startyr,endyr)
drlutcs=process_stash(infiles,'m01s02i206',startyr,endyr)
drsdt=process_stash(infiles,'m01s01i207',startyr,endyr)
drsut=process_stash(infiles,'m01s01i208',startyr,endyr)
drsutcs=process_stash(infiles,'m01s01i209',startyr,endyr)

dN=drsdt-drlut-drsut
dLWcs=-1*drlutcs
dSWcs=drsdt-drsutcs
dLWcre=drlutcs-drlut
dSWcre=drsutcs-drsut
dNETcre=dLWcre+dSWcre

dTs.long_name='dTs'
dT.long_name='dT'
dSWcs.long_name='dSWcs'
dLWcs.long_name='dLWcs'
dSWcre.long_name='dSWcre'
dLWcre.long_name='dLWcre'
dN.long_name='dN'
dNETcre.long_name='dNETcre'
drlut.long_name='drlut'
drlutcs.long_name='drlutcs'
drsut.long_name='drsut'
drsutcs.long_name='drsutcs'
drsdt.long_name='drsdt'
iris.save([dTs,dT,dN,dSWcs,dLWcs,dSWcre,dLWcre,dNETcre,drlut,drlutcs,drsut,drsutcs,drsdt],''+dirout+exp+'_'+suite+'_'+model+'_Monthly.nc')
