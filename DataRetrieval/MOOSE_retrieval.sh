#!/bin/bash -l
#SBATCH --mem=500
#SBATCH --ntasks=4
#SBATCH --qos=long
#SBATCH --time=2880
#SBATCH --export=NONE

suite=$1
startyr=$2
endyr=$3
wrkdir=$(pwd)
echo $wrkdir
cp filterspec filterspec_$suite
sed -i 's/startyear/'$startyr'/g' $wrkdir/filterspec_$suite
sed -i 's/endyear/'$endyr'/g' $wrkdir/filterspec_$suite
mkdir -p $SCRATCH/ML/ModelData/$suite/apm/
cd $SCRATCH/ML/ModelData/$suite/apm/
echo 'Retrieving from MOOSE'
echo $suite
echo apm
echo 'from ' $startyr ' to ' $endyr
moo select -i $wrkdir/filterspec_$suite moose:/crum/$suite/apm.pp/ . > MOOSE_retrieval_$suite_apm_output.txt
cd $wrkdir
echo 'Retrieval from MOOSE complete'
