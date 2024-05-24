wrkdir=$(pwd)
for var in dN dNETcre
do
mkdir -p $wrkdir/../Output/$var/Saves/
mkdir -p $wrkdir/../Output/$var/Figures/
mkdir -p $wrkdir/SpiceOutput/
cp slurm.template slurm_${var}.sh
sed -i 's/spice.out/spice_'$var'.out/g' $wrkdir/slurm_$var.sh
sed -i 's/err.out/err_'$var'.out/g' $wrkdir/slurm_$var.sh
sed -i 's/variable/'$var'/g' $wrkdir/slurm_$var.sh
sbatch slurm_$var.sh
rm slurm_$var.sh
done
