#!/bin/bash
suite1="u-cw549" # HG3 amip-piForcing extended
startyr="1870"
endyr="2020"
sbatch MOOSE_retrieval.sh $suite1 $startyr $endyr

