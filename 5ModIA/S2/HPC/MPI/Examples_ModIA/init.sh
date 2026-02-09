#!/bin/bash

# à modifier en indiquant où est installé simgrid
# SIMGRID=/usr/include/simgrid

# export PATH=${SIMGRID}/bin:${PATH}

alias Smpirun="smpirun -hostfile ${PWD}/archis/cluster_hostfile.txt -platform ${PWD}/archis/cluster_crossbar.xml"
