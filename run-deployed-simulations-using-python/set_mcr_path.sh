#!/bin/bash
# Usage: source set_mcr_path.sh /usr/local/MATLAB/MATLAB_Runtime/R2025b
# This script updates LD_LIBRARY_PATH for MATLAB Runtime libraries.
# See: https://www.mathworks.com/help/compiler/mcr-path-settings-for-run-time-deployment.html

if [ -z "$1" ]; then
  echo "Usage: source $0 <MCR_ROOT_PATH>"
  return 1 2>/dev/null || exit 1
fi

MCR_ROOT="$1"

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}\
${MCR_ROOT}/runtime/glnxa64:\
${MCR_ROOT}/bin/glnxa64:\
${MCR_ROOT}/sys/os/glnxa64:\
${MCR_ROOT}/extern/bin/glnxa64"

echo "LD_LIBRARY_PATH set for MATLAB Runtime at: ${MCR_ROOT}"
