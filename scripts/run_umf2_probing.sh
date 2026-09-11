#!/usr/bin/env bash
# Kept for provenance: outputs/probing/*/*_umf2.json were produced under this
# name. The logic now lives in run_sweep_probing.sh, parameterised by METHOD.
METHOD=umf2 exec bash "$(dirname "${BASH_SOURCE[0]}")/run_sweep_probing.sh" "$@"
