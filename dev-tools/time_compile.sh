#!/usr/bin/env bash
# Compiler launcher that logs the wall time of every compile to
# $HEONGPU_COMPILE_LOG (default /tmp/heongpu_compile_times.log).
# Use: cmake -D CMAKE_CUDA_COMPILER_LAUNCHER=dev-tools/time_compile.sh ...
log="${HEONGPU_COMPILE_LOG:-/tmp/heongpu_compile_times.log}"
src=""
for a in "$@"; do case "$a" in *.cu|*.cpp|*.c) src="$a";; esac; done
start=$(date +%s.%N)
"$@"
rc=$?
end=$(date +%s.%N)
printf '%.1f %s\n' "$(echo "$end - $start" | bc)" "${src:-$1}" >> "$log"
exit $rc
