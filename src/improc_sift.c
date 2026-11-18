// Compiles core/sift.c for the Arduino build (see improc.h).
// CMake builds this file with -ffp-contract=off; Arduino cannot set per-file
// flags, so the pragma below does the same (see improc_convert_f32.c).
#include "improc_config.h"

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

#include "../core/sift.c"
