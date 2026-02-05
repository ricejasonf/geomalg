#ifndef GEOMALG_PASSES_H
#define GEOMALG_PASSES_H

#include <geomalg/Dialect.h>
#include <mlir/Pass/Pass.h>

// Generated stuff
namespace geomalg {
#define GEN_PASS_DECL
#include "geomalg/GeomalgPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "geomalg/GeomalgPasses.h.inc"
}  // namespace geomalg

#endif  // GEOMALG_PASSES_H
