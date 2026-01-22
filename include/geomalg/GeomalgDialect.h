#ifndef GEOMALG_DIALECT_H
#define GEOMALG_DIALECT_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// #pragma clang diagnostic push
// #pragma clang diagnostic ignored "-Wunused-parameter"

#include "geomalg/GeomalgDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "geomalg/GeomalgTypes.h.inc"

#define GET_ATTRDEF_CLASSES
#include "geomalg/GeomalgAttrs.h.inc"

#define GET_OP_CLASSES
#include "geomalg/GeomalgOps.h.inc"

// #pragma clang diagnostic pop

#endif  // GEOMALG_DIALECT_H
