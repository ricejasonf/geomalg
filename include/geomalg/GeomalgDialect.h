#ifndef GEOMALG_DIALECT_H
#define GEOMALG_DIALECT_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace geomalg {
// Specify a weight (ie the DotProduct) for an edge in a weighted graph
// used to generate the metric tensor for the generalized inner product.
struct MetricEntry {
  uint32_t BladeTag1;
  uint32_t BladeTag2;
  int32_t DotProduct;

  // Ensure tags are in sorted order.
  MetricEntry(uint32_t Tag1, uint32_t Tag2, int32_t DP)
    : BladeTag1(Tag2 >= Tag1 ? Tag2 : Tag1),
      BladeTag2(Tag2 >= Tag1 ? Tag1 : Tag2),
      DotProduct(DP)
  { }
};
}

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
