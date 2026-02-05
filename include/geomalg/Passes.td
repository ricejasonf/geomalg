// GeomalgDialectPasses.td - MLIR Dialect Pass Definitions for Geomalg
//
// Copyright Jason Rice 2025
//
#ifndef GEOMALG_DIALECT_PASSES_TD
#define GEOMALG_DIALECT_PASSES_TD

include "mlir/Pass/PassBase.td"

//class GeomalgModulePass<string Name> : PassBase<Name, "geomalg::GeomalgModulePass">;

def TypeInferencePass : Pass<"geomalg-type-inference",
                             "mlir::func::FuncOp"> {
  let summary = "Substitute unknown types with inferred types where possible";
  let description = [{
    The results of operations and functions are initially constructed with the
    !geomalg.unknown type or some other more constrained placeholder type.
    We replace these with inferred types within this pass.
  }];

  let dependentDialects = ["geomalg::GeomalgDialect",
                           "mlir::func::FuncDialect"];
}

def ArgumentDeductionPass : Pass<"geomalg-argument-deduction",
                                 "mlir::ModuleOp"> {
  let summary = "Inline or instantiate functions with placeholder types";
  let description = [{
    For any call to a function with placeholder types, inline the function or
    instantiate a new function with the placeholder types for the arguments
    substituted.
  }];

  let dependentDialects = ["geomalg::GeomalgDialect",
                           "mlir::func::FuncDialect"];
}

#endif  // GEOMALG_DIALECT_PASSES_TD
