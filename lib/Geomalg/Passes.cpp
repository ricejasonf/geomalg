#include <geomalg/Dialect.h>
#include <geomalg/Passes.h>
#include <geomalg/Type.h>
#include <llvm/ADT/STLExtras.h>
#include <cassert>

// Generated stuff
namespace geomalg {
#define GEN_PASS_DEF_ARGUMENTDEDUCTIONPASS
#define GEN_PASS_DEF_TYPEINFERENCEPASS
#include "geomalg/GeomalgPasses.h.inc"
}

namespace {
using geomalg::isUnknown;
using geomalg::isZero;

struct TypeInferencePass
  : public geomalg::impl::TypeInferencePassBase<TypeInferencePass> {
  void runOnOperation() override;
};

struct ArgumentDeductionPass
  : public geomalg::impl::ArgumentDeductionPassBase<ArgumentDeductionPass> {
  void runOnOperation() override;
};

}  // namespace

void TypeInferencePass::runOnOperation() {
  mlir::func::FuncOp FuncOp = getOperation();
  // Assume Body has a single block.
  mlir::Block* Body = !FuncOp.getBody().empty()
    ? &FuncOp.getBody().front() : nullptr;

  mlir::func::ReturnOp ReturnOp;
  if (Body)
    ReturnOp = dyn_cast_if_present<mlir::func::ReturnOp>(Body->getTerminator());

  if (!ReturnOp || ReturnOp.getOperands().size() != 1 ||
      FuncOp.getResultTypes().size() != 1) {
    FuncOp.emitOpError("expecting single return type for function");
    return;
  }

  // At this point Body and ReturnOp are valid.

  // SumOp

  FuncOp.walk([](geomalg::SumOp SumOp) {
    if (!isUnknown(SumOp.getResult()))
      return mlir::WalkResult::advance();

    llvm::SmallVector<geomalg::BladeType, 8> BladeTypes;
    for (mlir::Value V : SumOp.getArgs()) {
      mlir::Type Type = V.getType();
      if (isUnknown(V))
        return mlir::WalkResult::advance();
      if (auto BT = dyn_cast<geomalg::BladeType>(Type))
        BladeTypes.push_back(BT);
      else if (auto MT = dyn_cast<geomalg::MultivectorType>(Type))
        llvm::append_range(BladeTypes, MT.getBlades());
      else
        assert(isZero(Type) &&
            "expecting a valid operand type to geomalg.sum");
    }
    
    mlir::Type NewType = createMultivectorType(BladeTypes);
    SumOp.getResult().setType(NewType);
    return mlir::WalkResult::advance();
  });

  // TODO infer results of inprod, outprod.

  // Finally infer the function return type by the operand
  // of the ReturnOp.
  mlir::Type OrigResultTy = FuncOp.getResultTypes().front();
  mlir::Type ReturnTy = ReturnOp.getOperands().front().getType();
  if (isUnknown(OrigResultTy) && !isUnknown(ReturnTy)) {
    // Replace the function type.
    mlir::FunctionType NewFT = mlir::FunctionType::get(
        FuncOp.getContext(), FuncOp.getArgumentTypes(), ReturnTy);
    FuncOp.setFunctionType(NewFT);
  }

}

void ArgumentDeductionPass::runOnOperation() {
  mlir::ModuleOp ModuleOp = getOperation();
  llvm::errs() << "\nTODO\n";
}
