#include <heavy/Context.h>
#include <heavy/Value.h>
#include <geomalg/GeomalgDialect.h>

extern "C" {
void geomalg_basis_vector_type(heavy::Context& C, heavy::ValueRefs Args) {
  heavy::write(llvm::errs(), Args[0]);

  if (Args.size() != 2)
    return C.RaiseError("invalid arity");
  if (!isa<heavy::Int>(Args[0]))
    return C.RaiseError("expecting integer: {}", Args[0]);
  if (!isa<heavy::Int>(Args[1]))
    return C.RaiseError("expecting integer: {}", Args[1]);

  int Tag = cast<heavy::Int>(Args[0]);
  int NormSquared = cast<heavy::Int>(Args[1]);

  // A basis vector must be a power of 2 that is
  // not highest representable power of 2.
  // This also means it is not a wedge product.)
  if (std::popcount(Tag) != 1)
    return C.RaiseError("expecting power of two: {}", Tag);

  mlir::MLIRContext* MLIRContext = getCurrentContext(C);
  auto BladeType = geomalg::BladeType::get(MLIRContext, Tag, NormSquared);

  C.Cont(C.CreateAny(BladeType));
}
}
