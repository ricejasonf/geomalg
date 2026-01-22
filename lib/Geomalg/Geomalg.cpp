#include <heavy/Context.h>
#include <heavy/MlirHelper.h>
#include <heavy/Value.h>
#include <geomalg/GeomalgDialect.h>

extern "C" {
void geomalg_basis_vector_type(heavy::Context& C, heavy::ValueRefs Args) {
  if (Args.size() != 2)
    return C.RaiseError("invalid arity");
  if (!isa<heavy::Int>(Args[0]))
    return C.RaiseError("expecting integer: {}", Args[0]);
  if (!isa<heavy::Int>(Args[1]))
    return C.RaiseError("expecting integer: {}", Args[1]);

  uint32_t Tag = static_cast<uint32_t>(cast<heavy::Int>(Args[0]));
  uint32_t NormSquared = static_cast<uint32_t>(cast<heavy::Int>(Args[1]));

  // A basis vector must be a power of 2 that is
  // not highest representable power of 2.
  // This also means it is not a wedge product.)
  if (std::popcount(Tag) != 1)
    return C.RaiseError("expecting power of two: {}", Args[0]);

  mlir::MLIRContext* MLIRContext = heavy::mlir_helper::getCurrentContext(C);
  auto BladeType = geomalg::BladeType::get(MLIRContext, Tag, NormSquared);

  C.Cont(C.CreateAny(BladeType));
}

void geomalg_multivector_type(heavy::Context& C, heavy::ValueRefs Args) {
  // Each argument is a possibly improper list of basis vector
  // types used to construct blades.
  for (heavy::Value List : Args) {
    for (heavy::Value BVArg : List) {
      // TODO Create Blade type.
    }
  }

  C.Cont(C.CreateAny(ResultType));
}
}
