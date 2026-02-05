#include <geomalg/Dialect.h>
#include <geomalg/Type.h>
#include <heavy/Context.h>
#include <heavy/MlirHelper.h>
#include <heavy/Value.h>

namespace geomalg {
// Implement utilities for construction and
// introspection of geomalg dialect types.

// Create a BladeType from a wedge product of nonnegative basis vectors
// which may yield the ZeroType. The input array will be sorted in place.
mlir::Type
createBladeType(llvm::MutableArrayRef<geomalg::BladeType> BladeTypes) {
  assert(!BladeTypes.empty());

  // Manually sort by canonical tag (ie without regard to sign bit.)
  // For each swap, we change the sign which may be incorrect if
  // elements are not unique, but we check that after sorting.
  uint32_t SignTag = 0;
  auto Swap = [&SignTag](geomalg::BladeType& A, geomalg::BladeType& B) {
      // We are expecting canonical basis vectors. (ie nonnegative)
      assert(A.isBasisVector() && B.isBasisVector());
      std::swap(A, B);
      SignTag ^= geomalg::BladeType::tag_sign_mask;
    };
  auto LessThanEqual = [](auto& A, auto& B) {
      return A.getTag() <= B.getTag();
    };

  for (unsigned I = 0; I < BladeTypes.size(); I++) {
    for (unsigned J = I + 1; J < BladeTypes.size(); J++) {
      if (!LessThanEqual(BladeTypes[I], BladeTypes[J]))
        Swap(BladeTypes[I], BladeTypes[J]);
    }
  }

  // If we have more than one of any basis
  // element then the whole thing becomes zero.
  size_t OrigSize = BladeTypes.size();
  llvm::unique(BladeTypes);
  if (OrigSize != BladeTypes.size())
    return mlir::Type(geomalg::ZeroType());

  uint32_t Tag = 0;
  for (geomalg::BladeType BladeType : BladeTypes)
    Tag |= BladeType.getTag();
  
  // Incorporate the sign bit.
  Tag |= SignTag;

  mlir::MLIRContext* MLIRContext = BladeTypes.front().getContext();
  return geomalg::BladeType::get(MLIRContext, Tag);
}

// Create a canonicalized type for a multivector.
// The result may be a BladeType for a single term.
mlir::Type
createMultivectorType(llvm::MutableArrayRef<geomalg::BladeType> BladeTypes) {
  if (BladeTypes.empty())
    return geomalg::ZeroType();

  // Transform all negative blades to positive.
  llvm::transform(BladeTypes, BladeTypes.begin(),
      [](auto BT) { return BT.getCanonicalType(); });

  // Sort by canonical tag and sign.
  llvm::sort(BladeTypes, [](auto& A, auto& B) {
      return (A.getCanonicalTag() < B.getCanonicalTag()) ||
             (A.isNonnegative() && !B.isNonnegative());
    });

  // Unique by pointer-like equality. (mlir::Types are uniqued)
  auto EndItr = llvm::unique(BladeTypes);
  BladeTypes = BladeTypes.take_front(
      std::distance(BladeTypes.begin(), EndItr));


  // Return a BladeType if we can.
  if (BladeTypes.size() == 1)
    return BladeTypes.front();

  mlir::MLIRContext* MLIRContext = BladeTypes.front().getContext();
  return geomalg::MultivectorType::get(MLIRContext, BladeTypes);
}

}  // namespace geomalg
