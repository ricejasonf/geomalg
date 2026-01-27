#include <heavy/Context.h>
#include <heavy/MlirHelper.h>
#include <heavy/Value.h>
#include <geomalg/GeomalgDialect.h>

namespace {
// This version of quicksort assumes unique entries.
template <typename Itr, typename SwapFn, typename LessThanFn>
void quicksort(Itr first, Itr last, SwapFn Swap, LessThanFn LessThan) {
  if (first >= last)
    return;

  // Calculate the pivot offset.
  ptrdiff_t pivot_distance = std::distance(first, last) / 2;

  Swap(*first, *(first + pivot_distance));

  // Store the current placeholder for the left side.
  Itr pivot = first;
  for (Itr itr = first + 1; itr < last; ++itr) {
    if (!LessThan(*first, *itr)) {
      ++pivot;
      Swap(*pivot, *itr);
    }
  }
  Swap(*first, *pivot);
  quicksort(first, pivot, Swap, LessThan);
  quicksort(pivot + 1, last, Swap, LessThan);
}
}  // namespace

extern "C" {
void geomalg_basis_vector_type(heavy::Context& C, heavy::ValueRefs Args) {
  if (Args.size() != 2)
    return C.RaiseError("invalid arity");
  if (!isa<heavy::Int>(Args[0]))
    return C.RaiseError("expecting integer: {}", Args[0]);

  uint32_t Tag = static_cast<uint32_t>(cast<heavy::Int>(Args[0]));

  // A basis vector must be a power of 2 that is
  // not highest representable power of 2.
  // This also means it is does not contain a wedge
  // product of two nontrivial vectors.
  if (std::popcount(Tag) != 1)
    return C.RaiseError("expecting power of two: {}", Args[0]);

  mlir::MLIRContext* MLIRContext = heavy::mlir_helper::getCurrentContext(C);
  auto BladeType = geomalg::BladeType::get(MLIRContext, Tag);

  C.Cont(C.CreateAny(BladeType));
}

// Construct a Blade type from a product of basis vectors
// (which also happen to be given as BladeTypes).
void geomalg_blade_type(heavy::Context& C, heavy::ValueRefs Args) {
  // BladeTypes will consist only of basis elements.
  llvm::SmallVector<geomalg::BladeType, 8> BladeTypes;
  for (heavy::Value Arg : Args) {
      // Push the positive version of the tag.
      if (auto BladeType = heavy::any_cast<geomalg::BladeType>(Arg);
          BladeType && BladeType.isBasisVector())
        BladeTypes.push_back(BladeType.getCanonicalType());
      else
        return C.RaiseError(
            "expecting basis element type (ie grade < 2): {}", Arg);
  }

  if (BladeTypes.empty())
    return C.RaiseError("expecting at least one basis element type");

  // Manually sort by canonical tag (ie without regard to sign bit.)
  // For each swap, we change the sign which may be incorrect if
  // elements are not unique, but we check that after sorting.
  uint32_t SignTag = 0;
  auto Swap = [&SignTag](geomalg::BladeType& A, geomalg::BladeType& B) {
      std::swap(A, B);
      SignTag ^= geomalg::BladeType::tag_sign_xor_mask;
    };
  auto LessThan = [](auto& A, auto& B) {
      return A.getCanonicalTag() < B.getCanonicalTag();
    };
  quicksort(BladeTypes.begin(), BladeTypes.end(), Swap, LessThan);

  // If we have more than one of any basis
  // element then the whole thing becomes zero.
  size_t OrigSize = BladeTypes.size();
  llvm::unique(BladeTypes);
  if (OrigSize != BladeTypes.size())
    return C.Cont(C.CreateAny(mlir::Type(geomalg::ZeroType())));

  uint32_t Tag = 0;
  for (geomalg::BladeType BladeType : BladeTypes)
    Tag &= BladeType.getTag();
  
  // Incorporate the sign bit.
  Tag |= SignTag;

  mlir::MLIRContext* MLIRContext = heavy::mlir_helper::getCurrentContext(C);
  mlir::Type BT = geomalg::BladeType::get(MLIRContext, Tag);
  return C.Cont(C.CreateAny(BT));
}

void geomalg_multivector_type(heavy::Context& C, heavy::ValueRefs Args) {
  // Each argument is a possibly improper list of blade types.
  // types used to construct blades.
  llvm::SmallVector<geomalg::BladeType, 8> BladeTypes;
  for (heavy::Value List : Args) {
    for (heavy::Value BVArg : List) {
      // Push the positive version of the tag.
      if (auto BladeType = heavy::any_cast<geomalg::BladeType>(BVArg))
        BladeTypes.push_back(BladeType.getCanonicalType());
      else
        return C.RaiseError("expecting blade type: {}", BVArg);
    }
  }

  // An empty Multivector is like the False sum type.
  // It exists as a contradiction.
  if (BladeTypes.empty())
    return C.RaiseError("multivector type must be nonempty");

  // Sort by canonical tag and sign.
  llvm::sort(BladeTypes, [](auto& A, auto& B) {
      return (A.getCanonicalTag() < B.getCanonicalTag()) ||
             (A.isNonnegative() && !B.isNonnegative());
    });

  // Unique by pointer-like equality. (mlir::Types are uniqued)
  llvm::unique(BladeTypes);

  mlir::MLIRContext* MLIRContext = heavy::mlir_helper::getCurrentContext(C);
  mlir::Type MT = geomalg::MultivectorType::get(MLIRContext, BladeTypes);
  C.Cont(C.CreateAny(MT));
}
}
