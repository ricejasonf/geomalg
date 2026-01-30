// Geomalg.td - MLIR Dialect Definitions for Geomalg
// Copyright Jason Rice 2025
//
#ifndef GEOMALG_DIALECT_TD
#define GEOMALG_DIALECT_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"

def Geomalg_Dialect : Dialect {
  let name = "geomalg";
  let useDefaultTypePrinterParser = 1;
}

def Geomalg_Metric : TypeDef<Geomalg_Dialect, "Metric", []> {
  let mnemonic = "metric";
  let description = [{
    Define a metric tensor by specifying the dot product for pairs of basis
    vectors defaulting to e_i . e_i = 1 and e_i . e_j = 0 for i != j.

    This forms a weighted graph with an adjacency matrix concpetually
    equivalent to the block diagonal matrix described in Eric Lengyel's
    "Geometric Algebra Illuminated". This means we can define more
    useful "basis vectors" opaquely as linear combinations of more
    fundamental orthonormal basis vectors. (e.g. ni and no)
  }];

  //let parameters = (ins ArrayRefParameter<"geomalg::MetricEntry">:$entries);
  let assemblyFormat = "`<` params `>`";
  let extraClassDeclaration = [{

  }];
}

def Geomalg_Zero : TypeDef<Geomalg_Dialect, "Zero", []> {
  let mnemonic = "zero";
  let description = [{
    A blade of scalar zero has the zero type (iff its statically known.)
  }];

  let extraClassDeclaration = [{
    uint32_t getCanonicalTag() const { return 0; }
    unsigned getGrade() const { return 0; }
    bool isBasisVector() const { return false; }
    static constexpr bool isZero() { return true; }
    static constexpr bool isNonnegative() { return true; }
  }];
}

def Geomalg_Blade : TypeDef<Geomalg_Dialect, "Blade", []> {
  let mnemonic = "blade";
  let description = [{
    A k-blade is a wedge product of standard basis vectors
    where k is the number of vectors in the product.

    The type is parameterized by a bitmap tag to indicate what elements are
    present in the wedge product by their power of two.
    e.g.  A bitmap value of 0 would represent a scalar
          where 7 (0111) would represent e_1^e_2^e_3.
          err...
          0111
          0*2^3 + 1*2^2 + 1*2^1 + 1*2^0 = 4 + 2 + 1 = 7
          Yes.

    Additionally, we use the high bit to indicate the sign which can
    denote the order of the elements in the wedge product
    (though not uniquely).
    e.g.  e_1^e_2^e_3 = - e_2^e_1^e_3

    With this, we establish the convention that product operands
    in sorted order have positive sign.
  }];

  let parameters = (ins "uint32_t":$tag);

  // Try to remember that the argument is a tag and not the grade.
  let assemblyFormat = "`<` $tag `>`";

  let extraClassDeclaration = [{
    static constexpr unsigned tag_bit_width = 32;
    static constexpr unsigned tag_sign_mask = 1 << (tag_bit_width - 1);

    uint32_t getCanonicalTag() const {  // Strip the sign bit.
      return ~tag_sign_mask & getTag();
    }

    unsigned getGrade() const {
      // Count all the bits except the sign bit.
      return std::popcount(getCanonicalTag());
    }

    // Basis vectors include 0-blades and 1-blades
    bool isBasisVector() const {
      // Include scalar as a basis vector.
      // Exclude negative blades.
      return getGrade() < 2 && isNonnegative();
    }

    geomalg::BladeType getCanonicalType() const {
      return get(getContext(), getCanonicalTag());
    }

    bool isNonnegative() const {
      return !static_cast<bool>(tag_sign_mask & getTag());
    }

    static constexpr bool isZero() { return false; }
  }];
}

class Geomalg_BladeOfGradeK<int K> :
  Type<
    CPred<
      "::llvm::cast<geomalg::BladeType>($_self).getGrade() == " # K>>;
                      
def Geomalg_Multivector : TypeDef<Geomalg_Dialect, "Multivector", []> {
  let mnemonic = "multivector";
  let description = [{
    A multivector is a sum of k-blades.
  }];

  let parameters = (ins ArrayRefParameter<"geomalg::BladeType">:$blades);

  let assemblyFormat = "`<` $blades `>`";
}

def Geomalg_Unknown : TypeDef<Geomalg_Dialect, "unknown", []> {
  let mnemonic = "unknown";
  let description = [{
    A blade or multivector whose type must be inferred.
  }];
}

def Geomalg_Value : AnyTypeOf<[Geomalg_Blade,
                               Geomalg_Multivector,
                               Geomalg_Unknown]>;

// Operations

class Geomalg_Op<string mnemonic, list<Trait> traits = []> :
    Op<Geomalg_Dialect, mnemonic, traits>;

def Geomalg_SumOp : Geomalg_Op<"sum", []> {
  let description = [{
    Take a sum of k-blades to produce a multivector.
  }];

  let arguments = (ins Variadic<Geomalg_Value>:$args);
  let results   = (outs Geomalg_Value:$result);
}

def Geomalg_OuterProductOp : Geomalg_Op<"outprod", []> {
  let description = [{
    Map two multivectors to a multivector by their
    exterior product (aka wedge product).
  }];

  let arguments = (ins Geomalg_Value:$arg1,
                       Geomalg_Value:$arg2);
  let results   = (outs Geomalg_Value:$result);
}

def Geomalg_InnerProductOp : Geomalg_Op<"inprod", []> {
  let description = [{
    Map two multivectors to a scalar by their
    interior product.
  }];

  let arguments = (ins Geomalg_Value:$arg1,
                       Geomalg_Value:$arg2);
  let results   = (outs Geomalg_BladeOfGradeK<0>:$result);
}

def Geomalg_GeometricProductOp : Geomalg_Op<"gprod", []> {
  let description = [{
    Map two multivectors to the sum of their interior
    and exterior products which produces a multivector.
  }];

  let arguments = (ins Geomalg_Value:$arg1,
                       Geomalg_Value:$arg2);
  let results   = (outs Geomalg_Value:$result);
}

def Geomalg_GeometricReverseOp : Geomalg_Op<"rev", []> {
  let description = [{
    Take the reverse of a k-blade that swaps vectors
    in the wedge product until the vectors are in reverse sorted
    order. The only effect of this operation may be a sign
    change depending on the grade of the k-blade.
  }];

  let arguments = (ins Geomalg_Blade:$arg);
  let results   = (outs Geomalg_Blade:$result);
}

def Geomalg_GeometricInvOp : Geomalg_Op<"inv", []> {
  let description = [{
    Take the multiplicative inverse. For a single k-blade,
    this is defined as applying the multiplicative inverse of
    the scalar value to the reverse of the k-blade.
  }];

  let arguments = (ins Geomalg_Value:$arg);
  let results   = (outs Geomalg_Value:$result);
}

#endif // GEOMALG_DIALECT_TD
