(import (heavy base))

(define-library (geomalg base)
  (export
    k-blade
    sum
    outprod
    inprod
    gprod
    rev
    inv
    basis-vector-type)
  (import (heavy base)
          (heavy mlir))
  (begin
    (load-plugin "libGeomalg.so")
    (define basis-vector-type-impl
      (load-builtin "geomalg_basis_vector_type"))

    (define-syntax basis-vector-type
      (syntax-rules (tag: norm-squared:)
        ((_ (tag: Tag) ; Tags are integer powers of two.
            (norm-squared: NormSquared))
         (basis-vector-type-impl
           Tag NormSquared))))

    ; Go full 5-d Conformal Geometric Algebra since
    ; everything we want is a subalgebra of that.
    (define e0 (basis-vector-type ; Scalars
                 (tag: 0)
                 (norm-squared: 1)))
    (define e1 (basis-vector-type
                 (tag: 1)
                 (norm-squared: 1)))
    (define e2 (basis-vector-type
                 (tag: 2)
                 (norm-squared: 1)))
    (define e3 (basis-vector-type
                 (tag: 4)
                 (norm-squared: 1)))
    (define ni (basis-vector-type
                 (tag: 8)
                 (norm-squared: 0)))
    (define no (basis-vector-type
                 (tag: 16)
                 (norm-squared: 0)))



    ))

