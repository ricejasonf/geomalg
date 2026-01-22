(import (heavy base))

(define-library (geomalg base)
  (export
    geomalg-module-init
    define-func
    basis-vector-type
    multivector-type
    e0 e1 e2 e3 ni no ;; Conformal GA basis vectors (e0 is scalar)
    ; k-blade ;; TODO make literal op
    sum
    outprod
    inprod
    gprod
    rev
    inv)
  (import (heavy base)
          (heavy mlir))
  (begin
    (load-plugin "libGeomalg.so")
    (define basis-vector-type-impl
      (load-builtin "geomalg_basis_vector_type"))
    (define multivector-type-impl
      (load-builtin "geomalg_multivector_type"))

    (define current-module 0)
    ;; Initialize a module and set it as current module.
    (define (geomalg-module-init name)
      (let ((ModuleOp
              (create-op "builtin.module"
                         (loc: 0)
                         (operands:)
                         (attributes: ("sym_name": name))
                         (result-types:)
                         (region "body" () 0)))) ;; Just create the region.
        ; Set the current insertion point to the region body.
        (set-insertion-point (entry-block (get-region ModuleOp)))))

    ;; Just initialize a monolithic module.
    (geomalg-module-init "geomalg_the_module")

    ;; If any function parameter type is unknown
    ;; then the func is used as a template.
    (define !geomalg.unknown (type "!geomalg.unknown"))

    (define-syntax basis-vector-type
      (syntax-rules (tag: norm-squared:)
        ((basis-vector-type (tag: Tag) ; Tags are integer powers of two.
            (norm-squared: NormSquared))
         (basis-vector-type-impl
           Tag NormSquared))))

    ;; Each argument is a non-empty list of basis vectors.
    ;; The resulting type is canonicalized by
    ;;    - adjusting sign of each blade according to
    ;;      the input order of the basis vectors
    ;;    - combining like terms with the same sign
    ;;    - sorting blades by tag
    (define-syntax multivector-type
      (syntax-rules ()
        ((multivector-type (BasisVector1M BasisVectorNM ...) ...)
         (multi-vector-type-impl
           '(BasisVector1M BasisVectorNM ...) ...))))

    ;; Go full 5-d Conformal Geometric Algebra since
    ;; everything we want is a subalgebra of that.
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

    (define-syntax define-func
      (syntax-rules()
        ((define-func FuncName ((ArgName : ArgType) ...)
                      Body)
         (create-op "func.func"
                    (loc: (syntax-source-loc FuncName))
                    (operands:)
                    (attributes:
                      ("sym_name" (string-attr FuncName))
                      ("function_type"
                        (type-attr (%function-type
                                     #(ArgType ...)
                                     #()))))
                    (result-types:)
                    (region: "body" Body)))))

    (define (sum-impl Loc . VN)
      (create-op "geomalg.sum"
                 (loc: Loc)
                 (operands: VN)
                 (attributes:)
                 (result-types: !geomalg.unknown)))

    (define-syntax sum
      (syntax-rules ()
        (sum V1 VN ...)
        (sum-impl (syntax-source-loc V1) V1 VN ...)))

    ;; TODO It would be nice to make a syntax to generate these
    ;;      but local syntax is not yet supported in heavy-scheme.
    ;;      The impl functions prevent syntax garbage creation.

    (define (outprod-impl Loc V1 V2)
      (create-op "geomalg.outprod"
                 (loc: Loc)
                 (operands: V1 V2)
                 (attributes:)
                 (result-types: !geomalg.unknown)))

    (define-syntax outprod
      (syntax-rules ()
        (outprod V1 V2)
        (outprod-impl (syntax-source-loc V1) V1 V2)))

    (define (inprod-impl Loc V1 V2)
      (create-op "geomalg.inprod"
                 (loc: Loc)
                 (operands: V1 V2)
                 (attributes:)
                 (result-types: !geomalg.unknown)))

    (define-syntax inprod
      (syntax-rules ()
        (inprod V1 V2)
        (inprod-impl (syntax-source-loc V1) V1 V2)))

    (define (gprod-impl Loc V1 V2)
      (create-op "geomalg.gprod"
                 (loc: Loc)
                 (operands: V1 V2)
                 (attributes:)
                 (result-types: !geomalg.unknown)))

    (define-syntax gprod
      (syntax-rules ()
        (gprod V1 V2)
        (gprod-impl (syntax-source-loc V1) V1 V2)))

    (define (rev-impl Loc V)
      (create-op "geomalg.rev"
                 (loc: Loc)
                 (operands: V)
                 (attributes:)
                 (result-types: !geomalg.unknown)))

    (define-syntax rev
      (syntax-rules ()
        (rev V)
        (rev-impl (syntax-source-loc V) V)))

    (define (inv-impl Loc V)
      (create-op "geomalg.inv"
                 (loc: Loc)
                 (operands: V)
                 (attributes:)
                 (result-types: !geomalg.unknown)))

    (define-syntax inv
      (syntax-rules ()
        (inv V)
        (inv-impl (syntax-source-loc V) V)))
    )) ;; define-library
