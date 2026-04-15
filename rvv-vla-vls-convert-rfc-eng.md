# RFC: Intrinsic to Convert Between RVV Scalable Vector and Fixed-length Vector

## Summary

This RFC proposes a set of new intrinsics for the RISC-V Vector Extension (RVV):

1. **`__riscv_convert_vector`**: Convert between RVV scalable vector type (VLA, e.g. `vint32m1_t`) and fixed-length vector type (VLS / GNU vector, e.g. `int32x4_t`). It works across different VLEN values.
2. **A set of RISC-V specific fixed-length vector types** (named `v<type><width>x<nelem>_t`), marked by a new `rvv_vls_vector_size` type attribute. This attribute **automatically enables the RVV vector calling convention**.

This RFC does not talk about the details of any single compiler. The goal is to let GCC, Clang/LLVM, and any RISC-V toolchain that follows the psABI use the same user interface and semantics.

## 1. Background and Motivation

### 1.1 RVV vector types

- **Scalable vector type (VLA)**: For example `vint32m1_t`. Its length is not known at compile time. It is decided at run time by VLEN × LMUL.
- **Fixed-length vector type (VLS)**: Uses the `riscv_rvv_vector_bits` type attribute (already part of the psABI), together with the `-mrvv-vector-bits=<N>` command line option. It fixes the RVV vector size so it is known at compile time. For example:

  ```c
  typedef vint32m1_t fixed_int32m1_t
      __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)));
  ```

- **GNU vector type**: A fixed-size vector declared with the `vector_size` attribute (for example `int32x4_t`). It is not tied to the RVV ABI and can be used across many CPU architectures.

NOTE: The GNU vector type does not require RVV. But today, when the compile target has RVV, both GCC and Clang/LLVM use RVV instructions to implement it.

### 1.2 Why we need a convert intrinsic

In practice, users often need to convert between these two kinds of types:

- Take data that came in as a fixed-length vector and feed it to an RVV scalable intrinsic.
- Take the result of a scalable intrinsic and turn it into a fixed-length vector so it can be stored, passed around, or used with other SIMD code.

When the layouts match, the conversion is close to a bit-cast. In the best case it should not produce any extra instructions.

GNU vectors can already express most common operations through C operators (`+`, `-`, `*`, `/`, `&`, `|`, `<<`, ...). But many operations that RVV offers **cannot** be written using only C operators. For these, the user has to use RVV scalable intrinsics. For example:

- **Fixed-point arithmetic**: saturating add/sub (`vsadd` / `vssub`), rounding shift (`vssra` / `vssrl`), saturating multiply (`vsmul`), and so on.
- **Reduction**: cross-lane sum / max / min / and / or / xor reduction (`vredsum` / `vredmax` / ...). C operators have no matching form.
- **Mixed-width operations**: widening / narrowing add, sub, multiply (`vwadd` / `vwmul` / `vnsrl` / ...), widening multiply-accumulate, and so on. Source and destination element widths are different, so GNU vector operators cannot express them.
- **Masked conditional ops**, **permutation / slide / gather**, **segment load/store**, and **float-only instructions** (such as `vfrec7`, `vfrsqrt7`) also need intrinsics.

In other words, the convert intrinsic is a **bridge**. It lets users write code on top of the friendly fixed-length vector interface, and still switch back to scalable intrinsics to use RVV features, while keeping the semantics compatible. Without it, users are left with two bad choices: rewrite everything as scalable intrinsics, or go through memory to change the type.

One thing to note: when a fixed vector is converted to a larger scalable vector, only the **low elements** (the range matching the fixed vector) are guaranteed to be defined. So any later RVV operation must make sure its active result **does not observe source elements above that range**. For lane-wise arithmetic or logic operations where `vl` is clearly limited, and each active lane only depends on the matching active source lane, this is usually not a problem. But for operations that read data across lanes, or that may indirectly index into higher elements (for example some uses of gather / permutation / slide / reduction), the user must make sure the semantics do not touch the undefined high part.

### 1.3 Prior art: Arm SVE / NEON Bridge

Arm provides `svset_neonq` / `svget_neonq` / `svdup_neonq` to bridge between SVE and NEON. They use a "one typed function per element type" style. Arm can take this path easily because a NEON Q register is always 128 bits, and each SVE element type maps to exactly one NEON type. There are only about 12 pairs.

RVV has many more dimensions, so the same approach does not fit.

## 2. Candidate Designs

### 2.1 Option A: A single polymorphic intrinsic, parameterized by type

```c
<DST_TYPE> __riscv_convert_vector(<DST_TYPE>, <SRC_TYPE> val);
```

Example:

```c
int32x4_t  v1;
vint32m1_t sv = __riscv_convert_vector(vint32m1_t, v1);
int32x4_t  v2 = __riscv_convert_vector(int32x4_t, sv);
```

**Pros**

- Very small API surface. Only one name.
- Naturally supports any (src, dst) type pair.
- Adding a new element type or LMUL does not change the header interface.

**Cons**

- This is not a normal C function (C has no first-class types). The compiler must support a call form that takes a type as an "argument".
  - Note: This pattern is already used in both Clang and GCC. Examples are `__builtin_offsetof(type, member)`, `__builtin_va_arg(ap, type)`, and `__builtin_convertvector(expr, type)`. Both compilers already have the parser / frontend support, so this is not a brand new problem.
- Error message design needs more care.

### 2.2 Option B: One function per type pair, with type suffix

```c
<DST_TYPE> __riscv_convert_vector_<SRC>_<DST>(<SRC> val);
```

Example:

```c
vint32m1_t sv = __riscv_convert_vector_i32x4_i32m1(v1);
int32x4_t  v2 = __riscv_convert_vector_i32m1_i32x4(sv);
```

**Pros**

- It is a standard C function. Type checking works through the function signature.
- The name is clear. It is easy for tools and FFI bindings to look up.

**Cons**

- The number of names explodes (~100+).
- Adding a new element type or LMUL means adding more to the header.
- Supporting cross-VLEN portability makes the number even bigger (see §3.2).

## 3. RVV-specific Challenges

### 3.1 Many dimensions

| Dimension | Variations |
|-----------|------------|
| Element type | i8/i16/i32/i64, u8/u16/u32/u64, f16/f32/f64, bf16 — about 12 |
| LMUL | mf8/mf4/mf2/m1/m2/m4/m8 — 7 (some are not legal, e.g. i64mf8) |
| VLEN setting | `-mrvv-vector-bits=` can be zvl / 32 / 64 / 128 / ... |

If we use Option B, just (elt × LMUL) alone gives about 60~70 scalable types. With both directions that is ~140 symbols.

> Note: Within the same translation unit, `__riscv_v_fixed_vlen` is fixed. So we do not need to enumerate "all VLEN × all LMUL". We only need (elt, LMUL).

### 3.2 Portability pushes Option B even higher

Only allowing conversion when "fixed size equals scalable size" breaks portability:

- Under zvl128b: `int32x4_t` (128 bit) ↔ `vint32m1_t` (128 bit), sizes are equal.
- Under zvl256b: `int32x4_t` (128 bit) ↔ `vint32m1_t` (256 bit), sizes are not equal. If we only allow equal sizes, the user would have to rewrite the code to `vint32mf2_t` to get 4 i32 elements.

In practice, users expect **`int32x4_t ↔ vint32m1_t` to work for all VLEN values**. To support this, the fixed side must be able to map to "every LMUL that can hold it" (`vint32mf2_t`, `vint32m1_t`, `vint32m2_t`, `vint32m4_t`, `vint32m8_t`). The number of combinations in Option B goes from O(elt × LMUL) to O(elt × LMUL × fixed_width_levels). Listing them one by one is not realistic.

This ability is very important for real users. It lets the same code run on machines with different VLEN values without any changes. Users do not need to write a different kernel version for each VLEN.

### 3.3 Portability across compile modes

Today, for fixed-length vectors, we have `riscv_rvv_vector_bits`-based types and GNU vector types. But `riscv_rvv_vector_bits`-based types can only be used in `-mrvv-vector-bits=zvl` mode, and their size always equals VLEN. GNU vector types work in all modes, but their size is fixed (for example `int32x4_t` is always 128 bit). They cannot map to scalable types at different VLEN values.

So in this proposal, we suggest that when a fixed-length vector is smaller than the scalable vector, the mapping is defined as "write to / read from the low part of the scalable vector". This gives cross-VLEN portability and lets the same code run correctly on all machines with VLEN ≥ 128.

The user only needs to write the code for the VLEN = 128 case. For example, if the user uses `int32x4_t`, the conversion semantics between it and `vint32m1_t` are the same on every machine with VLEN ≥ 128. The user does not need to write a different kernel per VLEN.

We also support the case where VLEN < 128, but the compiler will emit a compatibility warning. It tells the user that the cross-VLEN ABI is not the expected 128, and that all compilation units must stay consistent.

## 4. Design Decisions

The decisions below are the final choices adopted by this RFC. They will be referenced in later implementation specs.

### 4.1 Adopt Option A

**Adopt Option A (polymorphic, type-as-argument) as the only user-facing interface**:

```c
DST_TYPE __riscv_convert_vector(DST_TYPE, SRC_TYPE val);
```

Reasons:

- Option B suffers from the combinatorial explosion in §3.2 and the cross-mode portability problem in §3.3. It cannot cleanly express "the same fixed type maps to different LMUL values".
- Option A only looks at the size and element type of the two types. It does not depend on the compile mode or VLEN setting.

We do not provide the type-suffix style of Option B as a standard interface. A toolchain may wrap one if needed, but that is outside this RFC.

### 4.2 Same element type only

`__riscv_convert_vector` **only allows src and dst with the same element type**.

- Cross element type bit-cast is not allowed (for example `vint32m1_t ↔ vfloat32m1_t`, `vuint32m1_t ↔ vint32m1_t`).
- Pure sign / float ↔ int reinterpret should use the existing `__riscv_vreinterpret_*` intrinsics.
- This limit keeps the intrinsic focused on "container type conversion". It does not mix in the reinterpret topic.
- This also draws a clear line against the built-in `__builtin_convertvector`. `__builtin_convertvector` allows cross element type conversion (but does not guarantee the scalable ↔ fixed low-part insert/extract semantics).

### 4.3 Semantics when lengths differ

- **fixed → scalable**: The fixed value is written to the **low part** of the scalable vector. The **high part is undef** (not guaranteed to be zero, and earlier content is not kept).
  - Why undef and not zero: to avoid generating extra instructions. Users who need zero or merge can do it themselves using splat or existing insert intrinsics.
  - Semantic limit: Only the low part, the elements that match the fixed vector, are guaranteed to be defined. If a later RVV operation has active results that might read or depend on higher source elements, the result is undefined. The user must make sure the operation only observes the valid low range.
- **scalable → fixed**: Read from the **low part** of the scalable vector. The high part is discarded.
- **Size rule**: `fixed_bits ≤ known_min_bits_of_scalable`. If it is larger, it is a **compile-time error**.
  - `known_min_bits_of_scalable` = `LMUL × zvl*b` using the minimum VLEN setting (not the run-time VLEN).

For example, when `vl` is limited to the number of fixed lanes, and each result lane only depends on the matching input lane (like `vadd` / `vand` / `vsadd`), the operation is usually safe. But if an active result lane might observe a source lane beyond the fixed range, through indexing, slide, cross-lane permutation, or a reduction flow, we cannot assume it is safe just because `vl` is small.

### 4.4 Supported fixed-length types

The fixed side of `__riscv_convert_vector` accepts these three kinds of types, all with the same semantics:

1. VLS type declared with the `riscv_rvv_vector_bits` attribute (only usable in `-mrvv-vector-bits=zvl` mode).
2. Fixed vector type declared with the GNU `vector_size` attribute (usable in any mode).
3. The new `v<type><width>x<nelem>_t` type defined in §5 of this RFC (usable in any mode).

### 4.5 Relationship with LMUL trunc / ext intrinsics

`__riscv_convert_vector` **forbids** scalable ↔ scalable conversion. That is, src and dst cannot both be scalable vector types. LMUL changes between scalable types must use the existing `__riscv_vlmul_trunc_*` / `__riscv_vlmul_ext_*` intrinsics.

Reason: If we allowed scalable ↔ scalable, the size rule (`fixed_bits ≤ known_min_bits_of_scalable` and the undef-high-part rule during LMUL extension) would partly overlap but not fully match the semantics of `__riscv_vlmul_trunc_*` / `__riscv_vlmul_ext_*`. Error and warning messages would find it hard to point to the right intrinsic. That would hurt usability. With an explicit ban, each intrinsic has a clear job:

- `__riscv_convert_vector`: **scalable ↔ fixed** in one direction (both ways allowed).
- `__riscv_vlmul_trunc_*` / `__riscv_vlmul_ext_*`: LMUL change **scalable ↔ scalable**.
- `__riscv_vreinterpret_*`: Element type / sign change with the same width.

### 4.6 Naming

Use the `__riscv_*` prefix, matching existing RVV intrinsics.

### 4.7 Why a new intrinsic instead of reusing `__builtin_convertvector`

`__builtin_convertvector(expr, type)` is an existing builtin shared by Clang and GCC. Its semantics are close to "vector conversion with type as an argument". This RFC still chooses to define a new `__riscv_convert_vector` instead of extending `__builtin_convertvector`, for three reasons:

1. **Matches the RISC-V naming convention**: RVV intrinsics all use the `__riscv_*` prefix (for example `__riscv_vadd_vv_i32m1`, `__riscv_vreinterpret_*`, `__riscv_vlmul_trunc_*`). Users can tell from the name that this is an RVV-only interface. It matches existing toolchain, docs, and search habits.
2. **Avoids polluting the global / cross-architecture namespace**: `__builtin_convertvector` is a cross-architecture builtin. Adding RVV-specific semantics to it (like the low-part insert/extract for scalable ↔ fixed, or the `known_min` size rule) would make behavior hard to predict for users on other architectures. Putting RVV-specific semantics under `__riscv_*` keeps the general builtin clean.
3. **Specialized error messages**: A dedicated intrinsic can give precise, RVV-specific diagnostics for cases like "fixed size exceeds the `known_min` of `LMUL × zvl*b`", "element type not compatible with target LMUL", or "trying to use a VLS type under `-mrvv-vector-bits=scalable`". A general builtin must stay neutral across architectures, so it cannot easily point at RVV-specific fixes.

### 4.8 C++ interface: `__riscv::rvv::convert`

In C++ mode, besides the macro / builtin form of `__riscv_convert_vector`, this RFC also provides a template function form:

```cpp
namespace __riscv {
namespace rvv {

template <typename DstType, typename SrcType>
inline DstType convert(const SrcType &src) {
    return __riscv_convert_vector(DstType, src);
}

} // namespace rvv
} // namespace __riscv
```

Usage:

```cpp
vint32m1_t sv = __riscv::rvv::convert<vint32m1_t>(v);
int32x4_t  v2 = __riscv::rvv::convert<int32x4_t>(sv);
```

The caller can omit `SrcType` (deduced from the argument). When needed, the full form `convert<DstType, SrcType>(src)` also works.

**Why this namespace**:

- The C++ standard says "identifiers starting with two underscores are reserved for the implementation in any scope" (and a namespace name is an identifier). So `__riscv` is allowed by the standard, and it clearly says "provided by the toolchain".
- It matches the `__riscv_*` prefix of C intrinsics. The naming system is unified.
- It leaves room for sub-namespaces (`rvv`, and maybe future `__riscv::cmo`, `__riscv::zicbom`, and so on) for future growth.

**Implementation cost**: This template wrapper just forwards to `__riscv_convert_vector`. All type and size checks come from the builtin itself, so there is zero extra compiler cost. The only requirement is that the builtin can be parsed in a dependent context (the template body). This is already the case for existing type-as-argument builtins like `__builtin_convertvector`. Both Clang and GCC already support it.

## 5. New: RISC-V fixed vector types and calling convention

To let users **enjoy the RVV vector calling convention without porting effort**, this RFC also defines a set of RISC-V specific fixed-length vector types. When any of these types appears in a function signature, it automatically enables the vector calling convention.

### 5.1 Names and coverage

- **Naming rule**: `v<type><width>x<nelem>_t`
  - Examples: `vint32x4_t`, `vfloat64x2_t`, `vbfloat16x8_t`, `vuint8x16_t`.
- **Element types**: `bf16`, `fp16`, `fp32`, `fp64`, `[u]int{8,16,32,64}`.
- **Minimum number of elements**: **at least 2** (no `x1`, to avoid confusion with scalar).
- **Maximum total width**: up to `ABI_VLEN × LMUL = 128 × 8 = 1024 bit`, covering all widths matching m1 / m2 / m4 / m8 under ABI_VLEN = 128.

This RFC also defines a dedicated fixed-length predicate / mask type family, `vboolx<nelem>_t`, for this group. Compare operators return `vboolx<nelem>_t`, and `?:` uses that mask type for lane-wise select (see §5.2.1).

### 5.2 Type attribute

Each type carries the `rvv_vls_vector_size` attribute. It takes two parameters: the first is the size (in bytes), the second is an optional `ABI_VLEN`:

```
__attribute__((rvv_vls_vector_size(<SIZE>[, <ABI_VLEN>])));
```

```c
typedef int vint32x4_t
    __attribute__((rvv_vls_vector_size(16, 128)));
typedef int vint32x8_t
    __attribute__((rvv_vls_vector_size(32)));
```

The first parameter is the size of the fixed vector (in bytes). The second parameter is the `ABI_VLEN` that this type uses to trigger the vector calling convention. If the second parameter is not given, the default is `ABI_VLEN = 128`.

#### 5.2.1 Why a new `rvv_vls_vector_size` instead of reusing GNU `vector_size`

The fixed-length data vector types in this RFC follow GNU-vector-like lane-wise compare semantics, but the compare result type is a dedicated fixed-length mask type: `vboolx<nelem>_t`. Each lane maps to one bit in the mask. Bit value 1 means true, bit value 0 means false. For the conditional operator `?:`, this RFC defines that when the condition is `vboolx<nelem>_t`, `cond ? lhs : rhs` does a lane-wise select. So C and C++ can use the same form.

In other words, this RFC does not claim "every operator is fully equal to GNU vector". Instead:

- **compare**: still works lane by lane, but returns `vboolx<nelem>_t` rather than a same-width integer vector.
- **`?:`**: uses `vboolx<nelem>_t` as the condition type and performs lane-wise select in both C and C++.

Since the data-vector operators still mostly follow GNU vector, reusing GNU `vector_size` looks like the easiest choice. But this RFC still chooses a separate `rvv_vls_vector_size` attribute, for these reasons:

- **Automatically enables the vector calling convention**: `rvv_vls_vector_size` is one attribute that carries three pieces of info: "size", "ABI_VLEN", and "use RVV vector cc". If we reused `vector_size`, we would need to add another marker attribute (like `rvv_vls_vector_cc(ABI_VLEN)`) to enable cc. That makes declarations more verbose and easier to forget.
- **ABI_VLEN is an RVV-only concept**: GNU `vector_size` is neutral across architectures. Putting an RVV-only parameter like `ABI_VLEN` into it is not a good fit. A separate attribute can carry this info cleanly.
- **Separates diagnostics and ABI checks from generic GNU vectors**: Types with `rvv_vls_vector_size` are clearly in RVV context. The compiler can give precise messages for RVV-specific cases like `LMUL × ABI_VLEN` consistency or cross-TU ABI_VLEN mismatch. It does not affect the behavior of existing GNU `vector_size`.
- **Room for RVV-specific semantics**: The fixed-length mask type, its ABI rule, and lane-wise select semantics are RVV-specific. Keeping them under a dedicated attribute family avoids changing the meaning of generic GNU `vector_size`.

The specific limits for `?:` are:

- The condition must be `vboolx<nelem>_t`, usually produced by a compare on `v<type><width>x<nelem>_t`.
- `lhs` and `rhs` must be the same `v<type><width>x<nelem>_t`.
- This RFC **does not define** the scalar broadcast form `cond ? scalar : vec` or `cond ? vec : scalar`.

This RFC defines the layout and ABI of `vboolx<nelem>_t` as follows:

- **Layout**: bitmask form. Lane `i` maps to bit `i`.
- **Size**: `sizeof(vboolx<nelem>_t) = max(nelem / 8, 1)` bytes.
- **Alignment**: `alignof(vboolx<nelem>_t) = 1`.
- **ABI mapping**: pass-by-value and return follow the same psABI rule as the matching scalable bool vector type `vbool<M>_t`. The exact `M` is the one selected by the element-width / lane-count relation of the producing or consuming operation.

The operator set of `vboolx<nelem>_t` is intentionally small and mask-oriented:

- **Allowed unary operators**: `!`, `~`.
- **Allowed binary operators**: bitwise `&`, `|`, `^`.
- **Allowed comparison operators**: `==`, `!=`.
- **Allowed conditional form**: `cond ? lhs : rhs` where `cond` is `vboolx<nelem>_t` and `lhs` / `rhs` are the same `v<type><width>x<nelem>_t`.
- **Allowed assignment forms**: initialization, copy assignment, and the compound assignments `&=`, `|=`, `^=`.

The following operators are **not** defined for `vboolx<nelem>_t`:

- **Element access**: `[]`.
- **Short-circuit logical operators**: `&&`, `||`.
- **Arithmetic operators**: unary `+` / `-`, binary `+`, `-`, `*`, `/`, `%`.
- **Shift operators**: `<<`, `>>`, `<<=`, `>>=`.
- **Ordering comparisons**: `<`, `<=`, `>`, `>=`.
- **Increment / decrement**: `++`, `--`.
- **Dereference and member access**: `*`, `->`, `.`.

Reason: `vboolx<nelem>_t` is a packed predicate container, not a data vector. The allowed operators are only those that map cleanly to lane-wise mask algebra without implying scalar truthiness, element lvalues, or arithmetic interpretation.

### 5.3 Why we need this type layer: avoid the cost of the default ABI

The default RISC-V calling convention handles vector sizes like this:

- Smaller than `XLEN × 2` → passed through GPR.
- Larger than `XLEN × 2` → passed through memory.

Either way, if the callee wants to work on the data in vector registers, it must pay the cost of `GPR ↔ VectorReg` or `Memory ↔ VectorReg` moves. If we use the vector cc, the vector value is passed directly in vector registers, and this move cost is gone.

### 5.4 Calling convention trigger rule

- If the **argument list or return type of a function uses any type with `rvv_vls_vector_size`**, the function automatically uses the RVV vector calling convention (see [riscv-elf-psabi-doc PR #418](https://github.com/riscv-non-isa/riscv-elf-psabi-doc/pull/418)).
- **Users do not need to add a function attribute by hand**. The design goal is to lower porting cost. If every function needed a manual attribute to use vector cc, performance would be very bad because of the moves.
- The good thing about letting the type trigger cc: users only need to make the decision once, at the type declaration. All functions that use this type inherit the correct ABI. For example, a third-party library like highway only needs a `typedef` using this type.

### 5.5 Default ABI_VLEN and compatibility handling

- **Default `ABI_VLEN = 128`**, which matches most implementations at zvl128b and above.
- **For `zve32*` / `zve64*` environments with VLEN < 128**:
  - The compiler emits a **warning** and automatically lowers `ABI_VLEN` to the matching `zvl*b`.
  - Examples:
    - `rv64gc_zve32f`: uses `ABI_VLEN = 32`, with a warning saying the expected value was 128.
    - `rv64gc_zve32f_zvl128b`: uses `ABI_VLEN = 128`, **no** warning.
  - This clearly tells the user that under sub-128 configurations, the ABI across files / libraries is not the default 128. The user must make sure all compilation units stay consistent.

### 5.6 How this connects with `__riscv_convert_vector`

`v<type><width>x<nelem>_t` is a valid fixed-side src/dst in `__riscv_convert_vector`. It is **treated the same** as `riscv_rvv_vector_bits` VLS types and plain GNU `vector_size` types:

```c
vint32x4_t v;
vint32m1_t sv = __riscv_convert_vector(vint32m1_t, v);   // OK, writes to low part
vint32x4_t v2 = __riscv_convert_vector(vint32x4_t, sv);  // OK, reads low part
```

This RFC only defines `__riscv_convert_vector` for scalable ↔ fixed **data** vectors (integer / float / bf16). `vboolx<nelem>_t` is part of this RFC for compare / select and ABI purposes, but mask conversion intrinsics between `vboolx<nelem>_t` and scalable `vbool*` are still outside the scope of this proposal.

## 6. Usage Examples

### 6.1 A kernel that is portable across VLEN

```c
// Do RVV addition on 4 i32 values. The same code is correct on every
// implementation with VLEN >= 128.
vint32x4_t add4(vint32x4_t a, vint32x4_t b) {
    vint32m1_t va = __riscv_convert_vector(vint32m1_t, a);
    vint32m1_t vb = __riscv_convert_vector(vint32m1_t, b);
    vint32m1_t vc = __riscv_vadd_vv_i32m1(va, vb, 4);
    return __riscv_convert_vector(vint32x4_t, vc);
}
```

- `vint32x4_t` carries `rvv_vls_vector_size(16, 128)`. So `add4` automatically uses vector cc. The arguments `a` / `b` and the return value all go through vector registers.
- Even on a machine with VLEN = 256 / 512, where `vint32m1_t` is larger than `vint32x4_t`, the low-part semantics of `__riscv_convert_vector` guarantees that only the low 128 bits hold valid data. And `vsetvl` with `vl = 4` ensures only 4 elements are processed.

### 6.2 Interoperating with GNU vector

```c
typedef int int32x4_t __attribute__((vector_size(16)));

// Take a GNU vector, convert it to a scalable vector inside the callee to
// run an RVV-specific operation, then write the result back to the memory
// pointed by out. The function itself does not return a scalable vector, so
// it does not enable vector cc (also, its signature has no type carrying
// rvv_vls_vector_size).
void sat_add_and_store(int32x4_t v, int32_t addend, int32_t *out) {
    vint32m1_t sv = __riscv_convert_vector(vint32m1_t, v);
    // Fixed-point saturating add: the GNU vector operator `+` only wraps
    // around. It cannot express signed saturation, so we must use the RVV
    // intrinsic.
    sv = __riscv_vsadd_vx_i32m1(sv, addend, 4);
    __riscv_vse32_v_i32m1(out, sv, 4);              // Store back using RVV.
}
```

A plain GNU vector does not enable vector cc (no `rvv_vls_vector_size` attribute). But it can still be used as the src/dst of `__riscv_convert_vector`. This lets the user, inside the callee, switch to a scalable intrinsic, do work that operators cannot express, and write out the result through an RVV store.

### 6.3 Compare / select (`vboolx<N>_t` + lane-wise `?:`)

```c
vint32x4_t clamp_min(vint32x4_t x, vint32x4_t lo) {
    return (x < lo) ? lo : x;
}
```

- The result type of `x < lo` is `vboolx4_t`. Bit `i` corresponds to lane `i`; 1 means true and 0 means false.
- `vboolx4_t` uses bitmask layout. Its size is `max(4 / 8, 1) = 1` byte and its alignment is 1.
- `?:` is a lane-wise select controlled by `vboolx4_t`.
- The pass-by-value / return ABI rule of `vboolx4_t` follows the matching scalable bool vector type `vbool<M>_t`.

## 7. Open Issues

- **Future extensions**: If new element types (such as fp8) are added later, both the naming rule `v<type><width>x<nelem>_t` and `__riscv_convert_vector` can extend to them naturally.

## 8. References

- RVV intrinsic spec: <https://github.com/riscv-non-isa/rvv-intrinsic-doc>
- RVV vector calling convention: <https://github.com/riscv-non-isa/riscv-elf-psabi-doc/pull/418>
- Arm SVE / NEON bridge intrinsics: <https://github.com/llvm/llvm-project/blob/main/clang/lib/Headers/arm_neon_sve_bridge.h>
