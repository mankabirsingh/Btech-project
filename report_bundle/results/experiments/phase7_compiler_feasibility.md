# Phase 7 Compiler & Whole-Program Analysis Feasibility

**Project:** Provenance-Aware Security Risk Assessment of Human-AI Hybrid Code  
**Institution:** Delhi Technological University  
**Date:** October 27, 2026  
**Artifact Path:** [`results/experiments/phase7_compiler_feasibility.md`](file:///Users/mankabir/Desktop/Btech%20project/results/experiments/phase7_compiler_feasibility.md)  
**Baseline Audit:** [`results/experiments/phase7_baseline_audit.md`](file:///Users/mankabir/Desktop/Btech%20project/results/experiments/phase7_baseline_audit.md)  

---

## 1. Executive Summary & Feasibility Objective

Phase 7 investigates whether **compiler-assisted whole-program analysis** using Clang and LLVM can be reproducibly integrated into vulnerability detection for modern C++ codebases like Envoy. The hypothesis is that extracting compiler-verified Interprocedural Control-Flow Graphs (ICFG), Value-Flow Graphs (VFG), and virtual-method dispatch tables will eliminate single-function false positives by proving whether boundary conditions are satisfied upstream.

In strict compliance with empirical integrity rules:
> *"Start with one manageable target in Envoy rather than attempting to compile every repository immediately. If compilation fails, record the exact error and missing dependency. Do not pretend to have obtained compiler-derived information from source that was never successfully compiled."*

This report details the exact compiler toolchain status, repository source availability, concrete compilation attempts on Envoy functions, and the boundary conditions governing compiler-assisted modeling.

---

## 2. Compiler Toolchain & System Environment Audit

We inspected the local compiler toolchains and build utilities:

| Tool / Utility | Path / Availability | Version / Status | Capability Assessment |
| :--- | :--- | :--- | :--- |
| **Apple Clang (`clang`)** | `/usr/bin/clang` | Apple clang version 21.0.0 (clang-2100.1.1.101, arm64) | Available: supports `-fsyntax-only`, `-Xclang -dump-tokens`, AST dumps. |
| **Apple Clang++ (`clang++`)**| `/usr/bin/clang++` | Apple clang version 21.0.0 (C++17/20 support) | Available: parses standard C++ syntax when headers exist. |
| **LLVM Config (`llvm-config`)**| Not found | Unavailable in system PATH | Cannot link against dynamic LLVM C++ libraries directly. |
| **Bazel Build Tool (`bazel`)** | Not found | Unavailable in system PATH | Envoy's official build system is not installed. |
| **CMake (`cmake`)** | Not found | Unavailable in system PATH | C/C++ project build automation is not installed. |
| **Compilation Database** | Not found (`compile_commands.json`) | None present in repository | Compiler flags, include search paths (`-I`), and macro defines are absent. |
| **Envoy Git Tree** | Not found | Isolated parquet storage only | Unmodified project source files and header trees are not locally cloned. |

---

## 3. Concrete Compilation Attempts on Envoy Functions

We conducted direct compilation experiments on representative Envoy functions extracted from the locked benchmark [`vulnerability_test_heldout_projects_clean.parquet`](file:///Users/mankabir/Desktop/Btech%20project/data/splits/vulnerability_test_heldout_projects_clean.parquet):

### Target 1: Production Implementation Function (`FilterManager::sendLocalReply`)
**Source Sample:**
```cpp
void FilterManager::sendLocalReply(
    Code code, absl::string_view body,
    const std::function<void(ResponseHeaderMap& headers)>& modify_headers,
    const absl::optional<Grpc::Status::GrpcStatus> grpc_status, absl::string_view details) {
  state_.filter_call_state_ |= FilterCallState::LocalReply;
  ActiveStreamFilterBase* filter = nullptr;
  ...
}
```
**Compilation Command:**
```bash
clang++ -fsyntax-only -std=c++17 /tmp/tmpbz9c14rx.cpp
```
**Compiler Output & Fatal Diagnostic:**
```
/tmp/tmpbz9c14rx.cpp:1:6: error: use of undeclared identifier 'FilterManager'
    1 | void FilterManager::sendLocalReply(
      |      ^
/tmp/tmpbz9c14rx.cpp:2:5: error: unknown type name 'Code'
    2 |     Code code, absl::string_view body,
      |     ^
/tmp/tmpbz9c14rx.cpp:2:16: error: use of undeclared identifier 'absl'
    2 |     Code code, absl::string_view body,
      |                ^
/tmp/tmpbz9c14rx.cpp:3:30: error: unknown type name 'ResponseHeaderMap'
    3 |     const std::function<void(ResponseHeaderMap& headers)>& modify_headers,
      |                              ^
4 errors generated.
```
**Diagnostic Analysis:**
1. The class `FilterManager` is declared in `source/common/http/filter_manager.h`, which is not present.
2. The type `Code` is an Envoy HTTP status enum declared in `envoy/http/codes.h`.
3. The namespace `absl` requires Google Abseil C++ headers (`absl/strings/string_view.h`, `absl/types/optional.h`).
4. Clang aborts parsing at line 1, generating zero AST nodes for the function body.

---

### Target 2: GoogleTest Unit Test Fixture (`HttpHealthCheckerImplTest`)
**Source Sample:**
```cpp
TEST_F(HttpHealthCheckerImplTest, GoAwayProbeInProgress) {
  initialize();
  EXPECT_CALL(*event_logger_, logEject(...));
  ...
}
```
**Compilation Command:**
```bash
clang++ -fsyntax-only /tmp/tmpophxjvk_.cpp
```
**Compiler Output & Fatal Diagnostic:**
```
/tmp/tmpophxjvk_.cpp:1:8: error: unknown type name 'HttpHealthCheckerImplTest'
    1 | TEST_F(HttpHealthCheckerImplTest, GoAwayProbeInProgress) {
      |        ^
/tmp/tmpophxjvk_.cpp:1:35: error: unknown type name 'GoAwayProbeInProgress'
    1 | TEST_F(HttpHealthCheckerImplTest, GoAwayProbeInProgress) {
      |                                   ^
/tmp/tmpophxjvk_.cpp:1:1: error: C++ requires a type specifier for all declarations
    1 | TEST_F(HttpHealthCheckerImplTest, GoAwayProbeInProgress) {
      | ^
3 errors generated.
```
**Diagnostic Analysis:**
`TEST_F` is a macro defined in `gtest/gtest.h`. Without the GoogleTest framework headers, Clang interprets `TEST_F` as an unknown function declaration without a return type.

---

## 4. Root Cause Analysis: The Function-Level Dataset Boundary

The inability to compile Envoy functions directly with Clang is not an implementation error; it is an intrinsic architectural boundary of modern software mining:
1. **Corpus Extraction Artifact:** Vulnerability benchmarks (BigVul, DiverseVul, Devign) extract functions as string literals stripped of their build context, `#include` directives, and compilation databases.
2. **Envoy Build Monolith:** Envoy cannot be compiled as loose translation units. It requires:
   - The Bazel build engine and hermetic toolchains.
   - Over 50 external third-party dependencies (Abseil, BoringSSL, Protobuf, gRPC, libevent, yaml-cpp).
   - Generated Protobuf/gRPC C++ headers produced during the build step.
3. **The Data Fabrication Prohibition:** Claiming to run full whole-program LLVM compilation, link-time inter-procedural alias analysis, or bitcode ICFG generation on raw function strings without their headers would constitute academic fabrication.

---

## 5. Feasible Program Analysis Architecture (Phase 3 & 4)

While full compiler bitcode generation is infeasible without repository checkouts, we can implement a **sound, reproducible program analysis pipeline** supported by the actual artifacts:
1. **Clang Lexical Analysis:** Apple Clang's front-end preprocessor lexer (`clang -Xclang -dump-tokens`) extracts verified token sequences, distinguishing keywords, identifiers, and literals.
2. **Syntax-Tolerant Intra-Commit Control-Flow Graphs (CFG):** Construct Basic Block graphs and control-flow edges from control structures (`if`, `else`, `while`, `for`, `switch`, `return`, `throw`, `goto`).
3. **Guard Dominance Slicing:** Verify whether an upstream branch condition dominates sensitive operations (buffer writes, socket sends, pointer dereferences).
4. **Transparent Relationship Labeling:** Explicitly categorize every relationship:
   - *Statically Inferred Relationship* (syntactically resolved intra-commit call edge).
   - *Unresolved Relationship* (callee whose body is in an external, unmodified file).
   - *Dynamic Interface Dispatch* (virtual call through an abstract interface pointer, e.g., `filter->decodeHeaders()`).
