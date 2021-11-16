// RUN: circt-opt %s -lower-std-to-handshake -split-input-file | FileCheck %s

// -----

// Simple affine.for with an empty loop body.

func @empty_body () -> () {
  affine.for %i = 0 to 10 { }
  return
}

// CHECK: handshake.func @empty_body(%arg0: none, ...) -> none attributes {argNames = ["inCtrl"], resNames = ["outCtrl"]} {
// CHECK:   %0:4 = fork(%arg0) {control = true} : (none) -> (none, none, none, none)
// CHECK:   %1 = constant(%0#2) {value = 0 : index} : (none) -> index
// CHECK:   %2 = constant(%0#1) {value = 10 : index} : (none) -> index
// CHECK:   %3 = constant(%0#0) {value = 1 : index} : (none) -> index
// CHECK:   %4 = branch(%0#3) {control = true} : (none) -> none
// CHECK:   %5 = branch(%1) : (index) -> index
// CHECK:   %6 = branch(%2) : (index) -> index
// CHECK:   %7 = branch(%3) : (index) -> index
// CHECK:   %8 = mux(%12#2, %6, %32) : (index, index, index) -> index
// CHECK:   %9:2 = fork(%8) : (index) -> (index, index)
// CHECK:   %10 = mux(%12#1, %7, %31) : (index, index, index) -> index
// CHECK:   %11:2 = control_merge(%4, %33) {control = true} : (none, none) -> (none, index)
// CHECK:   %12:3 = fork(%11#1) : (index) -> (index, index, index)
// CHECK:   %13 = mux(%12#0, %5, %34) : (index, index, index) -> index
// CHECK:   %14:2 = fork(%13) : (index) -> (index, index)
// CHECK:   %15 = arith.cmpi slt, %14#1, %9#1 : index
// CHECK:   %16:4 = fork(%15) : (i1) -> (i1, i1, i1, i1)
// CHECK:   %trueResult, %falseResult = conditional_branch(%16#3, %9#0) : (i1, index) -> (index, index)
// CHECK:   sink(%falseResult) : (index) -> ()
// CHECK:   %trueResult_0, %falseResult_1 = conditional_branch(%16#2, %10) : (i1, index) -> (index, index)
// CHECK:   sink(%falseResult_1) : (index) -> ()
// CHECK:   %trueResult_2, %falseResult_3 = conditional_branch(%16#1, %11#0) {control = true} : (i1, none) -> (none, none)
// CHECK:   %trueResult_4, %falseResult_5 = conditional_branch(%16#0, %14#0) : (i1, index) -> (index, index)
// CHECK:   sink(%falseResult_5) : (index) -> ()
// CHECK:   %17 = merge(%trueResult_4) : (index) -> index
// CHECK:   %18 = merge(%trueResult_0) : (index) -> index
// CHECK:   %19 = merge(%trueResult) : (index) -> index
// CHECK:   %20:2 = control_merge(%trueResult_2) {control = true} : (none) -> (none, index)
// CHECK:   sink(%20#1) : (index) -> ()
// CHECK:   %21 = branch(%17) : (index) -> index
// CHECK:   %22 = branch(%18) : (index) -> index
// CHECK:   %23 = branch(%19) : (index) -> index
// CHECK:   %24 = branch(%20#0) {control = true} : (none) -> none
// CHECK:   %25 = merge(%21) : (index) -> index
// CHECK:   %26 = merge(%22) : (index) -> index
// CHECK:   %27:2 = fork(%26) : (index) -> (index, index)
// CHECK:   %28 = merge(%23) : (index) -> index
// CHECK:   %29:2 = control_merge(%24) {control = true} : (none) -> (none, index)
// CHECK:   sink(%29#1) : (index) -> ()
// CHECK:   %30 = arith.addi %25, %27#1 : index
// CHECK:   %31 = branch(%27#0) : (index) -> index
// CHECK:   %32 = branch(%28) : (index) -> index
// CHECK:   %33 = branch(%29#0) {control = true} : (none) -> none
// CHECK:   %34 = branch(%30) : (index) -> index
// CHECK:   %35:2 = control_merge(%falseResult_3) {control = true} : (none) -> (none, index)
// CHECK:   sink(%35#1) : (index) -> ()
// CHECK:   return %35#0 : none
// CHECK: }

// -----

// Simple load store pair in the loop body.

func @load_store () -> () {
  %A = memref.alloc() : memref<10xf32>
  affine.for %i = 0 to 10 {
    %0 = affine.load %A[%i] : memref<10xf32>
    affine.store %0, %A[%i] : memref<10xf32>
  }
  return
}

// CHECK: handshake.func @load_store(%arg0: none, ...) -> none attributes {argNames = ["inCtrl"], resNames = ["outCtrl"]} {
// CHECK:   %0:3 = memory(%28#0, %28#1, %addressResults) {id = 0 : i32, ld_count = 1 : i32, lsq = false, st_count = 1 : i32, type = memref<10xf32>} : (f32, index, index) -> (f32, none, none)
// CHECK:   %1:2 = fork(%0#2) {control = true} : (none) -> (none, none)
// CHECK:   %2:4 = fork(%arg0) {control = true} : (none) -> (none, none, none, none)
// CHECK:   %3 = constant(%2#2) {value = 0 : index} : (none) -> index
// CHECK:   %4 = constant(%2#1) {value = 10 : index} : (none) -> index
// CHECK:   %5 = constant(%2#0) {value = 1 : index} : (none) -> index
// CHECK:   %6 = branch(%2#3) {control = true} : (none) -> none
// CHECK:   %7 = branch(%3) : (index) -> index
// CHECK:   %8 = branch(%4) : (index) -> index
// CHECK:   %9 = branch(%5) : (index) -> index
// CHECK:   %10 = mux(%14#2, %8, %40) : (index, index, index) -> index
// CHECK:   %11:2 = fork(%10) : (index) -> (index, index)
// CHECK:   %12 = mux(%14#1, %9, %39) : (index, index, index) -> index
// CHECK:   %13:2 = control_merge(%6, %41) {control = true} : (none, none) -> (none, index)
// CHECK:   %14:3 = fork(%13#1) : (index) -> (index, index, index)
// CHECK:   %15 = mux(%14#0, %7, %42) : (index, index, index) -> index
// CHECK:   %16:2 = fork(%15) : (index) -> (index, index)
// CHECK:   %17 = arith.cmpi slt, %16#1, %11#1 : index
// CHECK:   %18:4 = fork(%17) : (i1) -> (i1, i1, i1, i1)
// CHECK:   %trueResult, %falseResult = conditional_branch(%18#3, %11#0) : (i1, index) -> (index, index)
// CHECK:   sink(%falseResult) : (index) -> ()
// CHECK:   %trueResult_0, %falseResult_1 = conditional_branch(%18#2, %12) : (i1, index) -> (index, index)
// CHECK:   sink(%falseResult_1) : (index) -> ()
// CHECK:   %trueResult_2, %falseResult_3 = conditional_branch(%18#1, %13#0) {control = true} : (i1, none) -> (none, none)
// CHECK:   %trueResult_4, %falseResult_5 = conditional_branch(%18#0, %16#0) : (i1, index) -> (index, index)
// CHECK:   sink(%falseResult_5) : (index) -> ()
// CHECK:   %19 = merge(%trueResult_4) : (index) -> index
// CHECK:   %20:3 = fork(%19) : (index) -> (index, index, index)
// CHECK:   %21 = merge(%trueResult_0) : (index) -> index
// CHECK:   %22 = merge(%trueResult) : (index) -> index
// CHECK:   %23:2 = control_merge(%trueResult_2) {control = true} : (none) -> (none, index)
// CHECK:   %24:3 = fork(%23#0) {control = true} : (none) -> (none, none, none)
// CHECK:   %25 = join(%24#2, %1#1, %0#1) {control = true} : (none, none, none) -> none
// CHECK:   sink(%23#1) : (index) -> ()
// CHECK:   %26, %addressResults = load(%20#2, %0#0, %24#1) : (index, f32, none) -> (f32, index)
// CHECK:   %27 = join(%24#0, %1#0) {control = true} : (none, none) -> none
// CHECK:   %28:2 = store(%26, %20#1, %27) : (f32, index, none) -> (f32, index)
// CHECK:   %29 = branch(%20#0) : (index) -> index
// CHECK:   %30 = branch(%21) : (index) -> index
// CHECK:   %31 = branch(%22) : (index) -> index
// CHECK:   %32 = branch(%25) {control = true} : (none) -> none
// CHECK:   %33 = merge(%29) : (index) -> index
// CHECK:   %34 = merge(%30) : (index) -> index
// CHECK:   %35:2 = fork(%34) : (index) -> (index, index)
// CHECK:   %36 = merge(%31) : (index) -> index
// CHECK:   %37:2 = control_merge(%32) {control = true} : (none) -> (none, index)
// CHECK:   sink(%37#1) : (index) -> ()
// CHECK:   %38 = arith.addi %33, %35#1 : index
// CHECK:   %39 = branch(%35#0) : (index) -> index
// CHECK:   %40 = branch(%36) : (index) -> index
// CHECK:   %41 = branch(%37#0) {control = true} : (none) -> none
// CHECK:   %42 = branch(%38) : (index) -> index
// CHECK:   %43:2 = control_merge(%falseResult_3) {control = true} : (none) -> (none, index)
// CHECK:   sink(%43#1) : (index) -> ()
// CHECK:   return %43#0 : none
// CHECK: }

// TODO: affine expr in loop bounds
// TODO: nested loops
// TODO: yield carries values
