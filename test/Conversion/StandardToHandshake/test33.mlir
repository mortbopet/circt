// NOTE: Assertions have been autogenerated by utils/update_mlir_test_checks.py
// RUN: circt-opt -lower-std-to-handshake %s | FileCheck %s
  func @test() {
// CHECK:       module {

// CHECK-LABEL:   handshake.func @test(
// CHECK-SAME:                         %[[VAL_0:.*]]: none, ...) -> none attributes {argNames = ["inCtrl"], resNames = ["outCtrl"]} {
// CHECK:           %[[VAL_1:.*]]:5 = memory(%[[VAL_2:.*]]#0, %[[VAL_2]]#1, %[[VAL_3:.*]], %[[VAL_4:.*]]) {id = 0 : i32, ld_count = 2 : i32, lsq = false, st_count = 1 : i32, type = memref<10xf32>} : (f32, index, index, index) -> (f32, f32, none, none, none)
// CHECK:           %[[VAL_5:.*]]:2 = fork(%[[VAL_1]]#4) {control = true} : (none) -> (none, none)
// CHECK:           %[[VAL_6:.*]]:2 = fork(%[[VAL_1]]#3) {control = true} : (none) -> (none, none)
// CHECK:           %[[VAL_7:.*]]:3 = fork(%[[VAL_0]]) {control = true} : (none) -> (none, none, none)
// CHECK:           %[[VAL_8:.*]] = constant(%[[VAL_7]]#1) {value = 0 : index} : (none) -> index
// CHECK:           %[[VAL_9:.*]] = constant(%[[VAL_7]]#0) {value = 10 : index} : (none) -> index
// CHECK:           %[[VAL_10:.*]] = branch(%[[VAL_7]]#2) {control = true} : (none) -> none
// CHECK:           %[[VAL_11:.*]] = branch(%[[VAL_8]]) : (index) -> index
// CHECK:           %[[VAL_12:.*]] = branch(%[[VAL_9]]) : (index) -> index
// CHECK:           %[[VAL_13:.*]] = mux(%[[VAL_14:.*]]#1, %[[VAL_15:.*]], %[[VAL_12]]) : (index, index, index) -> index
// CHECK:           %[[VAL_16:.*]]:2 = fork(%[[VAL_13]]) : (index) -> (index, index)
// CHECK:           %[[VAL_17:.*]]:2 = control_merge(%[[VAL_18:.*]], %[[VAL_10]]) {control = true} : (none, none) -> (none, index)
// CHECK:           %[[VAL_14]]:2 = fork(%[[VAL_17]]#1) : (index) -> (index, index)
// CHECK:           %[[VAL_19:.*]] = mux(%[[VAL_14]]#0, %[[VAL_20:.*]], %[[VAL_11]]) : (index, index, index) -> index
// CHECK:           %[[VAL_21:.*]]:2 = fork(%[[VAL_19]]) : (index) -> (index, index)
// CHECK:           %[[VAL_22:.*]] = arith.cmpi slt, %[[VAL_21]]#1, %[[VAL_16]]#1 : index
// CHECK:           %[[VAL_23:.*]]:3 = fork(%[[VAL_22]]) : (i1) -> (i1, i1, i1)
// CHECK:           %[[VAL_24:.*]], %[[VAL_25:.*]] = conditional_branch(%[[VAL_23]]#2, %[[VAL_16]]#0) : (i1, index) -> (index, index)
// CHECK:           sink(%[[VAL_25]]) : (index) -> ()
// CHECK:           %[[VAL_26:.*]], %[[VAL_27:.*]] = conditional_branch(%[[VAL_23]]#1, %[[VAL_17]]#0) {control = true} : (i1, none) -> (none, none)
// CHECK:           %[[VAL_28:.*]], %[[VAL_29:.*]] = conditional_branch(%[[VAL_23]]#0, %[[VAL_21]]#0) : (i1, index) -> (index, index)
// CHECK:           sink(%[[VAL_29]]) : (index) -> ()
// CHECK:           %[[VAL_30:.*]] = merge(%[[VAL_28]]) : (index) -> index
// CHECK:           %[[VAL_31:.*]] = merge(%[[VAL_24]]) : (index) -> index
// CHECK:           %[[VAL_32:.*]]:2 = control_merge(%[[VAL_26]]) {control = true} : (none) -> (none, index)
// CHECK:           %[[VAL_33:.*]]:4 = fork(%[[VAL_32]]#0) {control = true} : (none) -> (none, none, none, none)
// CHECK:           %[[VAL_34:.*]]:2 = fork(%[[VAL_33]]#3) {control = true} : (none) -> (none, none)
// CHECK:           %[[VAL_35:.*]] = join(%[[VAL_34]]#1, %[[VAL_6]]#1, %[[VAL_5]]#1, %[[VAL_1]]#2) {control = true} : (none, none, none, none) -> none
// CHECK:           sink(%[[VAL_32]]#1) : (index) -> ()
// CHECK:           %[[VAL_36:.*]] = constant(%[[VAL_34]]#0) {value = 1 : index} : (none) -> index
// CHECK:           %[[VAL_37:.*]]:2 = fork(%[[VAL_36]]) : (index) -> (index, index)
// CHECK:           %[[VAL_38:.*]], %[[VAL_3]] = load(%[[VAL_37]]#1, %[[VAL_1]]#0, %[[VAL_33]]#2) : (index, f32, none) -> (f32, index)
// CHECK:           %[[VAL_39:.*]] = arith.addi %[[VAL_30]], %[[VAL_37]]#0 : index
// CHECK:           %[[VAL_40:.*]]:3 = fork(%[[VAL_39]]) : (index) -> (index, index, index)
// CHECK:           %[[VAL_41:.*]], %[[VAL_4]] = load(%[[VAL_40]]#2, %[[VAL_1]]#1, %[[VAL_33]]#1) : (index, f32, none) -> (f32, index)
// CHECK:           %[[VAL_42:.*]] = arith.addf %[[VAL_38]], %[[VAL_41]] : f32
// CHECK:           %[[VAL_43:.*]] = join(%[[VAL_33]]#0, %[[VAL_6]]#0, %[[VAL_5]]#0) {control = true} : (none, none, none) -> none
// CHECK:           %[[VAL_2]]:2 = store(%[[VAL_42]], %[[VAL_40]]#1, %[[VAL_43]]) : (f32, index, none) -> (f32, index)
// CHECK:           %[[VAL_15]] = branch(%[[VAL_31]]) : (index) -> index
// CHECK:           %[[VAL_18]] = branch(%[[VAL_35]]) {control = true} : (none) -> none
// CHECK:           %[[VAL_20]] = branch(%[[VAL_40]]#0) : (index) -> index
// CHECK:           %[[VAL_44:.*]]:2 = control_merge(%[[VAL_27]]) {control = true} : (none) -> (none, index)
// CHECK:           sink(%[[VAL_44]]#1) : (index) -> ()
// CHECK:           return %[[VAL_44]]#0 : none
// CHECK:         }
// CHECK:       }

    %10 = memref.alloc() : memref<10xf32>
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    br ^bb1(%c0 : index)
  ^bb1(%1: index):      // 2 preds: ^bb0, ^bb2
    %2 = arith.cmpi slt, %1, %c10 : index
    cond_br %2, ^bb2, ^bb3
  ^bb2: // pred: ^bb1
    %c1 = arith.constant 1 : index
    %5 = memref.load %10[%c1] : memref<10xf32>
    %3 = arith.addi %1, %c1 : index
    %7 = memref.load %10[%3] : memref<10xf32>
    %8 = arith.addf %5, %7 : f32
    memref.store %8, %10[%3] : memref<10xf32>
    br ^bb1(%3 : index)
  ^bb3: // pred: ^bb1
    return
  }
