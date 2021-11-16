  // NOTE: Assertions have been autogenerated by utils/update_mlir_test_checks.py
// RUN: circt-opt -lower-std-to-handshake %s | FileCheck %s
  func @if_only() {
// CHECK:       module {

// CHECK-LABEL:   handshake.func @if_only(
// CHECK-SAME:                            %[[VAL_0:.*]]: none, ...) -> none attributes {argNames = ["inCtrl"], resNames = ["outCtrl"]} {
// CHECK:           %[[VAL_1:.*]]:4 = fork(%[[VAL_0]]) {control = true} : (none) -> (none, none, none, none)
// CHECK:           %[[VAL_2:.*]] = constant(%[[VAL_1]]#2) {value = 0 : index} : (none) -> index
// CHECK:           %[[VAL_3:.*]]:2 = fork(%[[VAL_2]]) : (index) -> (index, index)
// CHECK:           %[[VAL_4:.*]] = constant(%[[VAL_1]]#1) {value = -1 : index} : (none) -> index
// CHECK:           %[[VAL_5:.*]] = arith.muli %[[VAL_3]]#0, %[[VAL_4]] : index
// CHECK:           %[[VAL_6:.*]] = constant(%[[VAL_1]]#0) {value = 20 : index} : (none) -> index
// CHECK:           %[[VAL_7:.*]] = arith.addi %[[VAL_5]], %[[VAL_6]] : index
// CHECK:           %[[VAL_8:.*]] = arith.cmpi sge, %[[VAL_7]], %[[VAL_3]]#1 : index
// CHECK:           %[[VAL_9:.*]], %[[VAL_10:.*]] = conditional_branch(%[[VAL_8]], %[[VAL_1]]#3) {control = true} : (i1, none) -> (none, none)
// CHECK:           %[[VAL_11:.*]]:2 = control_merge(%[[VAL_9]]) {control = true} : (none) -> (none, index)
// CHECK:           sink(%[[VAL_11]]#1) : (index) -> ()
// CHECK:           %[[VAL_12:.*]] = branch(%[[VAL_11]]#0) {control = true} : (none) -> none
// CHECK:           %[[VAL_13:.*]]:2 = control_merge(%[[VAL_12]], %[[VAL_10]]) {control = true} : (none, none) -> (none, index)
// CHECK:           sink(%[[VAL_13]]#1) : (index) -> ()
// CHECK:           return %[[VAL_13]]#0 : none
// CHECK:         }
// CHECK:       }

    %c0 = arith.constant 0 : index
    %c-1 = arith.constant -1 : index
    %1 = arith.muli %c0, %c-1 : index
    %c20 = arith.constant 20 : index
    %2 = arith.addi %1, %c20 : index
    %3 = arith.cmpi sge, %2, %c0 : index
    cond_br %3, ^bb1, ^bb2
  ^bb1: // pred: ^bb0
    br ^bb2
  ^bb2: // 2 preds: ^bb0, ^bb1
    return
  }
