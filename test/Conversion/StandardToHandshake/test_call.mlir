// RUN: circt-opt -lower-std-to-handshake -split-input-file %s | FileCheck %s

func @bar(%0 : i32) -> i32 {
// CHECK-LABEL:   handshake.func @bar(
// CHECK-SAME:                        %[[VAL_0:.*]]: i32,
// CHECK-SAME:                        %[[VAL_1:.*]]: none, ...) -> (i32, none) attributes {argNames = ["in0", "inCtrl"], resNames = ["out0", "outCtrl"]} {
// CHECK:           %[[VAL_2:.*]] = merge(%[[VAL_0]]) : (i32) -> i32
// CHECK:           return %[[VAL_2]], %[[VAL_1]] : i32, none
// CHECK:         }

  return %0 : i32
}

func @foo(%0 : i32) -> i32 {
// CHECK-LABEL:   handshake.func @foo(
// CHECK-SAME:                        %[[VAL_0:.*]]: i32,
// CHECK-SAME:                        %[[VAL_1:.*]]: none, ...) -> (i32, none) attributes {argNames = ["in0", "inCtrl"], resNames = ["out0", "outCtrl"]} {
// CHECK:           %[[VAL_2:.*]] = merge(%[[VAL_0]]) : (i32) -> i32
// CHECK:           %[[VAL_3:.*]]:2 = fork(%[[VAL_1]]) {control = true} : (none) -> (none, none)
// CHECK:           %[[VAL_4:.*]]:2 = instance @bar(%[[VAL_2]], %[[VAL_3]]#0) : (i32, none) -> (i32, none)
// CHECK:           sink(%[[VAL_4]]#1) {control = true} : (none) -> ()
// CHECK:           return %[[VAL_4]]#0, %[[VAL_3]]#1 : i32, none
// CHECK:         }

  %a1 = call @bar(%0) : (i32) -> i32
  return %a1 : i32
}

// -----

// Branching control flow with calls in each branch.

// CHECK-LABEL:   handshake.func @add(
func @add(%arg0 : i32, %arg1: i32) -> i32 {
  %0 = arith.addi %arg0, %arg1 : i32
  return %0 : i32
}

// CHECK-LABEL:   handshake.func @sub(
func @sub(%arg0 : i32, %arg1: i32) -> i32 {
  %0 = arith.subi %arg0, %arg1 : i32
  return %0 : i32
}

// CHECK:   handshake.func @main(%[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: none, ...) -> (i32, none) attributes {argNames = ["in0", "in1", "in2", "inCtrl"], resNames = ["out0", "outCtrl"]} {
// CHECK:           %[[VAL_4:.*]] = merge(%[[VAL_0]]) : (i32) -> i32
// CHECK:           %[[VAL_5:.*]] = merge(%[[VAL_1]]) : (i32) -> i32
// CHECK:           %[[VAL_6:.*]] = merge(%[[VAL_2]]) : (i1) -> i1
// CHECK:           %[[VAL_7:.*]]:3 = fork(%[[VAL_6]]) : (i1) -> (i1, i1, i1)
// CHECK:           %[[VAL_8:.*]], %[[VAL_9:.*]] = conditional_branch(%[[VAL_7]]#2, %[[VAL_4]]) : (i1, i32) -> (i32, i32)
// CHECK:           %[[VAL_10:.*]], %[[VAL_11:.*]] = conditional_branch(%[[VAL_7]]#1, %[[VAL_5]]) : (i1, i32) -> (i32, i32)
// CHECK:           %[[VAL_12:.*]], %[[VAL_13:.*]] = conditional_branch(%[[VAL_7]]#0, %[[VAL_3]]) {control = true} : (i1, none) -> (none, none)
// CHECK:           %[[VAL_14:.*]] = merge(%[[VAL_8]]) : (i32) -> i32
// CHECK:           %[[VAL_15:.*]] = merge(%[[VAL_10]]) : (i32) -> i32
// CHECK:           %[[VAL_16:.*]]:2 = control_merge(%[[VAL_12]]) {control = true} : (none) -> (none, index)
// CHECK:           %[[VAL_17:.*]]:2 = fork(%[[VAL_16]]#0) {control = true} : (none) -> (none, none)
// CHECK:           sink(%[[VAL_16]]#1) : (index) -> ()
// CHECK:           %[[VAL_18:.*]]:2 = handshake.instance @add(%[[VAL_14]], %[[VAL_15]], %[[VAL_17]]#1) : (i32, i32, none) -> (i32, none)
// CHECK:           sink(%[[VAL_18]]#1) {control = true} : (none) -> ()
// CHECK:           %[[VAL_19:.*]] = branch(%[[VAL_17]]#0) {control = true} : (none) -> none
// CHECK:           %[[VAL_20:.*]] = branch(%[[VAL_18]]#0) : (i32) -> i32
// CHECK:           %[[VAL_21:.*]] = merge(%[[VAL_9]]) : (i32) -> i32
// CHECK:           %[[VAL_22:.*]] = merge(%[[VAL_11]]) : (i32) -> i32
// CHECK:           %[[VAL_23:.*]]:2 = control_merge(%[[VAL_13]]) {control = true} : (none) -> (none, index)
// CHECK:           %[[VAL_24:.*]]:2 = fork(%[[VAL_23]]#0) {control = true} : (none) -> (none, none)
// CHECK:           sink(%[[VAL_23]]#1) : (index) -> ()
// CHECK:           %[[VAL_25:.*]]:2 = handshake.instance @sub(%[[VAL_21]], %[[VAL_22]], %[[VAL_24]]#1) : (i32, i32, none) -> (i32, none)
// CHECK:           sink(%[[VAL_25]]#1) {control = true} : (none) -> ()
// CHECK:           %[[VAL_26:.*]] = branch(%[[VAL_24]]#0) {control = true} : (none) -> none
// CHECK:           %[[VAL_27:.*]] = branch(%[[VAL_25]]#0) : (i32) -> i32
// CHECK:           %[[VAL_28:.*]]:2 = control_merge(%[[VAL_26]], %[[VAL_19]]) {control = true} : (none, none) -> (none, index)
// CHECK:           %[[VAL_29:.*]] = mux(%[[VAL_28]]#1, %[[VAL_27]], %[[VAL_20]]) : (index, i32, i32) -> i32
// CHECK:           return %[[VAL_29]], %[[VAL_28]]#0 : i32, none
// CHECK:         }
func @main(%arg0 : i32, %arg1 : i32, %cond : i1) -> i32 {
  cond_br %cond, ^bb1, ^bb2
^bb1:
  %0 = call @add(%arg0, %arg1) : (i32, i32) -> i32
  br ^bb3(%0 : i32)
^bb2:
  %1 = call @sub(%arg0, %arg1) : (i32, i32) -> i32
  br ^bb3(%1 : i32)
^bb3(%res : i32):
  return %res : i32
}
