// RUN: circt-opt --create-esi-wrapper %s | FileCheck %s

// CHECK-LABEL: hw.module @adder_esi(%in0: !esi.channel<i32>, %in1: !esi.channel<i32>, %clk: i1, %reset: i1) -> (out0: !esi.channel<i32>) {
// CHECK:         %0 = comb.xor %reset : i1
// CHECK:         %1 = esi.buffer %clk, %0, %in0 {stages = 1 : i64} : i32
// CHECK:         %2 = esi.buffer %clk, %0, %in1 {stages = 1 : i64} : i32
// CHECK:         %rawOutput, %valid = esi.unwrap.vr %1, %11 : i32
// CHECK:         %rawOutput_0, %valid_1 = esi.unwrap.vr %2, %11 : i32
// CHECK:         %adder.out0, %adder.done = hw.instance "adder" @adder(in0: %rawOutput: i32, in1: %rawOutput_0: i32, clk: %clk: i1, reset: %reset: i1, go: %11: i1) -> (out0: i32, done: i1)
// CHECK:         %chanOutput, %ready = esi.wrap.vr %adder.out0, %adder.done : i32
// CHECK:         %3 = esi.buffer %clk, %0, %chanOutput {stages = 1 : i64} : i32
// CHECK:         %false = hw.constant false
// CHECK:         %runReg = seq.compreg %6, %clk, %reset, %false  : i1
// CHECK:         %4 = comb.xor %adder.done : i1
// CHECK:         %5 = comb.and %runReg, %4 : i1
// CHECK:         %6 = comb.or %5, %11 : i1
// CHECK:         %7 = comb.xor %runReg : i1
// CHECK:         %8 = comb.and %7, %valid : i1
// CHECK:         %9 = comb.and %8, %valid_1 : i1
// CHECK:         %10 = comb.and %9, %adder.done : i1
// CHECK:         %11 = comb.and %10, %7 : i1
// CHECK:         hw.output %3 : !esi.channel<i32>
// CHECK:       }
// CHECK-LABEL: hw.module.extern @adder(%in0: i32, %in1: i32, %clk: i1, %reset: i1, %go: i1) -> (out0: i32, done: i1)

module  {
  calyx.program "adder"  {
    calyx.component @adder(%in0: i32, %in1: i32, %clk: i1 {clk}, %reset: i1 {reset}, %go: i1 {go}) -> (%out0: i32, %done: i1 {done}) {
      %true = hw.constant true
      %ret_arg0_reg.in, %ret_arg0_reg.write_en, %ret_arg0_reg.clk, %ret_arg0_reg.reset, %ret_arg0_reg.out, %ret_arg0_reg.done = calyx.register @ret_arg0_reg : i32, i1, i1, i1, i32, i1
      calyx.wires  {
        calyx.assign %out0 = %ret_arg0_reg.out : i32
        calyx.group @ret_assign_0  {
          calyx.group_done %ret_arg0_reg.done : i1
        }
      }
      calyx.control  {
        calyx.seq  {
          calyx.enable @ret_assign_0
        }
      }
    }
  }
}
