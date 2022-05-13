// RUN: circt-opt -pass-pipeline='calyx.program(calyx.component(calyx-tdcc))' %s

calyx.program "main" {
  calyx.component @main(%is_valid: i1, %go: i1 {go}, %reset: i1 {reset}, %clk: i1 {clk}) -> (%done: i1 {done}) {
    %false = hw.constant false
    %true = hw.constant true
    %c0_i32 = hw.constant 0 : i32
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i32, i1, i1, i1, i32, i1
    %is_even.in, %is_even.write_en, %is_even.clk, %is_even.reset, %is_even.out, %is_even.done = calyx.register @is_even : i1, i1, i1, i1, i1, i1
    %is_not_zero.in, %is_not_zero.write_en, %is_not_zero.clk, %is_not_zero.reset, %is_not_zero.out, %is_not_zero.done = calyx.register @is_not_zero : i1, i1, i1, i1, i1, i1
    calyx.wires {
      %0 = calyx.undef : i1
      calyx.group @one {
        %one.go = calyx.group_go %0 : i1
        calyx.assign %is_not_zero.in = %one.go ? %true : i1
        calyx.assign %is_not_zero.write_en = %one.go ? %true : i1
        calyx.group_done %is_not_zero.done : i1
      }
      calyx.group @two {
        %two.go = calyx.group_go %0 : i1
        calyx.assign %r.in = %two.go ? %c0_i32 : i32
        calyx.assign %r.write_en = %two.go ? %false : i1
        calyx.group_done %r.done : i1
      }
    }
    calyx.control {
      calyx.while %is_valid {
        calyx.if %is_even.out {
          calyx.enable @one
        } else {
          calyx.enable @two
        }
      }
    }
  }
}