// RUN: circt-opt -pass-pipeline='calyx.program(calyx.component(calyx-tdcc))' %s

calyx.program "main" {
  calyx.component @main(%go: i1 {go}, %reset: i1 {reset}, %clk: i1 {clk}) -> (%done: i1 {done}) {
    %true = hw.constant true
    %c1_i32 = hw.constant 1 : i32
    %c5_i32 = hw.constant 5 : i32
    %c4_i32 = hw.constant 4 : i32
    %lt_reg.in, %lt_reg.write_en, %lt_reg.clk, %lt_reg.reset, %lt_reg.out, %lt_reg.done = calyx.register @lt_reg : i1, i1, i1, i1, i1, i1
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i32, i1, i1, i1, i32, i1
    %add.left, %add.right, %add.out = calyx.std_add @add : i32, i32, i32
    %lt.left, %lt.right, %lt.out = calyx.std_lt @lt : i32, i32, i1
    calyx.wires {
      %0 = calyx.undef : i1
      calyx.group @do_add {
        %do_add.go = calyx.group_go %0 : i1
        calyx.assign %add.right = %do_add.go ? %c4_i32 : i32
        calyx.assign %add.left = %do_add.go ? %c4_i32 : i32
        calyx.assign %r.in = %do_add.go ? %add.out : i32
        calyx.assign %r.write_en = %do_add.go ? %true : i1
        calyx.group_done %r.done : i1
      }
      calyx.group @cond {
        %cond.go = calyx.group_go %0 : i1
        calyx.assign %lt_reg.in = %cond.go ? %lt.out : i1
        calyx.assign %lt_reg.write_en = %cond.go ? %true : i1
        calyx.assign %lt.right = %cond.go ? %c5_i32 : i32
        calyx.assign %lt.left = %cond.go ? %c1_i32 : i32
        calyx.group_done %lt_reg.done ? %true : i1
      }
    }
    calyx.control {
      calyx.seq {
        calyx.enable @cond
        calyx.while %lt_reg.out {
          calyx.seq {
            calyx.enable @do_add
            calyx.enable @cond
          }
        }
      }
    }
  }
}
