// RUN: circt-opt -pass-pipeline='calyx.program(calyx.component(calyx-tdcc))' %s

calyx.program "main" {
  calyx.component @main(%go: i1 {go}, %reset: i1 {reset}, %clk: i1 {clk}) -> (%done: i1 {done}) {
    %false = hw.constant false
    %true = hw.constant true
    %lt_reg.in, %lt_reg.write_en, %lt_reg.clk, %lt_reg.reset, %lt_reg.out, %lt_reg.done = calyx.register @lt_reg : i1, i1, i1, i1, i1, i1
    %t.in, %t.write_en, %t.clk, %t.reset, %t.out, %t.done = calyx.register @t : i1, i1, i1, i1, i1, i1
    %f.in, %f.write_en, %f.clk, %f.reset, %f.out, %f.done = calyx.register @f : i1, i1, i1, i1, i1, i1
    %lt.left, %lt.right, %lt.out = calyx.std_lt @lt : i1, i1, i1
    calyx.wires {
      %0 = calyx.undef : i1
      calyx.group @true {
        %true.go = calyx.group_go %0 : i1
        calyx.assign %t.in = %true.go ? %true : i1
        calyx.assign %t.write_en = %true.go ? %true : i1
        calyx.group_done %t.done : i1
      } 
      calyx.group @false {
        %false.go = calyx.group_go %0 : i1
        calyx.assign %f.in = %false.go ? %true : i1
        calyx.assign %f.write_en = %false.go ? %true : i1
        calyx.group_done %f.done : i1
      }
      calyx.group @cond {
        %cond.go = calyx.group_go %0 : i1
        calyx.assign %lt_reg.in = %cond.go ? %lt.out : i1
        calyx.assign %lt_reg.write_en = %cond.go ? %true : i1
        calyx.assign %lt.left = %cond.go ? %true : i1
        calyx.assign %lt.right = %cond.go ? %false : i1
        calyx.group_done %lt_reg.done ? %true : i1
      }
    }
    calyx.control {
      calyx.seq {
        calyx.enable @cond
        calyx.if %lt_reg.out {
          calyx.seq {
            calyx.enable @true
          }
        } else {
          calyx.seq {
            calyx.enable @false
          }
        }
      }
    }
  }
}