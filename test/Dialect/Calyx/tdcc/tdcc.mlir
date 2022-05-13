// RUN: circt-opt -pass-pipeline='calyx.program(calyx.component(calyx-compile-control))' %s | FileCheck %s

calyx.program "main" {
  calyx.component @Z(%go : i1 {go}, %reset : i1 {reset}, %clk : i1 {clk}) -> (%flag :i1, %done : i1 {done}) {
    %c1_1 = hw.constant 1 : i1
    calyx.wires { calyx.assign %done = %c1_1 : i1 }
    calyx.control {}
  }

  calyx.component @main(%go : i1 {go}, %reset : i1 {reset}, %clk : i1 {clk}) -> (%done : i1 {done}) {
    %z.go, %z.reset, %z.clk, %z.flag, %z.done = calyx.instance @z of @Z : i1, i1, i1, i1, i1
    calyx.wires {
      %undef = calyx.undef : i1
      calyx.group @A {
        %A.go = calyx.group_go %undef : i1
        calyx.assign %z.go = %A.go ? %go : i1
        calyx.group_done %z.done : i1
      }

      calyx.group @B {
        %B.go = calyx.group_go %undef : i1
        calyx.group_done %z.flag ? %z.done : i1
      }
    }

    calyx.control {
      calyx.seq {
        calyx.enable @A
        calyx.enable @B
      }
    }
  }
}
