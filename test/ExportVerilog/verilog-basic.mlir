// RUN: circt-translate %s -export-verilog -verify-diagnostics --lowering-options=alwaysFF | FileCheck %s --strict-whitespace

// CHECK-LABEL: module inputs_only(
// CHECK-NEXT: input a, b);
hw.module @inputs_only(%a: i1, %b: i1) {
  hw.output
}

// CHECK-LABEL: module no_ports();
hw.module @no_ports() {
  hw.output
}

// CHECK-LABEL: module Expressions(
// CHECK-NEXT:    input  [3:0]  in4,
// CHECK-NEXT:    input         clock,
// CHECK-NEXT:    output        out1a, out1b, out1c,
// CHECK-NEXT:    output [3:0]  out4, out4s,
// CHECK-NEXT:    output [15:0] out16, out16s,
// CHECK-NEXT:    output [16:0] sext17);

hw.module @Expressions(%in4: i4, %clock: i1) ->
  (%out1a: i1, %out1b: i1, %out1c: i1,
   %out4: i4, %out4s: i4, %out16: i16, %out16s: i16, %sext17: i17) {
  %c1_i4 = hw.constant 1 : i4
  %c2_i4 = hw.constant 2 : i4
  %c3_i4 = hw.constant 3 : i4
  %c-1_i4 = hw.constant -1 : i4
  %c0_i4 = hw.constant 0 : i4
  %false = hw.constant false
  %c0_i2 = hw.constant 0 : i2
  %c0_i5 = hw.constant 0 : i5
  %c0_i6 = hw.constant 0 : i6
  %c0_i10 = hw.constant 0 : i10

  // CHECK: wire [3:0] _T = in4 >> in4;
  %7 = comb.extract %in4 from 2 : (i4) -> i1

  %10 = comb.shru %in4, %in4 : i4

  // CHECK: wire [1:0] _T_0 = in4[1:0];
  // CHECK: wire [1:0] _T_1 = in4[3:2];
  // CHECK: wire [8:0] _T_2 = {1'h0, in4, in4};
  // CHECK: wire [4:0] _T_3 = 5'h0 - {in4[3], in4};

  // CHECK: assign w1 = ~in4;
  %3 = comb.xor %in4, %c-1_i4 : i4

  // CHECK: assign w1 = in4 % 4'h1;
  %4 = comb.modu %in4, %c1_i4 : i4

  // CHECK: assign w1 = {2'h0, _T_0};
  %5 = comb.extract %in4 from 0 : (i4) -> i2

  // CHECK: assign w1 = {2'h0, _T_1 | {in4[2], 1'h0}};
  %6 = comb.extract %in4 from 2 : (i4) -> i2
  %8 = comb.concat %7, %false : (i1, i1) -> i2
  %9 = comb.or %6, %8 : i2

  // CHECK: assign w1 = _T;
  // CHECK: assign w1 = clock ? (clock ? 4'h1 : 4'h2) : 4'h3;
  // CHECK: assign w1 = clock ? 4'h1 : clock ? 4'h2 : 4'h3;
  %11 = comb.shrs %in4, %in4 : i4
  %12 = comb.concat %false, %in4, %in4 : (i1, i4, i4) -> i9
  %13 = comb.mux %clock, %c1_i4, %c2_i4 : i4
  %14 = comb.mux %clock, %13, %c3_i4 : i4
  %15 = comb.mux %clock, %c2_i4, %c3_i4 : i4
  %16 = comb.mux %clock, %c1_i4, %15 : i4

  // CHECK: assign w1 = {2'h0, _T_1 | _T_0};
  %17 = comb.or %6, %5 : i2
  %18 = comb.concat %c0_i2, %in4 : (i2, i4) -> i6

  // CHECK: assign w2 = {6'h0, in4, clock, clock, in4};
  // CHECK: assign w2 = {10'h0, {2'h0, in4} ^ {{..}}2{in4[3]}}, in4} ^ {6{clock}}};
  %19 = comb.sext %in4 : (i4) -> i6
  %20 = comb.sext %clock : (i1) -> i6
  %21 = comb.xor %18, %19, %20 : i6
  %22 = comb.sext %in4 : (i4) -> i5
  %23 = comb.sub %c0_i5, %22 : i5
  %25 = comb.concat %c0_i2, %5 : (i2, i2) -> i4
  %26 = comb.concat %c0_i2, %9 : (i2, i2) -> i4
  %27 = comb.concat %c0_i2, %17 : (i2, i2) -> i4

  %w1 = sv.wire : !hw.inout<i4>
  %w1_use = sv.read_inout %w1 : !hw.inout<i4>

  sv.assign %w1, %3 : i4
  sv.assign %w1, %4 : i4
  sv.assign %w1, %25 : i4
  sv.assign %w1, %26 : i4
  sv.assign %w1, %10 : i4
  sv.assign %w1, %14 : i4
  sv.assign %w1, %16 : i4
  sv.assign %w1, %27 : i4
  sv.assign %w1, %10 : i4
  
  %29 = comb.concat %c0_i6, %in4, %clock, %clock, %in4 : (i6, i4, i1, i1, i4) -> i16
  %30 = comb.concat %c0_i10, %21 : (i10, i6) -> i16

  %w2 = sv.wire : !hw.inout<i16>
  %w2_use = sv.read_inout %w2 : !hw.inout<i16>
  sv.assign %w2, %29 : i16
  sv.assign %w2, %30 : i16

  // CHECK: assign w3 = {{..}}7{_T_2[8]}}, _T_2};
  // CHECK: assign w3 = {{..}}11{_T_3[4]}}, _T_3};
  %32 = comb.sext %12 : (i9) -> i16
  %33 = comb.sext %23 : (i5) -> i16

  %w3 = sv.wire : !hw.inout<i16>
  %w3_use = sv.read_inout %w3 : !hw.inout<i16>
  sv.assign %w3, %32 : i16
  sv.assign %w3, %33 : i16


 // CHECK: assign out1a = ^in4;
  %0 = comb.parity %in4 : i4
  // CHECK: assign out1b = &in4;
  %1 = comb.icmp eq %in4, %c-1_i4 : i4
  // CHECK: assign out1c = |in4;
  %2 = comb.icmp ne %in4, %c0_i4 : i4

  // CHECK: assign out4s = $signed($signed(in4) >>> in4);
  // CHECK: assign sext17 = {w3[15], w3};
  %35 = comb.sext %w3_use : (i16) -> i17
  hw.output %0, %1, %2, %w1_use, %11, %w2_use, %w3_use, %35 : i1, i1, i1, i4, i4, i16, i16, i17
}

// CHECK-LABEL: module Precedence(
hw.module @Precedence(%a: i4, %b: i4, %c: i4) -> (%out1: i1, %out: i10) {
  %false = hw.constant false
  %c0_i2 = hw.constant 0 : i2
  %c0_i4 = hw.constant 0 : i4
  %c0_i5 = hw.constant 0 : i5
  %c0_i3 = hw.constant 0 : i3
  %c0_i6 = hw.constant 0 : i6
  %_out1_output = sv.wire  : !hw.inout<i1>
  %_out_output = sv.wire  : !hw.inout<i10>

  // CHECK: wire [4:0] _T = {1'h0, b};
  // CHECK: wire [4:0] _T_0 = _T + {1'h0, c};
  // CHECK: wire [5:0] _T_1 = {2'h0, a};
  // CHECK: wire [5:0] _T_2 = {1'h0, _T_0};
  // CHECK: assign _out_output = {4'h0, _T_1 + _T_2};
  // CHECK: wire [4:0] _T_3 = {1'h0, a} + _T;
  // CHECK: assign _out_output = {4'h0, {1'h0, _T_3} - {2'h0, c}};
  // CHECK: assign _out_output = {4'h0, _T_1 - _T_2};
  // CHECK: wire [7:0] _T_4 = {4'h0, b};
  // CHECK: wire [8:0] _T_5 = {5'h0, a};
  // CHECK: assign _out_output = {1'h0, _T_5 + {1'h0, _T_4 * {4'h0, c}}};
  // CHECK: wire [8:0] _T_6 = {5'h0, c};
  // CHECK: assign _out_output = {1'h0, {1'h0, {4'h0, a} * _T_4} + _T_6};
  // CHECK: assign _out_output = {1'h0, {4'h0, _T_3} * _T_6};
  // CHECK: assign _out_output = {1'h0, _T_5 * {4'h0, _T_0}};
  // CHECK: assign _out_output = {5'h0, _T_3} * {5'h0, _T_0};
  // CHECK: assign _out1_output = ^_T_0;
  // CHECK: assign _out1_output = b < c | b > c;
  // CHECK: assign _out_output = {6'h0, (b ^ c) & {3'h0, _out1_output}};
  // CHECK: assign _out_output = {2'h0, _out_output[9:2]};
  // CHECK: assign _out1_output = _out_output < {6'h0, a};
  %0 = comb.concat %false, %b : (i1, i4) -> i5
  %1 = comb.concat %false, %c : (i1, i4) -> i5
  %2 = comb.add %0, %1 : i5
  %3 = comb.concat %c0_i2, %a : (i2, i4) -> i6
  %4 = comb.concat %false, %2 : (i1, i5) -> i6
  %5 = comb.add %3, %4 : i6
  %6 = comb.concat %c0_i4, %5 : (i4, i6) -> i10
  sv.assign %_out_output, %6 : i10
  %7 = comb.concat %false, %a : (i1, i4) -> i5
  %8 = comb.add %7, %0 : i5
  %9 = comb.concat %false, %8 : (i1, i5) -> i6
  %10 = comb.concat %c0_i2, %c : (i2, i4) -> i6
  %11 = comb.sub %9, %10 : i6
  %12 = comb.concat %c0_i4, %11 : (i4, i6) -> i10
  sv.assign %_out_output, %12 : i10
  %13 = comb.sub %3, %4 : i6
  %14 = comb.concat %c0_i4, %13 : (i4, i6) -> i10
  sv.assign %_out_output, %14 : i10
  %15 = comb.concat %c0_i4, %b : (i4, i4) -> i8
  %16 = comb.concat %c0_i4, %c : (i4, i4) -> i8
  %17 = comb.mul %15, %16 : i8
  %18 = comb.concat %c0_i5, %a : (i5, i4) -> i9
  %19 = comb.concat %false, %17 : (i1, i8) -> i9
  %20 = comb.add %18, %19 : i9
  %21 = comb.concat %false, %20 : (i1, i9) -> i10
  sv.assign %_out_output, %21 : i10
  %22 = comb.concat %c0_i4, %a : (i4, i4) -> i8
  %23 = comb.mul %22, %15 : i8
  %24 = comb.concat %false, %23 : (i1, i8) -> i9
  %25 = comb.concat %c0_i5, %c : (i5, i4) -> i9
  %26 = comb.add %24, %25 : i9
  %27 = comb.concat %false, %26 : (i1, i9) -> i10
  sv.assign %_out_output, %27 : i10
  %28 = comb.concat %c0_i4, %8 : (i4, i5) -> i9
  %29 = comb.mul %28, %25 : i9
  %30 = comb.concat %false, %29 : (i1, i9) -> i10
  sv.assign %_out_output, %30 : i10
  %31 = comb.concat %c0_i4, %2 : (i4, i5) -> i9
  %32 = comb.mul %18, %31 : i9
  %33 = comb.concat %false, %32 : (i1, i9) -> i10
  sv.assign %_out_output, %33 : i10
  %34 = comb.concat %c0_i5, %8 : (i5, i5) -> i10
  %35 = comb.concat %c0_i5, %2 : (i5, i5) -> i10
  %36 = comb.mul %34, %35 : i10
  sv.assign %_out_output, %36 : i10
  %37 = comb.parity %2 : i5
  sv.assign %_out1_output, %37 : i1
  %38 = comb.icmp ult %b, %c : i4
  %39 = comb.icmp ugt %b, %c : i4
  %40 = comb.or %38, %39 : i1
  sv.assign %_out1_output, %40 : i1
  %41 = comb.xor %b, %c : i4
  %42 = sv.read_inout %_out1_output : !hw.inout<i1>
  %43 = comb.concat %c0_i3, %42 : (i3, i1) -> i4
  %44 = comb.and %41, %43 : i4
  %45 = comb.concat %c0_i6, %44 : (i6, i4) -> i10
  sv.assign %_out_output, %45 : i10
  %46 = sv.read_inout %_out_output : !hw.inout<i10>
  %47 = comb.extract %46 from 2 : (i10) -> i8
  %48 = comb.concat %c0_i2, %47 : (i2, i8) -> i10
  sv.assign %_out_output, %48 : i10
  %49 = comb.concat %c0_i6, %a : (i6, i4) -> i10
  %50 = comb.icmp ult %46, %49 : i10
  sv.assign %_out1_output, %50 : i1
  hw.output %42, %46 : i1, i10
}

// CHECK-LABEL: module CmpSign(
hw.module @CmpSign(%a: i4, %b: i4, %c: i4, %d: i4) ->
 (%o0: i1, %o1: i1, %o2: i1, %o3: i1, %o4: i1, %o5: i1, %o6: i1, %o7: i1,
  %o8: i1, %o9: i1, %o10: i1, %o11: i1, %o12: i1, %o13: i1, %o14: i1, %o15: i1) {
  // CHECK: assign o0 = a < b;
  %0 = comb.icmp ult %a, %b : i4
  // CHECK-NEXT: assign o1 = $signed(c) < $signed(d);
  %1 = comb.icmp slt %c, %d : i4
  // CHECK-NEXT: assign o2 = $signed(a) < $signed(b);
  %2 = comb.icmp slt %a, %b : i4
  // CHECK-NEXT: assign o3 = a <= b;
  %3 = comb.icmp ule %a, %b : i4
  // CHECK-NEXT: assign o4 = $signed(c) <= $signed(d);
  %4 = comb.icmp sle %c, %d : i4
  // CHECK-NEXT: assign o5 = $signed(a) <= $signed(b);
  %5 = comb.icmp sle %a, %b : i4
  // CHECK-NEXT: assign o6 = a > b;
  %6 = comb.icmp ugt %a, %b : i4
  // CHECK-NEXT: assign o7 = $signed(c) > $signed(d);
  %7 = comb.icmp sgt %c, %d : i4
  // CHECK-NEXT: assign o8 = $signed(a) > $signed(b);
  %8 = comb.icmp sgt %a, %b : i4
  // CHECK-NEXT: assign o9 = a >= b;
  %9 = comb.icmp uge %a, %b : i4
  // CHECK-NEXT: assign o10 = $signed(c) >= $signed(d);
  // CHECK-NEXT: assign o11 = $signed(a) >= $signed(b);
  %10 = comb.icmp sge %c, %d : i4
  %11 = comb.icmp sge %a, %b : i4
  // CHECK-NEXT: assign o12 = a == b;
  %12 = comb.icmp eq %a, %b : i4
  // CHECK-NEXT: assign o13 = c == d;
  %13 = comb.icmp eq %c, %d : i4
  // CHECK-NEXT: assign o14 = a != b;
  %14 = comb.icmp ne %a, %b : i4
  // CHECK-NEXT: assign o15 = c != d;
  %15 = comb.icmp ne %c, %d : i4

  hw.output %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15 : i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1
}

// CHECK-LABEL: module MultiUseExpr
hw.module @MultiUseExpr(%a: i4) ->
 (%b0: i1, %b1: i1, %b2: i1, %b3: i1, %b4: i2) {
  %false = hw.constant false
  %c1_i5 = hw.constant 1 : i5
  %c-1_i5 = hw.constant -1 : i5
  %c-1_i4 = hw.constant -1 : i4

  // CHECK: wire _T = ^a;
  %0 = comb.parity %a : i4
  // CHECK-NEXT: wire [4:0] _T_0 = {1'h0, a} << 5'h1;
  %1 = comb.concat %false, %a : (i1, i4) -> i5
  %2 = comb.shl %1, %c1_i5 : i5

  // CHECK-NEXT: wire [3:0] _T_1 = ~a;
  // CHECK-NEXT: assign b0 = _T;
  // CHECK-NEXT: assign b1 = ^_T;
  // CHECK-NEXT: assign b2 = &_T_0;
  // CHECK-NEXT: assign b3 = ^_T_0;
  // CHECK-NEXT: assign b4 = _T_1[3:2];
  %3 = comb.parity %0 : i1
  %4 = comb.icmp eq %2, %c-1_i5 : i5
  %5 = comb.parity %2 : i5
  %6 = comb.xor %a, %c-1_i4 : i4
  %7 = comb.extract %6 from 2 : (i4) -> i2
  hw.output %0, %3, %4, %5, %7 : i1, i1, i1, i1, i2
}

hw.module.extern @MyExtModule(%in: i8) -> (%out: i1) attributes {verilogName = "FooExtModule"}
hw.module.extern @MyParameterizedExtModule(%in: i8) -> (%out: i1)

// CHECK-LABEL: module ExternMods
hw.module @ExternMods(%a_in: i8) {
  // CHECK: MyParameterizedExtModule #(
  // CHECK:   .CFG(FOO)
  // CHECK: ) xyz2
  hw.instance "xyz2" @MyParameterizedExtModule(%a_in) {parameters = {CFG = #sv.verbatim.parameter<"FOO">}} : (i8) -> i1
  // CHECK: MyParameterizedExtModule #(
  // CHECK:   .CFG("STRING")
  // CHECK: ) xyz3
  hw.instance "xyz3" @MyParameterizedExtModule(%a_in) {parameters = {CFG = #sv.verbatim.parameter<"\"STRING\"">}} : (i8) -> i1
}

// CHECK-LABEL: module UseInstances
hw.module @UseInstances(%a_in: i8) -> (%a_out1: i1, %a_out2: i1) {
  // CHECK: FooExtModule xyz (
  // CHECK:   .in  (a_in),
  // CHECK:   .out (a_out1)
  // CHECK: );
  // CHECK: MyParameterizedExtModule #(
  // CHECK:   .DEFAULT(64'd0),
  // CHECK:   .DEPTH(3.500000e+00),
  // CHECK:   .FORMAT("xyz_timeout=%d\n"),
  // CHECK:   .WIDTH(32)
  // CHECK: ) xyz2 (
  // CHECK:   .in  (a_in),
  // CHECK:   .out (a_out2)
  // CHECK: );
  %xyz.out = hw.instance "xyz" @MyExtModule(%a_in) : (i8) -> i1
  %xyz2.out = hw.instance "xyz2" @MyParameterizedExtModule(%a_in) {parameters = {DEFAULT = 0 : i64, DEPTH = 3.500000e+00 : f64, FORMAT = "xyz_timeout=%d\0A", WIDTH = 32 : i8}} : (i8) -> i1
  hw.output %xyz.out, %xyz2.out : i1, i1
}

// CHECK-LABEL: module Stop(
hw.module @Stop(%clock: i1, %reset: i1) {
  // CHECK: always @(posedge clock) begin
  // CHECK:   `ifndef SYNTHESIS
  // CHECK:     if (`STOP_COND_ & reset)
  // CHECK:       $fatal;
  // CHECK:   `endif
  // CHECK: end // always @(posedge)
  sv.always posedge %clock  {
    sv.ifdef.procedural "SYNTHESIS"  {
    } else  {
      %0 = sv.verbatim.expr "`STOP_COND_" : () -> i1
      %1 = comb.and %0, %reset : i1
      sv.if %1  {
        sv.fatal
      }
    }
  }
  hw.output
}

// CHECK-LABEL: module Print
hw.module @Print(%clock: i1, %reset: i1, %a: i4, %b: i4) {
  %false = hw.constant false
  %c1_i5 = hw.constant 1 : i5

  // CHECK: wire [4:0] _T = {1'h0, a} << 5'h1;
  // CHECK: always @(posedge clock) begin
  // CHECK:   if (`PRINTF_COND_ & reset)
  // CHECK:     $fwrite(32'h80000002, "Hi %x %x\n", _T, b);
  // CHECK: end // always @(posedge)
  %0 = comb.concat %false, %a : (i1, i4) -> i5
  %1 = comb.shl %0, %c1_i5 : i5
  sv.always posedge %clock  {
    %2 = sv.verbatim.expr "`PRINTF_COND_" : () -> i1
    %3 = comb.and %2, %reset : i1
    sv.if %3  {
      sv.fwrite "Hi %x %x\0A"(%1, %b) : i5, i4
    }
  }
  hw.output
}

// CHECK-LABEL: module UninitReg1(
hw.module @UninitReg1(%clock: i1, %reset: i1, %cond: i1, %value: i2) {
  %c-1_i2 = hw.constant -1 : i2
  %count = sv.reg  : !hw.inout<i2>

  // CHECK: wire [1:0] _T = ~{2{reset}} & (cond ? value : count);
  // CHECK-NEXT: always_ff @(posedge clock)
  // CHECK-NEXT:   count <= _T;

  %0 = sv.read_inout %count : !hw.inout<i2>
  %1 = comb.mux %cond, %value, %0 : i2
  %2 = comb.sext %reset : (i1) -> i2
  %3 = comb.xor %2, %c-1_i2 : i2
  %4 = comb.and %3, %1 : i2
  sv.alwaysff(posedge %clock)  {
    sv.passign %count, %4 : i2
  }
  hw.output
}

// https://github.com/llvm/circt/issues/755
// CHECK-LABEL: module UnaryParensIssue755(
// CHECK: assign b = |(~a);
hw.module @UnaryParensIssue755(%a: i8) -> (%b: i1) {
  %c-1_i8 = hw.constant -1 : i8
  %c0_i8 = hw.constant 0 : i8
  %0 = comb.xor %a, %c-1_i8 : i8
  %1 = comb.icmp ne %0, %c0_i8 : i8
  hw.output %1 : i1
}

sv.bind @__BindEmissionInstance__ in @BindEmission {output_file = {directory = "BindTest", exclude_from_filelist = true, exclude_replicated_ops = true, name = "BindEmissionInstance.sv"}}
// CHECK-LABL: module BindEmissionInstance()
hw.module @BindEmissionInstance() {
  hw.output
}
// CHECK-LABEL: module BindEmission()
hw.module @BindEmission() -> () {
  // CHECK-NEXT: // This instance is elsewhere emitted as a bind statement
  // CHECK-NEXT: // BindEmissionInstance BindEmissionInstance ();
  hw.instance "BindEmissionInstance" sym @__BindEmissionInstance__ @BindEmissionInstance() {doNotPrint = true} : () -> ()
  hw.output
}

sv.bind.interface @__Interface__ {output_file = {directory = "BindTest", exclude_from_filelist = true, exclude_replicated_ops = true, name = "BindInterface.sv"}}
sv.interface @Interface {
  sv.interface.signal @a : i1
  sv.interface.signal @b : i1
}

hw.module @BindInterface() -> () {
  %bar = sv.interface.instance sym @__Interface__ {doNotPrint = true} : !sv.interface<@Interface>
  hw.output
}

// CHECK-LABEL: FILE "BindTest{{.}}BindEmissionInstance.sv"
// CHECK: bind BindEmission BindEmissionInstance BindEmissionInstance ();

// CHECK-LABEL: FILE "BindTest{{.}}BindInterface.sv"
// CHECK: bind BindInterface Interface bar (.*);
