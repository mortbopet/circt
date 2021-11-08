// RUN: circt-opt -lower-std-to-handshake -verify-diagnostics %s 

// expected-error @+1 {{'builtin.module' op symbol 'bar' called in both ESI and non-ESI context, which is illegal.}}
module {
  func @main() {
    call @bar() : () -> ()
    call @bar() {ESI} : () -> ()
    return
  }
  func private @bar() -> ()
}
