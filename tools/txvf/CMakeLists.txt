get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

file(GLOB globbed *.cpp)

add_llvm_executable(txvf ${globbed})

llvm_update_compile_flags(txvf)
target_link_libraries(txvf PRIVATE
  ${dialect_libs}
  CIRCTHandshake
  )
