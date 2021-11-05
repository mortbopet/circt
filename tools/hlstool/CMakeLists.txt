set(LLVM_LINK_COMPONENTS
  Support
  )

add_llvm_tool(hlstool
 hlstool.cpp
)
llvm_update_compile_flags(hlstool)
target_link_libraries(hlstool PRIVATE
  CIRCTExportVerilog
  CIRCTImportFIRFile
  CIRCTFIRRTLToHW
  CIRCTFIRRTLTransforms
  CIRCTSVTransforms
  CIRCTCalyx
  CIRCTSCFToCalyx
  CIRCTESI
  CIRCTHandshake
  CIRCTHandshakeToFIRRTL
  CIRCTStandardToHandshake

  MLIRParser
  MLIRSupport
  MLIRIR
  MLIROptLib
  MLIRStandard
  MLIRTransforms
  MLIRTranslation
  )
