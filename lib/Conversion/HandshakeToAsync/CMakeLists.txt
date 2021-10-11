add_circt_library(CIRCTHandshakeToAsync
  HandshakeToAsync.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/HandshakeToAsync

  LINK_LIBS PUBLIC
  CIRCTHandshake
  MLIRAsync
  MLIRIR
  MLIRPass
  MLIRStandard
  MLIRSupport
  MLIRTransforms
  )
