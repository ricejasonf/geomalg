#include <geomalg/Dialect.h>
#include <geomalg/Passes.h>
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include <string>

int main(int argc, char ** argv) {
  mlir::DialectRegistry DialectRegistry;
  DialectRegistry.insert<geomalg::GeomalgDialect>();
  DialectRegistry.insert<mlir::func::FuncDialect>();

  geomalg::registerGeomalgPasses();

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "geomalg optimizer driver\n", DialectRegistry));
}
