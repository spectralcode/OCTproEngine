#include "bindings/bindings_common.h"

// ============================================
// PYBIND11 MODULE DEFINITION
// ============================================

PYBIND11_MODULE(octproengine, m) {
	m.doc() = "OCTproEngine - High-performance OCT processing library";

	// Register all components
	register_exceptions(m);
	register_enums(m);
	register_backend_config(m);
	register_configuration(m);
	register_processor(m);
	register_recorder(m);

	// Module version
	m.attr("__version__") = OPE_VERSION_STRING;
}
