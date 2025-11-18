#include "../../include/processortool.h"
#include <stdexcept>

namespace ope {

ProcessorTool::~ProcessorTool() {
	detach();
}

void ProcessorTool::attachToProcessor(Processor* newProcessor) {
	if (this->processor && this->processor != newProcessor) {
		detach();
	}

	this->processor = newProcessor;

	if (this->processor) {
		configureCallbacks();
	}
}

void ProcessorTool::detach() {
	if (this->processor) {
		cleanupCallbacks();
		this->processor = nullptr;
	}
}

bool ProcessorTool::isAttached() const {
	return this->processor != nullptr;
}

Processor* ProcessorTool::getProcessor() const {
	return this->processor;
}

void ProcessorTool::cleanupCallbacks() {
	if (this->processor) {
		if (this->rawCallbackId >= 0) {
			this->processor->removeInputCallback(this->rawCallbackId);
			this->rawCallbackId = -1;
		}
		if (this->processedCallbackId >= 0) {
			this->processor->removeOutputCallback(this->processedCallbackId);
			this->processedCallbackId = -1;
		}
	}
}

} // namespace ope