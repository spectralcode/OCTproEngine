#ifndef OPE_BACKEND_INTERFACE_H
#define OPE_BACKEND_INTERFACE_H

#include <functional>
#include <vector>
#include "../../include/processorconfiguration.h"
#include "../../include/iobuffer.h"

namespace ope {

// Abstract interface that all backends must implement
class ProcessingBackend {
public:
	virtual ~ProcessingBackend() = default;
	
	// Lifecycle
	virtual void initialize(const ProcessorConfiguration& config) = 0;
	virtual void cleanup() = 0;
	
	virtual void setOutputCallback(std::function<void(const IOBuffer&)> callback) = 0;
	
	// Main processing pipeline
	// Backend manages output buffers internally and calls callback when done
	virtual void process(IOBuffer& input) = 0;
	
	// Configuration updates
	virtual void updateConfig(const ProcessorConfiguration& config) = 0;
	
	// Hot-swap curve updates
	virtual void updateResamplingCurve(const float* curve, size_t length) = 0; //todo: think about if there should be a distinction between "update" and "set" (maybe "update" to indicate hot-swappable data),	maybe rename. use "set" instead of "update"
	virtual void updateDispersionCurve(const float* curve, size_t length) = 0;
	virtual void updateWindowCurve(const float* curve, size_t length) = 0;
	
	// Buffer management - for thread-safe, high-throughput acquisition
	virtual IOBuffer& getInputBuffer(int index) = 0;
	virtual IOBuffer& getNextAvailableInputBuffer() = 0;
	virtual int getNumInputBuffers() const = 0;
	virtual int getOutputBufferCount() const = 0;

	// Output buffer release. called when all consumers are done with a buffer
	virtual void releaseOutputBuffer(IOBuffer* buffer) = 0;
	
	// Post-process background management
	virtual void requestPostProcessBackgroundRecording() = 0;
	virtual void setPostProcessBackgroundProfile(const float* background, size_t length) = 0; // todo consitent naming. either "set" or "update"
	virtual const std::vector<float>& getPostProcessBackgroundProfile() const = 0;

	// Fixed-pattern noise management (profile is interleaved floats: real0, imag0, real1, imag1...)
	virtual void requestFixedPatternNoiseDetermination() = 0;
	virtual void setFixedPatternNoiseProfile(const float* profileInterleaved, size_t complexPairs) = 0;
	virtual const std::vector<float>& getFixedPatternNoiseProfile() const = 0;

};

} // namespace ope

#endif // OPE_BACKEND_INTERFACE_H