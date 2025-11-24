#ifndef OPE_PROCESSORTOOL_H
#define OPE_PROCESSORTOOL_H

#include <memory>
#include "processor.h"
#include "export.h"

namespace ope {

/**
 * @brief Base class for tools that use raw (input) and/or processed (output) OCT data from a Processor
 *
 * ProcessorTool provides a common interface for tools that need access to
 * raw and/or processed OCT data. Tools can attach to a processor to receive
 * callbacks, or work standalone if they don't require a processor.
 *
 */
class OPE_API ProcessorTool {
public:
	ProcessorTool() = default;
	virtual ~ProcessorTool();

	/**
	 * @brief Attach this tool to a processor
	 *
	 * Registers callbacks with the processor to receive data.
	 * Calls configureCallbacks() to set up specific callbacks needed.
	 *
	 * @param processor The processor to attach to (can be nullptr to detach)
	 */
	void attachToProcessor(Processor* processor);

	/**
	 * @brief Detach from the current processor
	 *
	 * Removes all callbacks and clears the processor reference.
	 */
	void detach();

	/**
	 * @brief Check if this tool is attached to a processor
	 * @return true if attached, false otherwise
	 */
	bool isAttached() const;

	/**
	 * @brief Get the attached processor
	 * @return Pointer to attached processor, or nullptr if not attached
	 */
	Processor* getProcessor() const;


protected:
	/**
	 * @brief Configure callbacks when attaching to a processor
	 *
	 * Override this to register specific callbacks needed by the tool.
	 * Called automatically by attachToProcessor().
	 *
	 * Example:
	 * @code
	 * void configureCallbacks() override {
	 *     rawCallbackId = processor->addInputCallback(
	 *         [this](const IOBuffer& buf) { onInputData(buf); }
	 *     );
	 * }
	 * @endcode
	 */
	virtual void configureCallbacks() {}

	/**
	 * @brief Clean up callbacks when detaching from a processor
	 *
	 * Override this if you need custom cleanup beyond removing callbacks.
	 * Default implementation removes all registered callbacks.
	 * Called automatically by detach().
	 */
	virtual void cleanupCallbacks();

	// Protected members accessible to derived classes
	Processor* processor = nullptr;
	Processor::CallbackId rawCallbackId = -1;
	Processor::CallbackId processedCallbackId = -1;
};

} // namespace ope

#endif // OPE_PROCESSORTOOL_H