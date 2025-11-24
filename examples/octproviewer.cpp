// OCTproViewer - Interactive GUI for OCTproEngine, mainly for manual testing and visual verification of processed OCT data.
// Build with: cmake -DBUILD_EXAMPLES=ON -DBUILD_OCT_VIEWER=ON

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>

#include "processor.h"
#include "processorconfiguration.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <mutex>
#include <atomic>
#include <limits>
#include <cmath>

// ============================================================================
// Data Structures
// ============================================================================

struct ProcessingParams {
	bool resampling = false;
	int interpolationMethod = 0;  // 0=LINEAR, 1=CUBIC, 2=LANCZOS
	float resamplingCoeffs[4] = {0.0f, 2048.0f, -100.0f, 50.0f};

	bool windowing = false;
	int windowType = 0;
	float windowCenter = 0.5f;
	float windowFillFactor = 0.95f;

	bool dispersionComp = false;
	float dispersionCoeffs[4] = {0.0f, -36.0f, 12.0f, 24.0f};

	bool logScaling = true;
	float grayscaleMin = 30.0f;
	float grayscaleMax = 100.0f;

	bool bscanFlip = false;
	bool sinusoidalScanCorrection = false;

	bool dcBackgroundRemoval = false;
	int dcBackgroundWindowSize = 64;

	bool postProcessBgSubtraction = false;
	float postProcessBgWeight = 1.0f;
	float postProcessBgOffset = 0.0f;

	bool fpnRemoval = false;
	int fpnBscanCount = 1;
};

struct DataParams {
	int samplesPerAscan = 2048;
	int ascansPerBscan = 1024;
	int bscansPerBuffer = 2;
	int dataTypeBits = 16;  // 8, 12, or 16
	ope::Backend backend = ope::Backend::CUDA;

	bool hasChanged = false;

	ope::DataType getDataType() const {
		return dataTypeBits == 8 ? ope::DataType::UINT8 : ope::DataType::UINT16;
	}
};

struct DisplayState {
	std::atomic<bool> newDataAvailable{false};
	std::mutex dataMutex;
	std::vector<float> displayData;
	std::vector<float> allBscansData;
	int displayWidth = 0;
	int displayHeight = 0;
	int currentBscanIndex = 0;

	GLuint textureID = 0;
	int textureWidth = 0;
	int textureHeight = 0;

	float zoom = 1.0f;
	float imagePanX = 0.0f;
	float imagePanY = 0.0f;
};

struct AppState {
	ope::Processor* processor = nullptr;
	bool processorInitialized = false;

	DataParams dataParams;
	ProcessingParams procParams;
	DisplayState display;

	char filePathBuffer[512] = "";
	bool autoUpdate = true;

	std::vector<uint8_t> rawDataCache;
	bool hasDataLoaded = false;
};

// ============================================================================
// Utility Functions
// ============================================================================

std::vector<unsigned char> rotateImage90CCW(const unsigned char* input, int width, int height) {
	std::vector<unsigned char> output(width * height);
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			output[x * height + (height - 1 - y)] = input[y * width + x];
		}
	}
	return output;
}

void onProcessedData(const ope::IOBuffer& output, AppState* state) {
	std::lock_guard<std::mutex> lock(state->display.dataMutex);

	size_t floatCount = output.getSizeInBytes() / sizeof(float);
	state->display.allBscansData.resize(floatCount);
	std::memcpy(state->display.allBscansData.data(), output.getDataPointer(), output.getSizeInBytes());

	state->display.displayWidth = state->dataParams.samplesPerAscan / 2;
	state->display.displayHeight = state->dataParams.ascansPerBscan;

	size_t bscanSize = state->display.displayWidth * state->display.displayHeight;
	size_t bscanOffset = state->display.currentBscanIndex * bscanSize;

	state->display.displayData.resize(bscanSize);
	if (bscanOffset + bscanSize <= state->display.allBscansData.size()) {
		std::memcpy(state->display.displayData.data(),
					state->display.allBscansData.data() + bscanOffset,
					bscanSize * sizeof(float));
	}

	state->display.newDataAvailable = true;
}

// ============================================================================
// Processor Management
// ============================================================================

void applyProcessingParams(ope::Processor* proc, const ProcessingParams& params) {
	proc->enableResampling(params.resampling);
	proc->setInterpolationMethod(static_cast<ope::InterpolationMethod>(params.interpolationMethod));
	proc->setResamplingCoefficients(params.resamplingCoeffs);

	proc->enableWindowing(params.windowing);
	proc->setWindowParameters(
		static_cast<ope::WindowType>(params.windowType),
		params.windowCenter,
		params.windowFillFactor
	);

	proc->enableDispersionCompensation(params.dispersionComp);
	proc->setDispersionCoefficients(params.dispersionCoeffs);

	proc->enableLogScaling(params.logScaling);
	proc->setGrayscaleRange(params.grayscaleMin, params.grayscaleMax);
	proc->setSignalMultiplicatorAndAddend(1.0f, 0.0f);

	proc->enableBscanFlip(params.bscanFlip);
	proc->enableSinusoidalScanCorrection(params.sinusoidalScanCorrection);

	proc->enableBackgroundRemoval(params.dcBackgroundRemoval);
	proc->setBackgroundRemovalWindowSize(params.dcBackgroundWindowSize);

	proc->enablePostProcessBackgroundSubtraction(params.postProcessBgSubtraction);
	proc->setPostProcessBackgroundWeight(params.postProcessBgWeight);
	proc->setPostProcessBackgroundOffset(params.postProcessBgOffset);

	proc->enableFixedPatternNoiseRemoval(params.fpnRemoval);
	proc->setFixedPatternNoiseBscanCount(params.fpnBscanCount);
}

void initializeProcessor(AppState* state) {
	if (state->processor) {
		state->processor->cleanup();
		delete state->processor;
	}

	state->processor = new ope::Processor(state->dataParams.backend);

	state->processor->setInputParameters(
		state->dataParams.samplesPerAscan,
		state->dataParams.ascansPerBscan,
		state->dataParams.bscansPerBuffer,
		state->dataParams.getDataType()
	);

	applyProcessingParams(state->processor, state->procParams);

	state->processor->setOutputCallback(
		[state](const ope::IOBuffer& output) { onProcessedData(output, state); }
	);

	state->processor->initialize();
	state->processorInitialized = true;

	std::cout << "Processor initialized ("
			  << (state->dataParams.backend == ope::Backend::CUDA ? "CUDA" :
			      (state->dataParams.backend == ope::Backend::CPU ? "CPU" : "OpenCL"))
			  << ")" << std::endl;
}

void reprocessData(AppState* state) {
	if (!state->processorInitialized || !state->hasDataLoaded) return;

	applyProcessingParams(state->processor, state->procParams);

	ope::IOBuffer& buffer = state->processor->getNextAvailableInputBuffer();
	std::memcpy(buffer.getDataPointer(), state->rawDataCache.data(),
				std::min(buffer.getSizeInBytes(), state->rawDataCache.size()));

	state->processor->process(buffer);
}

// ============================================================================
// Data Generation and Loading
// ============================================================================

void generateTestData(AppState* state) {
	if (!state->processorInitialized) return;

	const auto& dp = state->dataParams;

	// Lambda for fringe pattern generation
	auto generateFringe = [&dp](int ascan, int sample) {
		const double PI = 3.14159265358979323846;
		double k = sample / double(dp.samplesPerAscan);
		double f = 0.15 * std::cos(2*PI*50*k) * std::exp(-50*k);
		f += 0.12 * std::cos(2*PI*80*k + ascan*0.01) * std::exp(-10*std::abs(k-0.3));
		f += 0.10 * std::cos(2*PI*110*k) * std::exp(-10*std::abs(k-0.4));
		f += 0.09 * std::cos(2*PI*140*k + ascan*0.02) * std::exp(-10*std::abs(k-0.5));
		f += 0.38 * std::cos(2*PI*120*k) * std::exp(-10*std::abs(k-0.6));
		f += 0.36 * std::cos(2*PI*170*k) * std::exp(-10*std::abs(k-0.7));
		f += 0.50 * std::cos(2*PI*200*k + ascan*0.15) * std::exp(-10*std::abs(k-0.75));
		f += 1.10 * std::cos(2*PI*0.1*k*ascan) * std::exp(-10*std::abs(k-0.75));
		f += 0.01 * ((std::rand() % 1000) - 500) / 500.0;
		return f;
	};

	double mid = (dp.dataTypeBits == 8) ? 128.0 : (dp.dataTypeBits == 12) ? 2048.0 : 32768.0;
	double scale = mid * 0.6;
	double maxVal = mid * 2 - 1;

	ope::IOBuffer& buffer = state->processor->getNextAvailableInputBuffer();
	void* ptr = buffer.getDataPointer();

	for (int b = 0; b < dp.bscansPerBuffer; ++b) {
		for (int a = 0; a < dp.ascansPerBscan; ++a) {
			for (int s = 0; s < dp.samplesPerAscan; ++s) {
				size_t idx = b * dp.ascansPerBscan * dp.samplesPerAscan +
					        a * dp.samplesPerAscan + s;
				double val = mid + generateFringe(a, s) * scale;
				val = val < 0 ? 0 : (val > maxVal ? maxVal : val);

				if (dp.dataTypeBits == 8)
					static_cast<uint8_t*>(ptr)[idx] = uint8_t(val);
				else
					static_cast<uint16_t*>(ptr)[idx] = uint16_t(val);
			}
		}
	}

	state->rawDataCache.resize(buffer.getSizeInBytes());
	std::memcpy(state->rawDataCache.data(), buffer.getDataPointer(), buffer.getSizeInBytes());
	state->hasDataLoaded = true;

	state->processor->process(buffer);
	std::cout << "Generated test data (" << buffer.getSizeInBytes() << " bytes)" << std::endl;
}

void loadFileData(AppState* state) {
	if (!state->processorInitialized) return;

	std::ifstream file(state->filePathBuffer, std::ios::binary);
	if (!file.is_open()) {
		std::cerr << "Failed to open file: " << state->filePathBuffer << std::endl;
		return;
	}

	ope::IOBuffer& buffer = state->processor->getNextAvailableInputBuffer();
	file.read(reinterpret_cast<char*>(buffer.getDataPointer()), buffer.getSizeInBytes());
	file.close();

	state->rawDataCache.resize(buffer.getSizeInBytes());
	std::memcpy(state->rawDataCache.data(), buffer.getDataPointer(), buffer.getSizeInBytes());
	state->hasDataLoaded = true;

	state->processor->process(buffer);
	std::cout << "Loaded " << buffer.getSizeInBytes() << " bytes from file" << std::endl;
}

// ============================================================================
// Display Functions
// ============================================================================

void updateTexture(AppState* state) {
	if (!state->display.newDataAvailable) return;

	std::lock_guard<std::mutex> lock(state->display.dataMutex);
	if (state->display.displayData.empty()) return;

	// Convert float data to grayscale
	std::vector<unsigned char> grayscaleData(state->display.displayData.size());
	for (size_t i = 0; i < state->display.displayData.size(); ++i) {
		float val = state->display.displayData[i];
		if (!std::isfinite(val)) val = 0.0f;
		val = val < 0.0f ? 0.0f : (val > 1.0f ? 1.0f : val);
		grayscaleData[i] = static_cast<unsigned char>(val * 255.0f);
	}

	// Rotate 90 degrees CCW
	auto rotatedData = rotateImage90CCW(grayscaleData.data(),
					                    state->display.displayWidth,
					                    state->display.displayHeight);

	state->display.textureWidth = state->display.displayHeight;
	state->display.textureHeight = state->display.displayWidth;

	// Update OpenGL texture
	if (state->display.textureID == 0) {
		glGenTextures(1, &state->display.textureID);
		glBindTexture(GL_TEXTURE_2D, state->display.textureID);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, 0x812F);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, 0x812F);
	} else {
		glBindTexture(GL_TEXTURE_2D, state->display.textureID);
	}

	// Convert to RGB
	std::vector<unsigned char> rgbData(rotatedData.size() * 3);
	for (size_t i = 0; i < rotatedData.size(); ++i) {
		rgbData[i * 3] = rgbData[i * 3 + 1] = rgbData[i * 3 + 2] = rotatedData[i];
	}

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,
				 state->display.textureWidth, state->display.textureHeight,
				 0, GL_RGB, GL_UNSIGNED_BYTE, rgbData.data());

	state->display.newDataAvailable = false;
}

void switchBscan(AppState* state, int index) {
	std::lock_guard<std::mutex> lock(state->display.dataMutex);

	if (state->display.allBscansData.empty() ||
		index < 0 || index >= state->dataParams.bscansPerBuffer) return;

	state->display.currentBscanIndex = index;

	size_t bscanSize = state->display.displayWidth * state->display.displayHeight;
	size_t bscanOffset = index * bscanSize;

	state->display.displayData.resize(bscanSize);
	if (bscanOffset + bscanSize <= state->display.allBscansData.size()) {
		std::memcpy(state->display.displayData.data(),
					state->display.allBscansData.data() + bscanOffset,
					bscanSize * sizeof(float));
		state->display.newDataAvailable = true;
	}
}

// ============================================================================
// UI Helper Functions
// ============================================================================

bool SliderFloatWithReprocess(const char* label, float* v, float min, float max,
					          AppState* state, const char* format = "%.2f") {
	if (ImGui::SliderFloat(label, v, min, max, format)) {
		if (state->autoUpdate) reprocessData(state);
		return true;
	}
	return false;
}

bool CheckboxWithReprocess(const char* label, bool* v, AppState* state) {
	if (ImGui::Checkbox(label, v)) {
		if (state->autoUpdate) reprocessData(state);
		return true;
	}
	return false;
}

bool InputFloatWithReprocess(const char* label, float* v, AppState* state,
					         float step = 1.0f, const char* format = "%.6f") {
	if (ImGui::InputFloat(label, v, step, step, format)) {
		if (state->autoUpdate) reprocessData(state);
		return true;
	}
	return false;
}

// ============================================================================
// UI Rendering
// ============================================================================

void renderDataParametersUI(AppState* state) {
	auto& dp = state->dataParams;

	ImGui::SeparatorText("Data Parameters");

	dp.hasChanged |= ImGui::InputInt("Samples/A-scan", &dp.samplesPerAscan, 1, 1);
	dp.hasChanged |= ImGui::InputInt("A-scans/B-scan", &dp.ascansPerBscan, 1, 1);
	dp.hasChanged |= ImGui::InputInt("B-scans/Buffer", &dp.bscansPerBuffer, 1, 1);

	const char* bitDepths[] = {"8-bit", "12-bit", "16-bit"};
	int bitIdx = dp.dataTypeBits == 8 ? 0 : (dp.dataTypeBits == 12 ? 1 : 2);
	if (ImGui::Combo("Bit Depth", &bitIdx, bitDepths, 3)) {
		dp.dataTypeBits = bitIdx == 0 ? 8 : (bitIdx == 1 ? 12 : 16);
		dp.hasChanged = true;
	}

	const char* backends[] = {"CUDA", "CPU", "OpenCL"};
	int backendIdx = dp.backend == ope::Backend::CUDA ? 0 : (dp.backend == ope::Backend::CPU ? 1 : 2);
	if (ImGui::Combo("Backend", &backendIdx, backends, 3)) {
		dp.backend = backendIdx == 0 ? ope::Backend::CUDA : (backendIdx == 1 ? ope::Backend::CPU : ope::Backend::OPENCL);
		dp.hasChanged = true;
	}

	if (dp.hasChanged) {
		initializeProcessor(state);
		dp.hasChanged = false;
		if (state->hasDataLoaded) reprocessData(state);
	}
}

void renderDataLoadingUI(AppState* state) {
	ImGui::SeparatorText("Load Data");

	static int dataSource = 0;
	ImGui::Combo("Data Source", &dataSource, "Generate Test Data\0Load Custom File\0");

	if (dataSource == 0) {
		bool highlightButton = !state->hasDataLoaded;
		if (highlightButton) {
			ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.6f, 0.2f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.7f, 0.3f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.15f, 0.5f, 0.15f, 1.0f));
		}

		if (ImGui::Button("Generate & Process Test Data", ImVec2(-1, 0))) {
			generateTestData(state);
		}

		if (highlightButton) {
			ImGui::PopStyleColor(3);
		}

		if (!state->hasDataLoaded) {
			ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.2f, 1.0f), "Click here to generate test data and see the viewer in action!");
		}
	} else {
		ImGui::InputText("File Path", state->filePathBuffer, sizeof(state->filePathBuffer));
		if (ImGui::Button("Load & Process Custom File", ImVec2(-1, 0))) {
			loadFileData(state);
		}
	}

	if (state->dataParams.bscansPerBuffer > 1) {
		ImGui::Spacing();
		ImGui::Text("B-scan Navigation:");
		if (ImGui::SliderInt("B-scan Index", &state->display.currentBscanIndex,
					        0, state->dataParams.bscansPerBuffer - 1)) {
			switchBscan(state, state->display.currentBscanIndex);
		}
	}
}

void renderProcessingUI(AppState* state) {
	auto& pp = state->procParams;

	ImGui::SeparatorText("Display");
	CheckboxWithReprocess("Log Scaling", &pp.logScaling, state);
	SliderFloatWithReprocess("Grayscale Min", &pp.grayscaleMin, -50.0f, 200.0f, state);
	SliderFloatWithReprocess("Grayscale Max", &pp.grayscaleMax, -50.0f, 200.0f, state);

	ImGui::SeparatorText("Resampling");
	CheckboxWithReprocess("Enable Resampling", &pp.resampling, state);
	const char* interp[] = {"Linear", "Cubic", "Lanczos"};
	if (ImGui::Combo("Interpolation", &pp.interpolationMethod, interp, 3)) {
		if (state->autoUpdate) reprocessData(state);
	}
	ImGui::Text("Resampling Coefficients (c0 + c1*k + c2*k^2 + c3*k^3):");
	for (int i = 0; i < 4; ++i) {
		char label[8];
		snprintf(label, sizeof(label), "c%d", i);
		InputFloatWithReprocess(label, &pp.resamplingCoeffs[i], state);
	}

	ImGui::SeparatorText("Windowing");
	CheckboxWithReprocess("Enable Windowing", &pp.windowing, state);
	const char* windows[] = {"Hann", "Gauss", "Sine", "Lanczos", "Rect", "Flat Top"};
	if (ImGui::Combo("Window Type", &pp.windowType, windows, 6)) {
		if (state->autoUpdate) reprocessData(state);
	}
	SliderFloatWithReprocess("Window Center", &pp.windowCenter, 0.0f, 1.0f, state, "%.3f");
	SliderFloatWithReprocess("Window Fill", &pp.windowFillFactor, 0.0f, 1.0f, state, "%.3f");

	ImGui::SeparatorText("Dispersion Compensation");
	CheckboxWithReprocess("Enable Dispersion", &pp.dispersionComp, state);
	ImGui::Text("Dispersion Coefficients (d0 + d1*k + d2*k^2 + d3*k^3):");
	for (int i = 0; i < 4; ++i) {
		char label[8];
		snprintf(label, sizeof(label), "d%d", i);
		InputFloatWithReprocess(label, &pp.dispersionCoeffs[i], state);
	}

	ImGui::SeparatorText("DC Background Removal");
	CheckboxWithReprocess("Enable DC Removal", &pp.dcBackgroundRemoval, state);
	if (ImGui::InputInt("DC Window Size", &pp.dcBackgroundWindowSize, 1, 1)) {
		if (state->autoUpdate) reprocessData(state);
	}

	// Post-Process Background Subtraction
	ImGui::SeparatorText("Post-Process Background Subtraction");
	CheckboxWithReprocess("Enable BG Subtraction", &pp.postProcessBgSubtraction, state);
	if (ImGui::Button("Record Background Profile", ImVec2(-1, 0))) {
		if (state->processorInitialized && state->hasDataLoaded) {
			state->processor->requestPostProcessBackgroundRecording();
			reprocessData(state);
		}
	}
	SliderFloatWithReprocess("BG Weight", &pp.postProcessBgWeight, 0.0f, 2.0f, state);
	SliderFloatWithReprocess("BG Offset", &pp.postProcessBgOffset, -100.0f, 100.0f, state);

	// Fixed Pattern Noise
	ImGui::SeparatorText("Fixed-Pattern Noise Removal");
	CheckboxWithReprocess("Enable FPN Removal", &pp.fpnRemoval, state);
	if (ImGui::InputInt("FPN B-scan Count", &pp.fpnBscanCount, 1, 1)) {
		if (state->autoUpdate) reprocessData(state);
	}
	if (ImGui::Button("Determine Fixed Pattern Noise", ImVec2(-1, 0))) {
		if (state->processorInitialized && state->hasDataLoaded) {
			state->processor->requestFixedPatternNoiseDetermination();
			reprocessData(state);
		}
	}

	ImGui::SeparatorText("Misc");
	CheckboxWithReprocess("B-scan Flip", &pp.bscanFlip, state);
	CheckboxWithReprocess("Sinusoidal Scan Correction", &pp.sinusoidalScanCorrection, state);
}

void renderImageDisplay(AppState* state) {
	if (state->display.textureID == 0) {
		ImGui::Text("No image data. Load a file to display.");
		return;
	}

	ImGui::BeginChild("ImageScrollRegion", ImVec2(0, -30), true);

	ImVec2 availSize = ImGui::GetContentRegionAvail();
	ImVec2 imageSize(state->display.textureWidth * state->display.zoom,
					 state->display.textureHeight * state->display.zoom);

	ImVec2 canvasSize(
		imageSize.x > availSize.x ? imageSize.x : availSize.x,
		imageSize.y > availSize.y ? imageSize.y : availSize.y
	);

	ImGui::InvisibleButton("canvas", canvasSize);
	bool isHovered = ImGui::IsItemHovered();
	bool isActive = ImGui::IsItemActive();

	ImVec2 canvasTopLeft = ImGui::GetItemRectMin();
	ImVec2 imagePos = canvasTopLeft;

	if (imageSize.x < availSize.x) imagePos.x += (availSize.x - imageSize.x) * 0.5f;
	if (imageSize.y < availSize.y) imagePos.y += (availSize.y - imageSize.y) * 0.5f;
	imagePos.x += state->display.imagePanX;
	imagePos.y += state->display.imagePanY;

	ImGui::GetWindowDrawList()->AddImage(
		(ImTextureID)(intptr_t)state->display.textureID,
		imagePos,
		ImVec2(imagePos.x + imageSize.x, imagePos.y + imageSize.y),
		ImVec2(0, 0), ImVec2(1, 1)
	);

	// Mouse controls
	if (isHovered) {
		float wheel = ImGui::GetIO().MouseWheel;
		if (wheel != 0.0f) {
			ImVec2 mousePos = ImGui::GetMousePos();
			ImVec2 mousePosInImage(mousePos.x - imagePos.x, mousePos.y - imagePos.y);

			float oldZoom = state->display.zoom;
			state->display.zoom += wheel * 0.1f;
			state->display.zoom = state->display.zoom < 0.1f ? 0.1f :
					             (state->display.zoom > 10.0f ? 10.0f : state->display.zoom);

			float zoomRatio = state->display.zoom / oldZoom;
			state->display.imagePanX += mousePosInImage.x * (1.0f - zoomRatio);
			state->display.imagePanY += mousePosInImage.y * (1.0f - zoomRatio);
		}

		if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
			float zoomX = availSize.x / state->display.textureWidth;
			float zoomY = availSize.y / state->display.textureHeight;
			state->display.zoom = (zoomX < zoomY) ? zoomX : zoomY;
			state->display.imagePanX = 0.0f;
			state->display.imagePanY = 0.0f;
		}
	}

	if (isActive && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
		ImVec2 delta = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left, 0.0f);
		state->display.imagePanX += delta.x;
		state->display.imagePanY += delta.y;
		ImGui::ResetMouseDragDelta(ImGuiMouseButton_Left);
	}

	ImGui::EndChild();

	ImGui::Separator();
	ImGui::Text("Size: %d x %d | Zoom: %.1fx | Controls: scroll/drag/double-click",
				state->display.textureWidth, state->display.textureHeight, state->display.zoom);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
	// GLFW setup
	glfwSetErrorCallback([](int error, const char* desc) {
		std::cerr << "GLFW Error " << error << ": " << desc << std::endl;
	});

	if (!glfwInit()) return -1;

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	GLFWwindow* window = glfwCreateWindow(1600, 900, "OCTproViewer", nullptr, nullptr);
	if (!window) {
		glfwTerminate();
		return -1;
	}

	glfwMakeContextCurrent(window);
	glfwSwapInterval(1);

	// ImGui setup
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGui::StyleColorsDark();
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init("#version 330");

	// Application state
	AppState state;
	if (argc > 1) {
		strncpy(state.filePathBuffer, argv[1], sizeof(state.filePathBuffer) - 1);
	}

	// Initialize processor on first run
	if (!state.processorInitialized) {
		initializeProcessor(&state);
	}

	// Set initial window positions and sizes
	static bool firstFrame = true;

	// Main loop
	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();
		updateTexture(&state);

		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		// Control Panel
		if (firstFrame) {
			ImGui::SetNextWindowPos(ImVec2(10, 10));
			ImGui::SetNextWindowSize(ImVec2(400, 850));
		}
		ImGui::Begin("OCT Controls", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

		renderDataParametersUI(&state);
		renderDataLoadingUI(&state);

		ImGui::Separator();
		ImGui::Checkbox("Auto Update on Changes", &state.autoUpdate);
		if (ImGui::Button("Reprocess Current Data", ImVec2(-1, 0))) {
			reprocessData(&state);
		}

		renderProcessingUI(&state);

		ImGui::End();

		// Image Display
		if (firstFrame) {
			ImGui::SetNextWindowPos(ImVec2(420, 10));
			ImGui::SetNextWindowSize(ImVec2(1160, 850));
			firstFrame = false;
		}
		ImGui::Begin("OCT Image");
		renderImageDisplay(&state);
		ImGui::End();

		// Render
		ImGui::Render();
		int display_w, display_h;
		glfwGetFramebufferSize(window, &display_w, &display_h);
		glViewport(0, 0, display_w, display_h);
		glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		glfwSwapBuffers(window);
	}

	// Cleanup
	if (state.display.textureID != 0) {
		glDeleteTextures(1, &state.display.textureID);
	}
	if (state.processor) {
		state.processor->cleanup();
		delete state.processor;
	}

	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	glfwDestroyWindow(window);
	glfwTerminate();

	return 0;
}