#ifndef TEST_BACKEND_H
#define TEST_BACKEND_H

#include "backendconfig.h"
#include <iostream>
#include <string>

inline int selectTestBackend(int argc, char** argv, ope::Backend& backend) {
	struct Choice {
		const char* name;
		ope::Backend backend;
		bool (*available)();
	};
	const Choice choices[] = {
		{"CPU", ope::Backend::CPU, ope::BackendUtils::isCpuAvailable},
		{"CUDA", ope::Backend::CUDA, ope::BackendUtils::isCudaAvailable},
		{"OPENCL", ope::Backend::OPENCL, ope::BackendUtils::isOpenCLAvailable},
		{"VULKAN", ope::Backend::VULKAN, ope::BackendUtils::isVulkanAvailable}
	};
	if (argc == 2) {
		for (const auto& choice : choices) {
			if (argv[1] != std::string(choice.name)) continue;
			backend = choice.backend;
			if (!choice.available()) {
				std::cout << "SKIP: " << choice.name << " unavailable" << std::endl;
				return 77;
			}
			std::cout << "Backend: " << choice.name << std::endl;
			return 0;
		}
	}
	std::cerr << "Usage: " << argv[0] << " CPU|CUDA|OPENCL|VULKAN" << std::endl;
	return 1;
}

#endif // TEST_BACKEND_H
