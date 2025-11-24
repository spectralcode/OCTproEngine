#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include <stdexcept>
#include <string>

// Simple test assertion macro that works in both Debug and Release builds
// Unlike assert(), this will always execute the condition and throw on failure
#define TEST_ASSERT(condition, message) \
	do { \
		if (!(condition)) { \
			throw std::runtime_error(std::string("Test failed: ") + message + \
				" at " + __FILE__ + ":" + std::to_string(__LINE__)); \
		} \
	} while(0)

#endif // TEST_UTILS_H