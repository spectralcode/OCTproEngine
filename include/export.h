#ifndef OPE_EXPORT_H
#define OPE_EXPORT_H

#if defined(_WIN32) || defined(_WIN64)
	#ifdef BUILD_OPE_LIBRARY
		#define OPE_API __declspec(dllexport)
	#else
		#define OPE_API __declspec(dllimport)
	#endif
#elif defined(__GNUC__) || defined(__clang__)
	#ifdef BUILD_OPE_LIBRARY
		#define OPE_API __attribute__((visibility("default")))
	#else
		#define OPE_API
	#endif
#else
	#define OPE_API
#endif

#endif // OPE_EXPORT_H