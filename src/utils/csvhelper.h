#ifndef OPE_CSVHELPER_H
#define OPE_CSVHELPER_H

#include <string>
#include <vector>
#include <fstream>
#include <sstream>

namespace ope {

// Simple CSV file helper for saving and loading float arrays
class CSVHelper {
public:
	// Save single-column float data
	static bool save(const std::string& filepath,
					const std::vector<float>& data,
					const std::string& title = "") {
		std::ofstream file(filepath);
		if (!file.is_open()) return false;

		if (!title.empty()) {
			file << "# " << title << "\n";
		}
		file << "# OCTproEngine\n";
		file << "# Size: " << data.size() << "\n";
		file << "index,value\n";

		for (size_t i = 0; i < data.size(); ++i) {
			file << i << "," << data[i] << "\n";
		}
		return file.good();
	}

	// Save complex data (two columns: real, imaginary)
	static bool saveComplex(const std::string& filepath,
						   const std::vector<float>& data,
						   const std::string& title = "") {
		if (data.size() % 2 != 0) return false;  // Must be pairs

		std::ofstream file(filepath);
		if (!file.is_open()) return false;

		if (!title.empty()) {
			file << "# " << title << "\n";
		}
		file << "# OCTproEngine\n";
		file << "# Complex pairs: " << data.size() / 2 << "\n";
		file << "index,real,imaginary\n";

		for (size_t i = 0; i < data.size() / 2; ++i) {
			file << i << "," << data[i*2] << "," << data[i*2+1] << "\n";
		}
		return file.good();
	}

	// Load any CSV - auto-detects columns
	static std::vector<float> load(const std::string& filepath) {
		std::vector<float> result;
		std::ifstream file(filepath);
		if (!file.is_open()) return result;

		std::string line;
		bool isComplex = false;

		// Detect format from header
		while (std::getline(file, line)) {
			if (line.find("index,real,imaginary") != std::string::npos) {
				isComplex = true;
				break;
			} else if (line.find("index,value") != std::string::npos) {
				isComplex = false;
				break;
			}
		}

		// Read data
		while (std::getline(file, line)) {
			if (line.empty() || line[0] == '#') continue;

			std::istringstream iss(line);
			std::string index;
			std::getline(iss, index, ',');  // Skip index

			if (isComplex) {
				std::string real, imag;
				if (std::getline(iss, real, ',') && std::getline(iss, imag)) {
					result.push_back(std::stof(real));
					result.push_back(std::stof(imag));
				}
			} else {
				std::string value;
				if (std::getline(iss, value)) {
					result.push_back(std::stof(value));
				}
			}
		}

		return result;
	}
};

} // namespace ope

#endif // OPE_CSVHELPER_H