#ifndef OPE_CSVHELPER_H
#define OPE_CSVHELPER_H

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <locale>
#include <iomanip>
#include <limits>

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

		file.imbue(std::locale::classic());
		file << std::setprecision(std::numeric_limits<float>::max_digits10);
		if (!title.empty()) {
			file << "# " << title << "\n";
		}
		file << "index;value\n";

		for (size_t i = 0; i < data.size(); ++i) {
			file << i << ';' << data[i] << "\n";
		}
		file.close();
		return file.good();
	}

	// Save complex data (two columns: real, imaginary)
	static bool saveComplex(const std::string& filepath,
						   const std::vector<float>& data,
						   const std::string& title = "") {
		if (data.size() % 2 != 0) return false;  // Must be pairs

		std::ofstream file(filepath);
		if (!file.is_open()) return false;

		file.imbue(std::locale::classic());
		file << std::setprecision(std::numeric_limits<float>::max_digits10);
		if (!title.empty()) {
			file << "# " << title << "\n";
		}
		file << "index;real;imaginary\n";

		for (size_t i = 0; i < data.size() / 2; ++i) {
			file << i << ';' << data[i*2] << ';' << data[i*2+1] << "\n";
		}
		file.close();
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
		while (std::getline(file >> std::ws, line)) {
			if (line.empty() || line[0] == '#') continue;
			isComplex = line.find("index;real;imaginary") != std::string::npos ||
				line.find("Sample Number;Real;Imag") != std::string::npos;
			break;
		}

		// Read data
		while (std::getline(file >> std::ws, line)) {
			if (line.empty() || line[0] == '#') continue;

			std::istringstream iss(line);
			std::string index;
			std::getline(iss, index, ';');  // Skip index

			if (isComplex) {
				std::string real, imag;
				float realValue, imagValue;
				if (!std::getline(iss, real, ';') || !std::getline(iss, imag) ||
					!parseNumber(real, realValue) || !parseNumber(imag, imagValue)) return {};
				result.push_back(realValue);
				result.push_back(imagValue);
			} else {
				std::string value;
				float number;
				if (!std::getline(iss, value) || !parseNumber(value, number)) return {};
				result.push_back(number);
			}
		}
		if (file.bad()) return {};
		return result;
	}

private:
	static bool parseNumber(const std::string& field, float& value) {
		std::istringstream stream(field);
		stream.imbue(std::locale::classic());
		return (stream >> value) && (stream >> std::ws).eof() && std::isfinite(value);
	}
};

} // namespace ope

#endif // OPE_CSVHELPER_H
