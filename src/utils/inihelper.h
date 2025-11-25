#ifndef OPE_INIHELPER_H
#define OPE_INIHELPER_H

#include <string>
#include <map>
#include <fstream>
#include <sstream>

namespace ope {

// Simple INI file helper for saving and loading parameters to disk
// Keys are stored as "Section.Key" (e.g., "Data.signalLength")
class IniHelper {
public:
	using IniMap = std::map<std::string, std::string>;

	static bool saveToFile(const std::string& filepath, const IniMap& data) {
		std::ofstream file(filepath);
		if (!file.is_open()) {
			return false;
		}

		std::string currentSection;
		for (const auto& pair : data) {
			size_t dotPos = pair.first.find('.');
			if (dotPos == std::string::npos) {
				continue;
			}

			std::string section = pair.first.substr(0, dotPos);
			std::string key = pair.first.substr(dotPos + 1);

			if (section != currentSection) {
				if (!currentSection.empty()) {
					file << "\n";
				}
				file << "[" << section << "]\n";
				currentSection = section;
			}

			file << key << "=" << pair.second << "\n";
		}

		return file.good();
	}

	static bool loadFromFile(const std::string& filepath, IniMap& data) {
		std::ifstream file(filepath);
		if (!file.is_open()) {
			return false;
		}

		std::string line;
		std::string currentSection;

		while (std::getline(file, line)) {
			// Skip empty lines and comments
			if (line.empty() || line[0] == '#' || line[0] == ';') {
				continue;
			}

			// Section header
			if (line[0] == '[') {
				size_t end = line.find(']');
				if (end != std::string::npos) {
					currentSection = line.substr(1, end - 1);
				}
				continue;
			}

			// Key=value pair
			size_t eqPos = line.find('=');
			if (eqPos == std::string::npos) {
				continue;
			}

			std::string key = line.substr(0, eqPos);
			std::string value = line.substr(eqPos + 1);

			// Trim whitespace
			while (!key.empty() && (key.back() == ' ' || key.back() == '\t')) key.pop_back();
			while (!value.empty() && (value[0] == ' ' || value[0] == '\t')) value.erase(0, 1);

			if (!currentSection.empty()) {
				data[currentSection + "." + key] = value;
			}
		}

		return true;
	}

	static void field(IniMap& m, const std::string& key, int& val, bool saving) {
		if (saving) {
			m[key] = std::to_string(val);
		} else {
			auto it = m.find(key);
			if (it != m.end()) val = std::stoi(it->second);
		}
	}

	static void field(IniMap& m, const std::string& key, float& val, bool saving) {
		if (saving) {
			m[key] = std::to_string(val);
		} else {
			auto it = m.find(key);
			if (it != m.end()) val = std::stof(it->second);
		}
	}

	static void field(IniMap& m, const std::string& key, bool& val, bool saving) {
		if (saving) {
			m[key] = val ? "1" : "0";
		} else {
			auto it = m.find(key);
			if (it != m.end()) val = (std::stoi(it->second) != 0);
		}
	}

	template<typename E>
	static void fieldEnum(IniMap& m, const std::string& key, E& val, bool saving) {
		if (saving) {
			m[key] = std::to_string(static_cast<int>(val));
		} else {
			auto it = m.find(key);
			if (it != m.end()) val = static_cast<E>(std::stoi(it->second));
		}
	}
};

} // namespace ope

#endif // OPE_INIHELPER_H
