#pragma once
#include <string>

// The metadata parser uses util only to format unsupported-opcode errors.
std::string sd_format(const char* format, ...);
