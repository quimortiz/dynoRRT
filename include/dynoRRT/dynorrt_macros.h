#pragma once

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace dynorrt {

class pretty_runtime_exception : public std::runtime_error {
  // Adapted from:
  // https://stackoverflow.com/questions/348833/how-to-know-the-exact-line-of-code-where-an-exception-has-been-caused

public:
  pretty_runtime_exception(const std::string &arg, const char *file, int line,
                           const char *function)
      : std::runtime_error(arg) {
    std::ostringstream o;
    o << "Error in " << function << " (" << file << ":" << line << "): " << arg
      << std::endl;
    msg = o.str();
  }

  ~pretty_runtime_exception() throw() {}
  const char *what() const throw() { return msg.c_str(); }

private:
  std::string msg;
};
} // namespace dynorrt

#define THROW_PRETTY_DYNORRT(arg)                                              \
  throw pretty_runtime_exception(arg, __FILE__, __LINE__, __FUNCTION__);

#define CHECK_PRETTY_DYNORRT(condition, arg)                                   \
  if (!(condition)) {                                                          \
    throw pretty_runtime_exception(arg, __FILE__, __LINE__, __FUNCTION__);     \
  }

#define CHECK_PRETTY_DYNORRT__(condition)                                      \
  if (!(condition)) {                                                          \
    throw pretty_runtime_exception(#condition, __FILE__, __LINE__,             \
                                   __FUNCTION__);                              \
  }

#define MESSAGE_PRETTY_DYNORRT(arg)                                            \
  std::cout << "Message in " << __FUNCTION__ << " (" << __FILE__ << ":"        \
            << __LINE__ << ") --" << arg << std::endl;
