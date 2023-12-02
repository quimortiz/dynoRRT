#include "dynoRRT/dynorrt_macros.h"
#include "dynoRRT/toml_extra_macros.h"
#include <iostream>
#include <toml.hpp>

namespace foo {
struct Foo {
  std::string s = "default";
  double d = -1;
  int i = -1;

  void print();
};
} // namespace foo

TOML11_DEFINE_CONVERSION_NON_INTRUSIVE_OR(foo::Foo, s, d, i)

void foo::Foo::print() {
  toml::value v = *this;
  std::cout << v << std::endl;
}

int main() {
  // ```toml
  // title = "an example toml file"
  // nums  = [3, 1, 4, 1, 5]
  // ```
  auto data = toml::parse("../data/example.toml");
  auto f = toml::find<foo::Foo>(data, "foo");

  std::cout << "parsed f is " << std::endl;
  f.print();

  // find a value with the specified type from a table
  std::string title = toml::find<std::string>(data, "title");

  // convert the whole array into any container automatically
  std::vector<int> nums = toml::find<std::vector<int>>(data, "nums");

  // access with STL-like manner
  if (!data.contains("foo")) {
    data["foo"] = "bar";
  }

  // pass a fallback
  std::string name = toml::find_or<std::string>(data, "name", "not found");

  // width-dependent formatting
  std::cout << std::setw(80) << data << std::endl;

  return 0;
}
