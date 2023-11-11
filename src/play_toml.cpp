#include <iostream>
#include <toml.hpp>

// TOML11_DEFINE_CONVERSION_NON_INTRUSIVE

// ----------------------------------------------------------------------------
// TOML11_ARGS_SIZE
#define TOML11_FIND_MEMBER_VARIABLE_FROM_VALUE_OR(VAR_NAME)                    \
  obj.VAR_NAME = toml::find_or<decltype(obj.VAR_NAME)>(                        \
      v, TOML11_STRINGIZE(VAR_NAME), decltype(obj.VAR_NAME)(obj.VAR_NAME));

#define TOML11_DEFINE_CONVERSION_NON_INTRUSIVE_OR(NAME, ...)                   \
  namespace toml {                                                             \
  template <> struct from<NAME> {                                              \
    template <typename C, template <typename...> class T,                      \
              template <typename...> class A>                                  \
    static NAME from_toml(const basic_value<C, T, A> &v) {                     \
      NAME obj;                                                                \
      TOML11_FOR_EACH_VA_ARGS(TOML11_FIND_MEMBER_VARIABLE_FROM_VALUE_OR,       \
                              __VA_ARGS__)                                     \
      return obj;                                                              \
    }                                                                          \
  };                                                                           \
  template <> struct into<NAME> {                                              \
    static value into_toml(const NAME &obj) {                                  \
      ::toml::value v = ::toml::table{};                                       \
      TOML11_FOR_EACH_VA_ARGS(TOML11_ASSIGN_MEMBER_VARIABLE_TO_VALUE,          \
                              __VA_ARGS__)                                     \
      return v;                                                                \
    }                                                                          \
  };                                                                           \
  } /* toml */

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
