#include <toml.hpp>

// namespace dynorrt {

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

// }
