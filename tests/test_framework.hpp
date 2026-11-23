#pragma once

#include <functional>
#include <string>
#include <vector>

namespace qwen3_asr::test {

struct TestCase {
    std::string name;
    std::function<void()> function;
};

void register_test(const char * name, std::function<void()> function);
void check(bool condition, const char * expression, const char * file, int line);
int run_all_tests(const char * filter = nullptr);

}  // namespace qwen3_asr::test

#define QWEN3ASR_TEST_CASE(unique_name)                                      \
    static void unique_name();                                               \
    static const bool unique_name##_registered = []() {                      \
        ::qwen3_asr::test::register_test(#unique_name, &unique_name);        \
        return true;                                                         \
    }();                                                                     \
    static void unique_name()

#define CHECK(expression)                                                    \
    ::qwen3_asr::test::check(                                                \
        static_cast<bool>(expression), #expression, __FILE__, __LINE__)

#define REQUIRE(expression)                                                  \
    do {                                                                     \
        if (!(expression)) {                                                 \
            ::qwen3_asr::test::check(false, #expression, __FILE__, __LINE__);\
            return;                                                          \
        }                                                                    \
    } while (false)
