#include "test_framework.hpp"

#include <exception>
#include <iostream>
#include <utility>

namespace qwen3_asr::test {
namespace {

std::vector<TestCase> & tests() {
    static std::vector<TestCase> value;
    return value;
}

int * current_failures() {
    static int value = 0;
    return &value;
}

}  // namespace

void register_test(const char * name, std::function<void()> function) {
    tests().push_back(TestCase{name, std::move(function)});
}

void check(bool condition, const char * expression, const char * file, int line) {
    if (condition) {
        return;
    }

    ++(*current_failures());
    std::cerr << "CHECK failed: " << expression << " at " << file << ':'
              << line << '\n';
}

int run_all_tests(const char * filter) {
    int total_failures = 0;
    int executed = 0;
    const std::string filter_text = filter != nullptr ? filter : "";

    for (const auto & test : tests()) {
        if (!filter_text.empty() &&
            test.name.find(filter_text) == std::string::npos) {
            continue;
        }

        *current_failures() = 0;
        std::cout << "[ RUN      ] " << test.name << std::endl;
        try {
            test.function();
        } catch (const std::exception & error) {
            ++(*current_failures());
            std::cerr << "Unhandled exception: " << error.what() << '\n';
        } catch (...) {
            ++(*current_failures());
            std::cerr << "Unhandled non-standard exception\n";
        }
        ++executed;

        const int failures = *current_failures();
        if (failures == 0) {
            std::cout << "[       OK ] " << test.name << std::endl;
        } else {
            std::cout << "[  FAILED  ] " << test.name << " (" << failures
                      << " check(s))" << std::endl;
        }
        total_failures += failures;
    }

    if (executed == 0) {
        std::cerr << "no tests matched filter: " << filter_text << '\n';
        return 2;
    }

    std::cout << (total_failures == 0 ? "[  PASSED  ] " : "[  FAILED  ] ")
              << executed << " test(s), " << total_failures
              << " failed check(s)" << std::endl;
    return total_failures == 0 ? 0 : 1;
}

}  // namespace qwen3_asr::test

int main(int argc, char ** argv) {
    return qwen3_asr::test::run_all_tests(argc > 1 ? argv[1] : nullptr);
}
