#include <TheMendimeter/TheMendimeter.h>
#include <TheMendimeter/version.h>
#include <doctest/doctest.h>

#include <string>

TEST_CASE("TheMendimeter") {
  using namespace TheMendimeter;

  TheMendimeter TheMendimeter("Tests");

  CHECK(TheMendimeter.greet(LanguageCode::EN) == "Hello, Tests!");
  CHECK(TheMendimeter.greet(LanguageCode::DE) == "Hallo Tests!");
  CHECK(TheMendimeter.greet(LanguageCode::ES) == "¡Hola Tests!");
  CHECK(TheMendimeter.greet(LanguageCode::FR) == "Bonjour Tests!");
}

TEST_CASE("TheMendimeter version") {
  static_assert(std::string_view(TheMendimeter_VERSION) == std::string_view("1.0"));
  CHECK(std::string(TheMendimeter_VERSION) == std::string("1.0"));
}