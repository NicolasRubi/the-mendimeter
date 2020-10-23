#include <TheMendimeter/TheMendimeter.h>

using namespace TheMendimeter;

TheMendimeter::TheMendimeter(std::string _name) : name(_name) {}

std::string TheMendimeter::greet(LanguageCode lang) const {
  switch (lang) {
    default:
    case LanguageCode::EN:
      return "Hello, " + name + "!";
    case LanguageCode::DE:
      return "Hallo " + name + "!";
    case LanguageCode::ES:
      return "¡Hola " + name + "!";
    case LanguageCode::FR:
      return "Bonjour " + name + "!";
  }
}
