#include "text_normalizer.hpp"

#include <memory>
#include <stdexcept>

#include <unicode/normalizer2.h>
#include <unicode/translit.h>
#include <unicode/unistr.h>

namespace phono::core {

std::string normalize_text_utf8(const std::string& text_utf8) {
    if (text_utf8.empty()) {
        return {};
    }

    UErrorCode status = U_ZERO_ERROR;
    icu::UnicodeString normalized = icu::UnicodeString::fromUTF8(text_utf8);
    std::unique_ptr<icu::Transliterator> simplified(
        icu::Transliterator::createInstance("Traditional-Simplified", UTRANS_FORWARD, status));
    if (U_FAILURE(status) || simplified == nullptr) {
        throw std::runtime_error(
            "normalize_text_utf8: cannot initialize ICU Traditional-Simplified transliterator");
    }
    simplified->transliterate(normalized);

    status = U_ZERO_ERROR;
    const icu::Normalizer2* nfkc = icu::Normalizer2::getNFKCInstance(status);
    if (U_FAILURE(status)) {
        throw std::runtime_error("normalize_text_utf8: cannot initialize ICU NFKC");
    }
    icu::UnicodeString normalized_nfkc;
    nfkc->normalize(normalized, normalized_nfkc, status);
    if (U_FAILURE(status)) {
        throw std::runtime_error("normalize_text_utf8: ICU NFKC normalization failed");
    }

    std::string result;
    normalized_nfkc.toUTF8String(result);
    return result;
}

}  // namespace phono::core
