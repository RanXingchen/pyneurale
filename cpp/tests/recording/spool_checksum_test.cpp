/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

/// Checks the two digests before anything checks a spool.
///
/// Specification section 3 publishes CRC-32C known answers for exactly this
/// purpose: an implementation that gets the parameterization wrong would
/// otherwise reject every conforming spool it met and produce spools no other
/// implementation could read, with no evidence pointing at the cause.

#include "check_returns.h"
#include "crc32c.h"
#include "sha256.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace
{

using neurale::recording::crc32c;
using neurale::recording::Sha256;
using neurale::recording::sha256;

[[nodiscard]] std::vector<std::byte> as_bytes(std::string_view text)
{
    std::vector<std::byte> out;
    for (const auto character : text)
    {
        out.push_back(static_cast<std::byte>(static_cast<unsigned char>(character)));
    }
    return out;
}

[[nodiscard]] std::string hex(const std::array<std::uint8_t, 32>& digest)
{
    static constexpr char kDigits[] = "0123456789abcdef";
    std::string out;
    for (const auto value : digest)
    {
        out.push_back(kDigits[value >> 4]);
        out.push_back(kDigits[value & 0x0F]);
    }
    return out;
}

int test_crc32c_known_answers()
{
    CHECK(crc32c({}) == 0x00000000U);
    CHECK(crc32c(as_bytes("123456789")) == 0xE3069283U);

    const std::vector<std::byte> zeros(32, std::byte{0});
    CHECK(crc32c(zeros) == 0x8A9136AAU);
    const std::vector<std::byte> ones(32, std::byte{0xFF});
    CHECK(crc32c(ones) == 0x62A8AB43U);
    return 0;
}

int test_crc32c_streams_in_chunks()
{
    // The scanner computes a transaction body checksum over bytes it reads in
    // pieces, so chunked seeding must equal the one-shot value exactly.
    const auto data = as_bytes("the quick brown fox jumps over the lazy dog");
    const auto whole = crc32c(data);
    for (std::size_t split = 0; split <= data.size(); ++split)
    {
        const std::vector<std::byte> head(data.begin(),
                                          data.begin() + static_cast<std::ptrdiff_t>(split));
        const std::vector<std::byte> tail(data.begin() + static_cast<std::ptrdiff_t>(split),
                                          data.end());
        CHECK(crc32c(tail, crc32c(head)) == whole);
    }
    return 0;
}

int test_sha256_known_answers()
{
    CHECK(hex(sha256({})) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    CHECK(hex(sha256(as_bytes("abc"))) ==
          "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    CHECK(hex(sha256(as_bytes("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"))) ==
          "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");

    const std::vector<std::byte> million(1000000, static_cast<std::byte>('a'));
    CHECK(hex(sha256(million)) ==
          "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
    return 0;
}

int test_sha256_streams_in_chunks()
{
    const auto data = as_bytes("the quick brown fox jumps over the lazy dog");
    const auto whole = sha256(data);
    for (std::size_t split = 0; split <= data.size(); ++split)
    {
        Sha256 hasher;
        hasher.update({data.data(), split});
        hasher.update({data.data() + split, data.size() - split});
        CHECK(hasher.finish() == whole);
    }

    // Longer than one block, and split across a block boundary, because the
    // plan document a reader verifies is streamed through a bounded buffer.
    const std::vector<std::byte> long_input(1000, static_cast<std::byte>('a'));
    const auto long_whole = sha256(long_input);
    for (const std::size_t split : {std::size_t{1}, std::size_t{63}, std::size_t{64},
                                    std::size_t{65}, std::size_t{128}, std::size_t{999}})
    {
        Sha256 hasher;
        hasher.update({long_input.data(), split});
        hasher.update({long_input.data() + split, long_input.size() - split});
        CHECK(hasher.finish() == long_whole);
    }
    return 0;
}

} // namespace

int main()
{
    for (const auto test : {
             test_crc32c_known_answers,
             test_crc32c_streams_in_chunks,
             test_sha256_known_answers,
             test_sha256_streams_in_chunks,
         })
    {
        if (const auto line = test(); line != 0)
        {
            return line;
        }
    }
    return 0;
}
