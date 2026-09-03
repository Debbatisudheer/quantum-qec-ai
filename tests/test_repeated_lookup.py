from decoders.repeated_lookup import (
    RepeatedLookupDecoder
)


def main():

    print()
    print("===================================")
    print(" REPEATED LOOKUP DECODER TEST")
    print("===================================")

    decoder = RepeatedLookupDecoder()

    test_cases = {
        "00": [0, 0, 0],
        "10": [1, 0, 0],
        "11": [0, 1, 0],
        "01": [0, 0, 1],
    }

    all_passed = True

    for syndrome, expected in test_cases.items():

        predicted = decoder.decode(
            syndrome
        )

        passed = predicted == expected

        print(
            f"Syndrome {syndrome} -> "
            f"{predicted} : "
            f"{'PASS' if passed else 'FAIL'}"
        )

        if not passed:
            all_passed = False

    history = [
        "00",
        "10",
        "10",
        "01",
        "01",
    ]

    predicted = decoder.decode_history(
        history
    )

    expected = [0, 0, 1]

    passed = predicted == expected

    print()
    print(
        "History final decode : "
        f"{'PASS' if passed else 'FAIL'}"
    )

    if not passed:
        all_passed = False

    print()
    print(
        "==================================="
    )

    if all_passed:
        print(
            "REPEATED LOOKUP DECODER TEST : SUCCESS"
        )
    else:
        print(
            "REPEATED LOOKUP DECODER TEST : FAILED"
        )

    print(
        "==================================="
    )


if __name__ == "__main__":
    main()