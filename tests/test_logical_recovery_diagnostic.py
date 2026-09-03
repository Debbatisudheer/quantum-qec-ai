from evaluation.logical_recovery import LogicalRecovery


ALL_STATES = [
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1],
]


EXPECTED_LOGICAL = {
    (0, 0, 0): 0,
    (0, 0, 1): 0,
    (0, 1, 0): 0,
    (1, 0, 0): 0,

    (0, 1, 1): 1,
    (1, 0, 1): 1,
    (1, 1, 0): 1,
    (1, 1, 1): 1,
}


# ============================================================
# TEST 1
# BASIC RECOVERY
# ============================================================

def test_basic_recovery():

    print()
    print("=" * 60)
    print(" TEST 1: BASIC LOGICAL RECOVERY")
    print("=" * 60)

    recovery = LogicalRecovery()

    passed = True

    for state in ALL_STATES:

        recovered = recovery.recover(state)

        expected = EXPECTED_LOGICAL[
            tuple(state)
        ]

        status = (
            "PASS"
            if recovered == expected
            else "FAIL"
        )

        print(
            f"{state} -> "
            f"recovered={recovered} "
            f"expected={expected} "
            f"{status}"
        )

        if recovered != expected:
            passed = False

    print()

    print(
        "BASIC RECOVERY : "
        + ("PASS" if passed else "FAIL")
    )

    return passed


# ============================================================
# TEST 2
# EXPECTED ENCODED STATE
# ============================================================

def test_expected_state():

    print()
    print("=" * 60)
    print(" TEST 2: EXPECTED ENCODED STATE")
    print("=" * 60)

    recovery = LogicalRecovery()

    passed = True

    expected_mapping = {
        0: [0, 0, 0],
        1: [1, 1, 1],
    }

    for logical_state, expected in expected_mapping.items():

        result = recovery.expected_state(
            logical_state
        )

        status = (
            "PASS"
            if result == expected
            else "FAIL"
        )

        print(
            f"Logical state {logical_state} "
            f"-> result={result} "
            f"expected={expected} "
            f"{status}"
        )

        if result != expected:
            passed = False

    print()

    print(
        "EXPECTED STATE : "
        + ("PASS" if passed else "FAIL")
    )

    return passed


# ============================================================
# TEST 3
# RECOVERY RESULT
#
# We test recover() directly because the
# existing is_logical_success() interface
# should not be assumed.
# ============================================================

def test_recovery_mapping():

    print()
    print("=" * 60)
    print(" TEST 3: RECOVERY MAPPING")
    print("=" * 60)

    recovery = LogicalRecovery()

    passed = True

    for state in ALL_STATES:

        recovered = recovery.recover(state)

        expected = EXPECTED_LOGICAL[
            tuple(state)
        ]

        if recovered == expected:

            status = "PASS"

        else:

            status = "FAIL"
            passed = False

        print(
            f"state={state} "
            f"-> logical={recovered} "
            f"expected={expected} "
            f"{status}"
        )

    print()

    print(
        "RECOVERY MAPPING : "
        + ("PASS" if passed else "FAIL")
    )

    return passed


# ============================================================
# TEST 4
# CORRECTION XOR
# ============================================================

def test_correction_xor():

    print()
    print("=" * 60)
    print(" TEST 4: CORRECTION XOR")
    print("=" * 60)

    passed = True

    for actual in ALL_STATES:

        # Perfect decoder:
        # prediction == actual error.
        predicted = actual.copy()

        corrected = [
            a ^ p
            for a, p in zip(
                actual,
                predicted
            )
        ]

        expected = [0, 0, 0]

        status = (
            "PASS"
            if corrected == expected
            else "FAIL"
        )

        print(
            f"actual={actual} "
            f"predicted={predicted} "
            f"corrected={corrected} "
            f"{status}"
        )

        if corrected != expected:
            passed = False

    print()

    print(
        "CORRECTION XOR : "
        + ("PASS" if passed else "FAIL")
    )

    return passed


# ============================================================
# TEST 5
# ALL ACTUAL/PREDICTED COMBINATIONS
# ============================================================

def test_all_correction_combinations():

    print()
    print("=" * 60)
    print(
        " TEST 5: ALL CORRECTION COMBINATIONS"
    )
    print("=" * 60)

    recovery = LogicalRecovery()

    total = 0
    perfect_corrections = 0
    recovery_failures = 0

    for actual in ALL_STATES:

        for predicted in ALL_STATES:

            total += 1

            corrected = [
                a ^ p
                for a, p in zip(
                    actual,
                    predicted
                )
            ]

            recovered = recovery.recover(
                corrected
            )

            expected_recovery = (
                EXPECTED_LOGICAL[
                    tuple(corrected)
                ]
            )

            if recovered != expected_recovery:

                recovery_failures += 1

            if corrected == [
                0,
                0,
                0
            ]:

                perfect_corrections += 1

    print(
        f"Total combinations        : {total}"
    )

    print(
        f"Perfect physical corrections: "
        f"{perfect_corrections}"
    )

    print(
        f"Recovery mapping failures  : "
        f"{recovery_failures}"
    )

    passed = (
        total == 64
        and perfect_corrections == 8
        and recovery_failures == 0
    )

    print()

    print(
        "ALL COMBINATIONS : "
        + ("PASS" if passed else "FAIL")
    )

    return passed


# ============================================================
# TEST 6
# LOGICAL MAPPING SANITY
# ============================================================

def test_logical_mapping_sanity():

    print()
    print("=" * 60)
    print(
        " TEST 6: LOGICAL MAPPING SANITY"
    )
    print("=" * 60)

    recovery = LogicalRecovery()

    logical_zero_states = [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
    ]

    logical_one_states = [
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ]

    passed = True

    print()
    print("Logical 0 states:")

    for state in logical_zero_states:

        result = recovery.recover(state)

        status = (
            "PASS"
            if result == 0
            else "FAIL"
        )

        print(
            f"{state} -> {result} {status}"
        )

        if result != 0:
            passed = False

    print()
    print("Logical 1 states:")

    for state in logical_one_states:

        result = recovery.recover(state)

        status = (
            "PASS"
            if result == 1
            else "FAIL"
        )

        print(
            f"{state} -> {result} {status}"
        )

        if result != 1:
            passed = False

    print()

    print(
        "LOGICAL MAPPING : "
        + ("PASS" if passed else "FAIL")
    )

    return passed


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print(" LOGICAL RECOVERY DIAGNOSTIC")
    print("=" * 60)

    results = []

    results.append(
        test_basic_recovery()
    )

    results.append(
        test_expected_state()
    )

    results.append(
        test_recovery_mapping()
    )

    results.append(
        test_correction_xor()
    )

    results.append(
        test_all_correction_combinations()
    )

    results.append(
        test_logical_mapping_sanity()
    )

    print()
    print("=" * 60)
    print(" FINAL DIAGNOSTIC RESULT")
    print("=" * 60)

    passed = sum(results)
    total = len(results)

    print(
        f"Tests passed : {passed}/{total}"
    )

    if passed == total:

        print()
        print(
            "RESULT : SUCCESS"
        )

        print()
        print(
            "Logical recovery mapping and "
            "correction XOR logic are consistent."
        )

    else:

        print()
        print(
            "RESULT : FAILURE"
        )

        print()
        print(
            "A logical-recovery assumption "
            "still needs investigation."
        )

    print()
    print("=" * 60)
    print(
        " LOGICAL RECOVERY DIAGNOSTIC : COMPLETE"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()