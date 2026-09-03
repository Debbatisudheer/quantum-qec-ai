from correction.time_varying_correction import (
    TimeVaryingCorrectionEngine
)

from evaluation.logical_recovery import (
    LogicalRecovery
)


def test_time_varying_correction():

    print("\n===================================")
    print(" TIME-VARYING CORRECTION TEST")
    print("===================================")

    correction = TimeVaryingCorrectionEngine()
    recovery = LogicalRecovery()

    # --------------------------------
    # TEST 1
    # --------------------------------

    actual_error = [1, 0, 1]
    predicted_error = [1, 0, 1]

    result = correction.correct_sample(
        actual_error,
        predicted_error
    )

    print("\nTEST 1 — PERFECT AI PREDICTION")

    print(
        f"Actual error      : "
        f"{actual_error}"
    )

    print(
        f"Predicted error   : "
        f"{predicted_error}"
    )

    print(
        f"Corrected state   : "
        f"{result['corrected_state']}"
    )

    print(
        f"Correction        : "
        f"{result['correction_description']}"
    )

    assert result["corrected_state"] == [
        0, 0, 0
    ]

    assert result["physically_correct"] is True

    print("Physical correction: PASS")

    # --------------------------------
    # TEST 2
    # --------------------------------

    actual_error = [1, 1, 0]
    predicted_error = [0, 1, 0]

    result = correction.correct_sample(
        actual_error,
        predicted_error
    )

    print("\nTEST 2 — IMPERFECT AI PREDICTION")

    print(
        f"Actual error      : "
        f"{actual_error}"
    )

    print(
        f"Predicted error   : "
        f"{predicted_error}"
    )

    print(
        f"Corrected state   : "
        f"{result['corrected_state']}"
    )

    assert result["corrected_state"] == [
        1, 0, 0
    ]

    assert result["physically_correct"] is False

    print("Physical correction failure detected: PASS")

    # --------------------------------
    # TEST 3
    # --------------------------------
    #
    # Logical 0
    # Physical state [1,0,0]
    #
    # Majority = 0
    #
    # Therefore logical information survives.
    # --------------------------------

    logical_result = recovery.recover_sample(
        original_logical_state=0,
        corrected_state=[1, 0, 0]
    )

    print("\nTEST 3 — LOGICAL RECOVERY")

    print(
        "Original logical state : "
        f"{logical_result['original_logical_state']}"
    )

    print(
        "Recovered logical state: "
        f"{logical_result['recovered_logical_state']}"
    )

    print(
        "Logical success        : "
        f"{logical_result['logical_success']}"
    )

    assert (
        logical_result["recovered_logical_state"]
        == 0
    )

    assert (
        logical_result["logical_success"]
        is True
    )

    print("Logical recovery: PASS")

    # --------------------------------
    # TEST 4
    # --------------------------------
    #
    # Logical 0
    # Physical state [1,1,0]
    #
    # Majority = 1
    #
    # Therefore logical information is lost.
    # --------------------------------

    logical_result = recovery.recover_sample(
        original_logical_state=0,
        corrected_state=[1, 1, 0]
    )

    print("\nTEST 4 — LOGICAL FAILURE")

    print(
        "Original logical state : "
        f"{logical_result['original_logical_state']}"
    )

    print(
        "Recovered logical state: "
        f"{logical_result['recovered_logical_state']}"
    )

    print(
        "Logical success        : "
        f"{logical_result['logical_success']}"
    )

    assert (
        logical_result["recovered_logical_state"]
        == 1
    )

    assert (
        logical_result["logical_success"]
        is False
    )

    print("Logical failure detection: PASS")

    # --------------------------------
    # FINAL
    # --------------------------------

    print("\n===================================")
    print(" TIME-VARYING CORRECTION RESULT")
    print("===================================")

    print(
        "Correction engine : PASS"
    )

    print(
        "Physical correction : PASS"
    )

    print(
        "Logical recovery : PASS"
    )

    print(
        "Logical failure detection : PASS"
    )

    print(
        "RESULT : SUCCESS"
    )


if __name__ == "__main__":
    test_time_varying_correction()