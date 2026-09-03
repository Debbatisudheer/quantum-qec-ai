import csv
import json
from dataclasses import asdict
from typing import List

from dataset.schema import QECSample


class QECDatasetStorage:
    """
    Storage engine for QEC datasets.

    Responsible for:

        QECSample list
              ↓
        Save to CSV / JSON

        CSV / JSON
              ↓
        Load back into QECSample objects
    """

    def save_csv(
        self,
        dataset: List[QECSample],
        filepath: str
    ):
        """
        Save a QEC dataset to a CSV file.

        Args:
            dataset: List of QECSample objects
            filepath: Destination CSV path
        """

        if len(dataset) == 0:
            raise ValueError(
                "Dataset cannot be empty"
            )

        fieldnames = [
            "sample_id",
            "qec_code",
            "num_qubits",
            "logical_state",
            "original_state",
            "corrupted_state",
            "error_type",
            "error_qubit",
            "error_description",
            "syndrome",
            "target",
        ]

        with open(
            filepath,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames
            )

            writer.writeheader()

            for sample in dataset:
                writer.writerow(
                    asdict(sample)
                )

    def load_csv(
        self,
        filepath: str
    ) -> List[QECSample]:
        """
        Load a QEC dataset from a CSV file.

        Args:
            filepath: CSV file path

        Returns:
            List of QECSample objects
        """

        dataset = []

        with open(
            filepath,
            "r",
            newline="",
            encoding="utf-8"
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:

                error_qubit = row[
                    "error_qubit"
                ]

                if error_qubit == "":
                    error_qubit = None
                else:
                    error_qubit = int(
                        error_qubit
                    )

                target = row["target"]

                if target == "":
                    target = None
                else:
                    target = int(target)

                sample = QECSample(
                    sample_id=int(
                        row["sample_id"]
                    ),
                    qec_code=row[
                        "qec_code"
                    ],
                    num_qubits=int(
                        row["num_qubits"]
                    ),
                    logical_state=int(
                        row["logical_state"]
                    ),
                    original_state=row[
                        "original_state"
                    ],
                    corrupted_state=row[
                        "corrupted_state"
                    ],
                    error_type=row[
                        "error_type"
                    ],
                    error_qubit=error_qubit,
                    error_description=row[
                        "error_description"
                    ],
                    syndrome=row[
                        "syndrome"
                    ],
                    target=target,
                )

                dataset.append(sample)

        return dataset

    def save_json(
        self,
        dataset: List[QECSample],
        filepath: str
    ):
        """
        Save a QEC dataset to a JSON file.
        """

        if len(dataset) == 0:
            raise ValueError(
                "Dataset cannot be empty"
            )

        data = [
            asdict(sample)
            for sample in dataset
        ]

        with open(
            filepath,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                data,
                file,
                indent=4
            )

    def load_json(
        self,
        filepath: str
    ) -> List[QECSample]:
        """
        Load a QEC dataset from a JSON file.

        Returns:
            List of QECSample objects
        """

        with open(
            filepath,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        dataset = []

        for item in data:

            sample = QECSample(
                sample_id=item[
                    "sample_id"
                ],
                qec_code=item[
                    "qec_code"
                ],
                num_qubits=item[
                    "num_qubits"
                ],
                logical_state=item[
                    "logical_state"
                ],
                original_state=item[
                    "original_state"
                ],
                corrupted_state=item[
                    "corrupted_state"
                ],
                error_type=item[
                    "error_type"
                ],
                error_qubit=item[
                    "error_qubit"
                ],
                error_description=item[
                    "error_description"
                ],
                syndrome=item[
                    "syndrome"
                ],
                target=item[
                    "target"
                ],
            )

            dataset.append(sample)

        return dataset