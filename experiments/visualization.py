from typing import Any

from experiments.analysis import ExperimentAnalysis


class ExperimentVisualization:
    """
    Converts experiment analysis results into frontend-ready
    visualization data.
    """

    def __init__(self, analysis: ExperimentAnalysis):
        if not isinstance(analysis, ExperimentAnalysis):
            raise TypeError(
                "analysis must be an ExperimentAnalysis"
            )

        self.analysis = analysis

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------

    @staticmethod
    def _config_value(
        result: dict[str, Any],
        key: str,
        default=None,
    ):
        """
        Read a configuration value from the stored experiment.

        New results store configuration inside result["config"].

        The fallback to the top level keeps compatibility with
        older result formats.
        """

        config = result.get("config")

        if isinstance(config, dict):
            value = config.get(key)

            if value is not None:
                return value

        return result.get(key, default)

    @classmethod
    def _physical_noise(cls, result):
        return cls._config_value(
            result,
            "physical_noise_probability",
        )

    @classmethod
    def _measurement_noise(cls, result):
        return cls._config_value(
            result,
            "measurement_noise_probability",
        )

    @classmethod
    def _rounds(cls, result):
        return cls._config_value(
            result,
            "rounds",
        )

    @classmethod
    def _qec_code(cls, result):
        return cls._config_value(
            result,
            "qec_code",
        )

    # ---------------------------------------------------------
    # Analysis result normalization
    # ---------------------------------------------------------

    @staticmethod
    def _normalize_grouped_data(
        data,
        key_name: str,
    ):
        """
        Convert grouped analysis output into frontend rows.

        Supported input forms:

        1. Dictionary:
           {
               key: value,
               ...
           }

        2. List of dictionaries:
           [
               {
                   key_name: key,
                   "logical_success": value
               },
               ...
           ]

        3. List/tuple pairs:
           [
               (key, value),
               ...
           ]

        The frontend always receives:

        [
            {
                key_name: key,
                "logical_success": value
            }
        ]
        """

        if data is None:
            return []

        # ---------------------------------------------
        # Dictionary output
        # ---------------------------------------------

        if isinstance(data, dict):
            rows = []

            for key, value in data.items():
                rows.append(
                    {
                        key_name: key,
                        "logical_success": value,
                    }
                )

            return rows

        # ---------------------------------------------
        # List / tuple output
        # ---------------------------------------------

        if isinstance(data, (list, tuple)):
            rows = []

            for item in data:

                # Already-normalized dictionary
                if isinstance(item, dict):
                    if (
                        key_name in item
                        and "logical_success" in item
                    ):
                        rows.append(
                            {
                                key_name: item[key_name],
                                "logical_success": item[
                                    "logical_success"
                                ],
                            }
                        )
                        continue

                    # Common alternative naming
                    if (
                        key_name in item
                        and "success" in item
                    ):
                        rows.append(
                            {
                                key_name: item[key_name],
                                "logical_success": item[
                                    "success"
                                ],
                            }
                        )
                        continue

                # Pair: (key, value)
                if isinstance(item, (list, tuple)):
                    if len(item) >= 2:
                        rows.append(
                            {
                                key_name: item[0],
                                "logical_success": item[1],
                            }
                        )

            return rows

        return []

    # ---------------------------------------------------------
    # Chart 1
    # ---------------------------------------------------------

    def logical_success_by_rounds(
        self,
        results=None,
    ):
        """
        Logical success rate grouped by rounds.
        """

        results = self._resolve_results(results)

        analysis = self.analysis.analyze_rounds(
            results
        )

        return self._normalize_grouped_data(
            analysis,
            "rounds",
        )

    # ---------------------------------------------------------
    # Chart 2
    # ---------------------------------------------------------

    def logical_success_by_physical_noise(
        self,
        results=None,
    ):
        """
        Logical success rate grouped by physical noise.
        """

        results = self._resolve_results(results)

        analysis = self.analysis.analyze_physical_noise(
            results
        )

        return self._normalize_grouped_data(
            analysis,
            "physical_noise",
        )

    # ---------------------------------------------------------
    # Chart 3
    # ---------------------------------------------------------

    def logical_success_by_measurement_noise(
        self,
        results=None,
    ):
        """
        Logical success rate grouped by measurement noise.
        """

        results = self._resolve_results(results)

        analysis = (
            self.analysis.analyze_measurement_noise(
                results
            )
        )

        return self._normalize_grouped_data(
            analysis,
            "measurement_noise",
        )

    # ---------------------------------------------------------
    # Chart 4
    # ---------------------------------------------------------

    def decoder_comparison(
        self,
        results=None,
    ):
        """
        Logical success rate grouped by decoder.
        """

        results = self._resolve_results(results)

        analysis = self.analysis.analyze_decoders(
            results
        )

        return self._normalize_grouped_data(
            analysis,
            "decoder",
        )

    # ---------------------------------------------------------
    # Performance table
    # ---------------------------------------------------------

    def performance_comparison(
        self,
        results=None,
    ):
        """
        Build frontend-ready performance rows.

        Configuration values are read from the experiment
        configuration instead of assuming they exist at the
        top level of the stored result.
        """

        results = self._resolve_results(results)

        rows = []

        for result in results:
            rows.append(
                {
                    "experiment_id": result.get(
                        "experiment_id"
                    ),

                    "decoder": result.get(
                        "decoder_type"
                    ),

                    "qec_code": self._qec_code(
                        result
                    ),

                    "rounds": self._rounds(
                        result
                    ),

                    "physical_noise": (
                        self._physical_noise(
                            result
                        )
                    ),

                    "measurement_noise": (
                        self._measurement_noise(
                            result
                        )
                    ),

                    "logical_success": (
                        result.get(
                            "logical_accuracy"
                        )
                    ),

                    "physical_recovery": (
                        result.get(
                            "physical_accuracy"
                        )
                    ),

                    "bit_accuracy": (
                        result.get(
                            "bit_accuracy"
                        )
                    ),

                    "exact_accuracy": (
                        result.get(
                            "exact_accuracy"
                        )
                    ),

                    "training_seconds": (
                        result.get(
                            "training_seconds"
                        )
                    ),

                    "inference_seconds": (
                        result.get(
                            "inference_seconds"
                        )
                    ),

                    "samples_per_second": (
                        result.get(
                            "samples_per_second"
                        )
                    ),
                }
            )

        return rows

    # ---------------------------------------------------------
    # Complete visualization data
    # ---------------------------------------------------------

    def build(
        self,
        results=None,
    ):
        """
        Build all visualization datasets.
        """

        results = self._resolve_results(results)

        if not results:
            return {
                "logical_success_by_rounds": [],
                "logical_success_by_physical_noise": [],
                "logical_success_by_measurement_noise": [],
                "decoder_comparison": [],
                "performance_comparison": [],
            }

        return {
            "logical_success_by_rounds": (
                self.logical_success_by_rounds(
                    results
                )
            ),

            "logical_success_by_physical_noise": (
                self.logical_success_by_physical_noise(
                    results
                )
            ),

            "logical_success_by_measurement_noise": (
                self.logical_success_by_measurement_noise(
                    results
                )
            ),

            "decoder_comparison": (
                self.decoder_comparison(
                    results
                )
            ),

            "performance_comparison": (
                self.performance_comparison(
                    results
                )
            ),
        }

    # ---------------------------------------------------------
    # Frontend format
    # ---------------------------------------------------------

    def frontend_data(
        self,
        results=None,
    ):
        """
        Return the API structure expected by Next.js.
        """

        data = self.build(results)

        return {
            "charts": {
                "rounds": data[
                    "logical_success_by_rounds"
                ],

                "physical_noise": data[
                    "logical_success_by_physical_noise"
                ],

                "measurement_noise": data[
                    "logical_success_by_measurement_noise"
                ],

                "decoders": data[
                    "decoder_comparison"
                ],
            },

            "performance": data[
                "performance_comparison"
            ],
        }

    # ---------------------------------------------------------
    # Resolve results
    # ---------------------------------------------------------

    def _resolve_results(
        self,
        results=None,
    ):
        """
        Use supplied results or load all stored results.
        """

        if results is not None:
            return results

        return self.analysis.query.all_results()