import pandas as pd
import numpy as np

from pacsynth import (
    DpAggregateSeededSynthesizer,
    DpAggregateSeededParametersBuilder,
    AccuracyMode,
    FabricationMode,
)
from pacsynth import Dataset as AggregateSeededDataset
from snsynth.base import Synthesizer


"""
Wrapper for Private Aggregate Seeded Synthesizer from pac-synth:
https://pypi.org/project/pac-synth/.

A differentially-private synthesizer that relies on DP Marginals to
build synthetic data. It will compute DP Marginals (called aggregates)
for your dataset up to and including a specified reporting length, and
synthesize data based on the computed aggregated counts.

For documentation please refer to:
    - https://github.com/microsoft/synthetic-data-showcase
    - https://github.com/microsoft/synthetic-data-showcase/tree/main/docs/dp
"""


class AggregateSeededSynthesizer(Synthesizer):
    """
    SmartNoise class wrapper for Private Aggregate Seeded Synthesizer from pac-synth.
    Works with Pandas data frames, raw data and follows norms set by other SmartNoise synthesizers.

    :param reporting_length: The maximum length of the combinations to be synthesized.  For example,
        if reporting length is 2, the synthesizer will compute DP marginals for all two-column combinations
        in the dataset.
    :type reporting_length: int
    :param epsilon: The privacy budget to be used for the synthesizer.
    :type epsilon: float
    :param delta: The delta value to be used for the synthesizer.  If set, should be small, in the range
        of 1/(n * sqrt(n)), where n is the approximate number of records in the dataset.
    :param percentile_percentage: Because the synthesizer computes multiple n-way marginals, each individual may
        affect multiple marginals.  The ``percentile_percentage`` can remove the influence of outliers to
        reduce sensitivity and improve the accuracy of the synthesizer.  For example, if ``percentile_percentage``
        is 99, the synthesizer will use a sensitivity that can accomodate 99% of the individuals, and will ensure
        that the records of the outlier 1% are sampled to conform to this sensitivity.
    :type percentile_percentage: int
    :param percentile_epsilon_proportion: The proportion of the epsilon budget to be used to estimate the
        percentile sensitivity.
    :type percentile_epsilon_proportion: float
    :param verbose: Show diagnostic information about the synthesizer's progress.
    :type verbose: bool

    See the `pac-synth documentation <https://github.com/microsoft/synthetic-data-showcase/blob/main/docs/dp/README.md>`_ 
        for more details about these and other hyperparameters.

    Reuses code and modifies it lightly from 
        `pac-synth <https://github.com/microsoft/synthetic-data-showcase/tree/main/packages/lib-pacsynth>`_.
    """

    def __init__(
        self,
        reporting_length=3,
        epsilon=4.0,
        delta=None,
        percentile_percentage=99,
        percentile_epsilon_proportion=0.01,
        accuracy_mode=AccuracyMode.prioritize_long_combinations(),
        number_of_records_epsilon_proportion=0.005,
        fabrication_mode=FabricationMode.uncontrolled(),
        empty_value="",
        use_synthetic_counts=False,
        weight_selection_percentile=95,
        aggregate_counts_scale_factor=None,
        verbose=False
    ):
        """
        Wrapper for Private Aggregate Seeded Synthesizer from pac-synth.

        For more information about the parameters run `help('pacsynth.DpAggregateSeededParametersBuilder')`.
        """
        self.epsilon = epsilon
        self.delta = delta
        self.reporting_length = reporting_length
        self.percentile_percentage = percentile_percentage
        self.percentile_epsilon_proportion = percentile_epsilon_proportion
        self.accuracy_mode = accuracy_mode
        self.number_of_records_epsilon_proportion = number_of_records_epsilon_proportion
        self.fabrication_mode = fabrication_mode
        self.empty_value = empty_value
        self.use_synthetic_counts = use_synthetic_counts
        self.weight_selection_percentile = weight_selection_percentile
        self.aggregate_counts_scale_factor = aggregate_counts_scale_factor
        self.verbose = verbose
        self.preprocessed = False
        self.build_synthesizer()

    def build_synthesizer(self):
        builder = (
            DpAggregateSeededParametersBuilder()
            .reporting_length(self.reporting_length)
            .epsilon(self.epsilon)
            .percentile_percentage(self.percentile_percentage)
            .percentile_epsilon_proportion(self.percentile_epsilon_proportion)
            .accuracy_mode(self.accuracy_mode)
            .number_of_records_epsilon_proportion(self.number_of_records_epsilon_proportion)
            .fabrication_mode(self.fabrication_mode)
            .empty_value(self.empty_value)
            .use_synthetic_counts(self.use_synthetic_counts)
            .weight_selection_percentile(self.weight_selection_percentile)
        )

        if self.aggregate_counts_scale_factor is not None:
            builder = builder.aggregate_counts_scale_factor(
                self.aggregate_counts_scale_factor
            )

        if self.delta is not None:
            builder = builder.delta(self.delta)

        self.reporting_length = self.reporting_length
        self.parameters = builder.build()
        self.synth = DpAggregateSeededSynthesizer(self.parameters)
        self.dataset = None
        self.pandas = False

    def fit(
        self, 
        data,
        *ignore,
        use_columns=None,
        sensitive_zeros=None,
        transformer=None,
        categorical_columns=None,
        ordinal_columns=None,
        continuous_columns=None,
        preprocessor_eps=0.0,
        nullable=False
        ):
        """
        Fit the synthesizer model on the data.

        This will compute the differently private aggregates used to
        synthesize data.

        All the columns are supposed to be categorical, non-categorical columns
        should be binned in advance.

        For more information run `help('pacsynth.Dataset')` and
        `help('pacsynth.DpAggregateSeededSynthesizer.fit')`.

        :param data: The data for fitting the synthesizer model.
        :type data: pd.DataFrame, list[list[str]], AggregateSeededDataset
        :param use_columns: List of column names to be used, defaults to None, meaning use all columns
        :type use_columns: list[str], optional
        :param sensitive_zeros: List of column names containing '0' that should not be turned into empty strings.
        :type sensitive_zeros: list[str], optional
        """

        before_eps = self.epsilon

        train_data = self._get_train_data(
            data,
            style='cube',
            transformer=transformer,
            categorical_columns=categorical_columns, 
            ordinal_columns=ordinal_columns, 
            continuous_columns=continuous_columns, 
            nullable=True,
            preprocessor_eps=preprocessor_eps
        )

        if self.epsilon != before_eps:
            # preprocessor changed epsilon, rebuild synthesizer
            self.build_synthesizer()

        if self._transformer is None:
            raise ValueError("We weren't able to fit a transformer to the data. Please check your data and try again.")

        if self._transformer.output_width > 0:
            colnames = ["column_{}".format(i) for i in range(len(train_data[0]))]
            data = [colnames] + [[str(v) for v in row] for row in train_data]

            assert sensitive_zeros is None, "sensitive zeros cannot be set with a transformer, please set transformer=NoTransformer()"
            assert use_columns is None, "use columns cannot be set with a transformer, please set transformer=NoTransformer()"

            sensitive_zeros = colnames

            cards = self._transformer.cardinality
            if any (c is None for c in cards):
                raise ValueError("The transformer appears to have some continuous columns. Please provide only categorical or ordinal.")

            dimensionality = np.prod(cards)
            if self.verbose:
                print(f"Fitting with {dimensionality} dimensions")


        if isinstance(data, list) and all(map(lambda row: isinstance(row, list), data)):
            self.dataset = AggregateSeededDataset(
                data, use_columns=use_columns, sensitive_zeros=sensitive_zeros
            )
            self.pandas = False
        elif isinstance(data, pd.DataFrame):
            self.dataset = AggregateSeededDataset.from_data_frame(
                data, use_columns=use_columns, sensitive_zeros=sensitive_zeros
            )
            self.pandas = True
        elif isinstance(data, AggregateSeededDataset):
            self.dataset = data
            self.pandas = False
        else:
            raise ValueError(
                "data should be either in raw format (List[List[]]) or a be pandas data frame (pd.DataFrame)"
            )

        self.synth.fit(self.dataset)

    def sample(self, samples=None):
        """
        Sample from the synthesizer model.

        This will sample records from the generated differentially private aggregates.

        If `samples` is `None`, the synthesizer will use all the available differentially
        private attributes counts to synthesize records
        (which will produce a number close to original number of records).

        For more information run `help('pacsynth.DpAggregateSeededSynthesizer.sample')`.

        :param samples: The number of samples to create
        :type samples: int, None
        :return: Generated data samples, the output type adjusts accordingly to the input data.
        :rtype: Dataframe, list[list[str]]
        """
        result = self.synth.sample(samples)

        if self._transformer is not None and self._transformer.output_width > 0:
            result = [[int(v) if v != '' else None for v in row] for row in result[1:]]
            result = self._transformer.inverse_transform(result)
            return result

        if self.pandas is True:
            result = AggregateSeededDataset.raw_data_to_data_frame(result)

        return result


    def get_sensitive_aggregates(
        self, combination_delimiter=";", reporting_length=None
        ):
        """
        Returns the aggregates for the sensitive dataset. For more information run `help('pacsynth.Dataset.get_aggregates')`.

        :param combination_delimiter: Combination delimiter to use, default to ';'
        :type combination_delimiter: str, optional
        :param reporting_length: Maximum length (inclusive) to compute attribute combinations for, defaults to the configured value in the synthesizer
        :type reporting_length: int, optional

        :return: A dictionary with the combination string representation as key and the combination count as value
        :rtype: dict[str, int]
        """
        if self.dataset is None:
            raise RuntimeError(
                "make sure 'fit' method has been successfully called first"
            )

        if reporting_length is None:
            reporting_length = self.reporting_length

        return self.dataset.get_aggregates(reporting_length, combination_delimiter)

    def get_dp_aggregates(self, combination_delimiter=";"):
        """
        Returns the aggregates for the sensitive dataset protected with differential privacy.

        For more information run `help('pacsynth.DpAggregateSeededSynthesizer.get_dp_aggregates')`.

        :param combination_delimiter: Combination delimiter to use, default to ';'
        :type combination_delimiter: str, optional
        :return: A dictionary with the combination string representation as key and the combination count as value
        :rtype: dict[str, int]
        """
        return self.synth.get_dp_aggregates(combination_delimiter)

    def get_dp_number_of_records(self):
        """
        Gets the differentially private number of records computed with the `.fit` method.

        This is different than the number of records specified in the sample method, synthesized
        in the synthetic data. This refers to the differentially private protected number of records
        in original sensitive dataset (Laplacian noise added).

        For more information run `help('pacsynth.DpAggregateSeededSynthesizer.get_dp_number_of_records')`.

        :return: Number of sensitive records protect with differential privacy
        :rtype: int
        """
        return self.synth.get_dp_number_of_records()
