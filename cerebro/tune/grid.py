# Copyright 2020 Supun Nakandala, Yuhao Zhang, and Arun Kumar. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import itertools

import numpy as np

from .base import ModelSelection, is_larger_better, ModelSelectionResult, _HP, _HPChoice, update_model_results


class GridSearch(ModelSelection):
    """Performs grid search using the given param grid

    :param backend: Cerebro backend object (e.g., SparkBackend).
    :param store: Cerebro store object (e.g., LocalStore, HDFSStore).
    :param estimator_gen_fn: A function which takes
     in a dictionary of parameters and returns a Cerebro Estimator (e.g., cerebro.SparkEstimator).
    :param search_space: A dictionary object defining the parameter search space.
    :param num_epochs: Number of maximum epochs each model should be trained for.
    :param evaluation_metric: Evaluation metric used to pick the best model (default "loss").
    :param validation: (Optional) The ratio of the validation set (default: 0.25) or a string defining the column name
     defining the validation set. In the latter case the column value can be bool or int.
    :param label_columns: (Optional) A list containing the names of the label/output columns (default ['label']).
    :param feature_columns: (Optional) A list containing the names of the feature columns (default ['features']).
    :param verbose: Debug output verbosity (0-2). Defaults to 1.

    :return: :class:`cerebro.tune.ModelSelectionResult`
    """

    def __init__(self, backend, store, estimator_gen_fn, search_space, num_epochs,
                 evaluation_metric='loss', validation=0.25, label_columns=['label'], feature_columns=['features'],
                 verbose=1):
        super(GridSearch, self).__init__(backend, store, validation, estimator_gen_fn, evaluation_metric,
                                         label_columns, feature_columns, verbose)

        self.search_space = search_space
        # validate the search space
        self._validate_search_space()

        self.estimator_param_maps = self._generate_all_param_maps()
        self.num_epochs = num_epochs

    def _validate_search_space(self):
        search_space = self.search_space
        if not type(search_space) == dict:
            raise Exception('Search space has to be type dict. Provided: {}'.format(type(search_space)))

        if not all([isinstance(k, str) for k in search_space.keys()]):
            raise Exception('Only string values are allowed for hyperparameter space keys.')

        if not all([isinstance(k, _HPChoice) for k in search_space.values()]):
            raise Exception('All hyperparameter space values has to be of type cerebro.tune.base._HPChoice.'
                            ' Nested search spaces are not supported yet')

    def _generate_all_param_maps(self):
        keys = self.search_space.keys()
        grid_values = [v.options for v in self.search_space.values()]

        def _to_key_value_pairs(keys, values):
            # values = [v if isinstance(v, list) else v() for v in values]
            return [(key, value) for key, value in zip(keys, values)]

        return [dict(_to_key_value_pairs(keys, prod)) for prod in itertools.product(*[v if isinstance(v, list) else \
                                                                                          v() for v in grid_values])]

    def _fit_on_prepared_data(self, dataset_idx, metadata):
        return _fit_on_prepared_data(self, dataset_idx, metadata)


class RandomSearch(ModelSelection):
    """ Performs Random Search over the param grid

    :param backend: Cerebro backend object (e.g., SparkBackend).
    :param store: Cerebro store object (e.g., LocalStore, HDFSStore).
    :param estimator_gen_fn: A function which takes
     in a dictionary of parameters and returns a Cerebro Estimator (e.g., cerebro.SparkEstimator).
    :param search_space: A dictionary object defining the parameter search space.
    :param num_epochs: Number of maximum epochs each model should be trained for.
    :param evaluation_metric: Evaluation metric used to pick the best model (default: "loss").
    :param validation: (Optional) The ratio of the validation set (default: 0.25) or a string defining the column name
     defining the validation set. In the latter case the column value can be bool or int.
    :param label_columns: (Optional) A list containing the names of the label/output columns (default ['label']).
    :param feature_columns: (Optional) A list containing the names of the feature columns (default ['features']).
    :param verbose: Debug output verbosity (0-2). Defaults to 1.

    :return: :class:`cerebro.tune.ModelSelectionResult`
    """

    def __init__(self, backend, store, estimator_gen_fn, search_space, num_models, num_epochs, evaluation_metric='loss',
                 validation=0.25, label_columns=['label'], feature_columns=['features'], verbose=1):
        super(RandomSearch, self).__init__(backend, store, validation, estimator_gen_fn, evaluation_metric,
                                           label_columns, feature_columns, verbose)

        self.search_space = search_space
        # validate the search space
        self._validate_search_space()

        self.num_params = num_models
        self.estimator_param_maps = self._generate_all_param_maps()
        self.num_epochs = num_epochs

    def _validate_search_space(self):
        search_space = self.search_space
        if not type(search_space) == dict:
            raise Exception('Search space has to be type dict. Provided: {}'.format(type(search_space)))

        if not all([isinstance(k, str) for k in search_space.keys()]):
            raise Exception('Only string values are allowed for hyperparameter space keys.')

        if not all([isinstance(k, _HP) for k in search_space.values()]):
            raise Exception('All hyperparameter space values has to be of type cerebro.tune.base._HP.'
                            ' Nested search spaces are not supported yet')

    def _generate_all_param_maps(self):
        params = []
        keys = self.search_space.keys()
        for _ in range(self.num_params):
            param_dict = {}
            for k in keys:
                param_dict[k] = self.search_space[k].sample_value()
            params.append(param_dict)
        return params

    def _fit_on_prepared_data(self, dataset_idx, metadata):
        return _fit_on_prepared_data(self, dataset_idx, metadata)


def _fit_on_prepared_data(self, dataset_idx, metadata):
    # create estimators
    estimators = [self._estimator_gen_fn_wrapper(param) for param in self.estimator_param_maps]
    estimator_results = {model.getRunId(): {} for model in estimators}

    # log hyperparameters to TensorBoard
    self._log_hp_to_tensorboard(estimators, self.estimator_param_maps)

    # Trains the models up to the number of epochs specified. For each iteration also performs validation
    for epoch in range(self.num_epochs):
        epoch_results = self.backend.train_for_one_epoch(estimators, self.store, dataset_idx, self.feature_cols,
                                                         self.label_cols)
        update_model_results(estimator_results, epoch_results)

        epoch_results = self.backend.train_for_one_epoch(estimators, self.store, dataset_idx, self.feature_cols,
                                                         self.label_cols, is_train=False)
        update_model_results(estimator_results, epoch_results)

        self._log_epoch_metrics_to_tensorboard(estimators, estimator_results)

    # find the best model and crate ModelSearchModel
    models = [est.create_model(estimator_results[est.getRunId()], est.getRunId(), metadata) for est in estimators]
    val_metrics = [estimator_results[est.getRunId()]['val_' + self.evaluation_metric][-1] for est in estimators]
    best_model_idx = np.argmax(val_metrics) if is_larger_better(self.evaluation_metric) else np.argmin(val_metrics)
    best_model = models[best_model_idx]

    return ModelSelectionResult(best_model, estimator_results, models, [x+'__output' for x in self.label_cols])
