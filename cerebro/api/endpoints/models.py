# Copyright 2021 Supun Nakandala, and Arun Kumar. All Rights Reserved.
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
import logging

from flask import request
from flask_restplus import Resource
from ..restplus import api
from ..serializers import model
from ..parsers import experiment_id_argument
from ..database.dbo import Model, Experiment, ParamVal
from ..database import db

log = logging.getLogger(__name__)

ns = api.namespace('models', description='Operations related to models')


@ns.route('/')
class ModelsCollection(Resource):
    @api.marshal_list_with(model)
    @api.expect(experiment_id_argument, validate=True)
    def get(self):
        """
        Returns list of models.
        """
        args = experiment_id_argument.parse_args()
        exp_id = args['exp_id']
        return Model.query.filter(Model.exp_id == exp_id).all()

    @api.expect(model)
    def post(self):
        """
        Creates a new model.
        """
        data = request.json
        exp_id = data.get('exp_id')
        num_trained_epochs =  data.get('num_trained_epochs')
        max_train_epochs = data.get('max_train_epochs')
        exp = Experiment.query.filter(Experiment.id == exp_id).one()
        if exp.status in ['failed', 'stopping', 'stopped', 'completed']:
            raise BadRequest('Experiment is in {} staus. Cannot create new models.'.format(exp.status))

        model_dao = Model(exp_id, num_trained_epochs, max_train_epochs)

        for pval in data.get('param_vals'):
            name = pval.get('name')
            value = pval.get('value')
            pval_dao = ParamVal(model_dao.id, name, value)
            db.session.add(pval_dao)

        db.session.add(model_dao)
        db.session.commit()
        return model_dao.id, 201


@ns.route('/<string:id>')
@api.response(404, 'Model not found.')
class ModelItem(Resource):
    
    @api.marshal_with(model)
    def get(self, id):
        """
        Returns a model.
        """
        return Model.query.filter(Model.id == id).one()


    @api.response(204, 'Model successfully stopped.')
    def delete(self, id):
        """
        Stops a model.
        """
        raise NotImplementedError()
        # return None, 204