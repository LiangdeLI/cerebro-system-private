# Copyright 2021 Supun Nakandala and Arun Kumar. All Rights Reserved.
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
import uuid
from datetime import datetime
from . import db

################### Experiment ##################
class Experiment(db.Model):
    id = db.Column(db.String(32), primary_key=True)
    name = db.Column(db.String(32))
    description = db.Column(db.String(512))
    creation_time = db.Column(db.DateTime)
    last_update_time = db.Column(db.DateTime)
    status = db.Column(db.String(32))
    model_selection_algorithm = db.Column(db.String(32))
    max_num_models = db.Column(db.Integer())
    feature_columns = db.Column(db.String(512))
    label_columns = db.Column(db.String(512))
    max_train_epochs = db.Column(db.Integer())
    data_store_prefix_path = db.Column(db.String(512))
    executable_entrypoint = db.Column(db.String(512))

    param_defs = db.relationship('ParamDef', backref='experiment', lazy='dynamic')
    models = db.relationship('Model', backref='model', lazy='dynamic')


    def __init__(self, name, description, model_selection_algorithm, max_num_models, feature_columns, label_columns, max_train_epochs,
        data_store_prefix_path, executable_entrypoint):
        
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.creation_time = datetime.utcnow()
        self.last_update_time = self.creation_time
        self.status = 'created'
        self.model_selection_algorithm = model_selection_algorithm
        self.max_num_models = max_num_models
        self.feature_columns = feature_columns
        self.label_columns = label_columns
        self.max_train_epochs = max_train_epochs
        self.data_store_prefix_path = data_store_prefix_path
        self.executable_entrypoint = executable_entrypoint

    def __repr__(self):
        return '<Experiment %r>' % self.id


class Model(db.Model):
    id = db.Column(db.String(32), primary_key=True)
    exp_id = db.Column(db.String(32), db.ForeignKey('experiment.id'))
    creation_time = db.Column(db.DateTime)
    last_update_time = db.Column(db.DateTime)
    status = db.Column(db.String(32))
    num_trained_epochs = db.Column(db.Integer())
    max_train_epochs = db.Column(db.Integer())

    param_vals = db.relationship('ParamVal', backref='model', lazy='dynamic')

    def __init__(self, exp_id, num_trained_epochs, max_train_epochs):
        self.id = str(uuid.uuid4())
        self.exp_id = exp_id        
        self.creation_time = datetime.utcnow()
        self.last_update_time = self.creation_time
        self.status = 'created'
        self.num_trained_epochs = num_trained_epochs
        self.max_train_epochs = max_train_epochs

    def __repr__(self):
        return '<Model %r>' % self.id


class ParamDef(db.Model):
    exp_id = db.Column(db.String(32), db.ForeignKey('experiment.id'), primary_key=True)
    name = db.Column(db.String(32), primary_key=True)
    param_type = db.Column(db.String(32))
    choices = db.Column(db.String(512))
    min = db.Column(db.Float())
    max = db.Column(db.Float())
    q = db.Column(db.Integer())

    def __init__(self, exp_id, name, param_type, choices=None, min=0, max=0, q=0):
        self.exp_id = exp_id
        self.name = name
        self.param_type = param_type
        self.choices = choices
        self.min = min
        self.max = max
        self.q = q

    def __repr__(self):
        return '<ParamDef %r>' % self.id


class ParamVal(db.Model):
    name = db.Column(db.String(32), db.ForeignKey('param_def.name'), primary_key=True)
    model_id = db.Column(db.String(32), db.ForeignKey('model.id'))
    value = db.Column(db.String(32))
    
    def __init__(self, model_id, name, value):
        self.model_id = model_id
        self.name = name
        self.value = value

    def __repr__(self):
        return '<ParamVal %r>' % self.id
