import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import dash_table
import pandas as pd
from collections import OrderedDict
import os
import json
import requests
import base64, io
import socket

host = socket.gethostbyname(socket.gethostname())

canned_model = False

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, './style.css'],
        meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1.0'}])

pop_model_params = {}
pop_model_params['MobileNet'] = [{"name": "MobileNet-", "description": "Using logistic regression to ", "max_num_models": 100, 
                               "model_selection_algorithm": "GridSearch", "executable_entrypoint": 'Do not need'},
                              
                              {'name': 'Learning Rate', 'param_type': 'hp_loguniform', 'choices': 'None', 
                               'min': -4, 'max': 1, 'q': 'None', 'dtype': 'dtype_float'},
                              
                              {'name': 'Batch Size', 'param_type': 'hp_quniform', 'choices': 'None', 
                               'min': 16, 'max': 256, 'q': 16, 'dtype': 'dtype_int'},
                              
                              {'name': 'Regularization Type', 'param_type': 'hp_choice', 'choices': 'l1,l2', 
                               'min': 'None', 'max': 'None', 'q': 'None', 'dtype': 'dtype_str'},
                             
                              {'name': 'Regularization Strength', 'param_type': 'hp_loguniform', 'choices': 'None', 
                               'min': -3, 'max': 1, 'q': 'None', 'dtype': 'dtype_float'}]

pop_model_params['MLP'] = [{"name": "MLP-", "description": "Using multilayer perceptron to ", "max_num_models": 100,
                               "model_selection_algorithm": "GridSearch", "executable_entrypoint": 'Do not need'},
                              
                              {'name': 'lr', 'param_type': 'hp_loguniform', 'choices': 'None', 
                               'min': -4, 'max': 1, 'q': 'None', 'dtype': 'dtype_float'},
                              
                              {'name': 'batch_size', 'param_type': 'hp_quniform', 'choices': 'None', 
                               'min': 64, 'max': 256, 'q': 32, 'dtype': 'dtype_int'},
                          
                              {'name': 'num_layers', 'param_type': 'hp_quniform', 'choices': 'None', 
                               'min': 3, 'max': 6, 'q': 1, 'dtype': 'dtype_int'}]

pop_model_params['Bert-Base'] = [{"name": "Bert-Base-", "description": "Using distill BERT to ", "max_num_models": 100,
                               "model_selection_algorithm": "GridSearch", "executable_entrypoint": 'Do not need'},
                              
                              {'name': 'Learning Rate', 'param_type': 'hp_loguniform', 'choices': 'None', 
                               'min': -4, 'max': 1, 'q': 'None', 'dtype': 'dtype_float'},
                              
                              {'name': 'Batch Size', 'param_type': 'hp_quniform', 'choices': 'None', 
                               'min': 16, 'max': 256, 'q': 16, 'dtype': 'dtype_int'}]

pop_model_params['ResNet'] = [{"name": "ResNet-", "description": "Using ResNet to ", "max_num_models": 100,
                               "model_selection_algorithm": "GridSearch", "executable_entrypoint": 'Do not need'},
                              
                              {'name': 'learning_rate', 'param_type': 'hp_loguniform', 'choices': 'None', 
                               'min': -4, 'max': 1, 'q': 'None', 'dtype': 'dtype_float'},
                              
                              {'name': 'batch_size', 'param_type': 'hp_quniform', 'choices': 'None', 
                               'min': 16, 'max': 256, 'q': 16, 'dtype': 'dtype_int'},
                             
                              {'name': 'l2_reg', 'param_type': 'hp_loguniform', 'choices': 'None', 
                               'min': -3, 'max': 1, 'q': 'None', 'dtype': 'dtype_float'}]

pop_model_params['Bert-Dil'] = [{"name": "Bert-Distil-", "description": "Using distill BERT to ", "max_num_models": 100,
                               "model_selection_algorithm": "GridSearch", "executable_entrypoint": 'Do not need'},
                              
                              {'name': 'Learning Rate', 'param_type': 'hp_loguniform', 'choices': 'None', 
                               'min': -4, 'max': 1, 'q': 'None', 'dtype': 'dtype_float'},
                              
                              {'name': 'Batch Size', 'param_type': 'hp_quniform', 'choices': 'None', 
                               'min': 16, 'max': 256, 'q': 16, 'dtype': 'dtype_int'}]


EXPERIMENT_ELEMENT = (
    "name", "description", "max_num_models", "feature_columns", "label_columns", 
    "max_train_epochs", "data_store_prefix_path", "executable_entrypoint", "model_selection_algorithm"
)
EXPERIMENT_TYPE = (
    "text", "text", "number", "text", "text", "number", "text", "text", "string"
)

# words description on left column of experiment setup secton
exp_describe_id = (
    "exp_describe_name", 
    "exp_describe_describe", 
    "exp_describe_max_model",
    "exp_describe_features", 
    "exp_describe_label", 
    "exp_describe_max_epoch",
    "exp_describe_data_path", 
    "exp_describe_func",
    "exp_describe_search_strategy"
)
describe_list = (
    "Name of experiment (string type)", 
    "Description (optional)", 
    "Max number of models for random and autoML (integer type)",
    "Name of features (string type, comma seperated)", 
    "Name of label (string type)", 
    "Maximum training epochs (integer type)",
    "Data store prefix path (string type)", 
    "Estimator function name (<module_name>:<function_name>)",
    "Hyperparameters search strategy (please select)"
)

tensorboard_frame = html.Iframe(src="http://localhost:6006/", style={"height": "1067px", "width": "100%"})

clone_checkbox = dcc.RadioItems(
                    id='clone_mode',
                    options=[
                        {'label': 'Warm Start on Base Model', 'value': 'warm'},
                        {'label': 'Start From Scratch', 'value': 'new'}
                    ],
                    value='warm',
                    labelStyle={'display': 'inline-block', "margin-right": "20px"},
                    style={"margin-bottom": "5px"}
                )

clone_table = dash_table.DataTable(
                id='clone_table',
                data=[],
                columns=[
                    {"name": 'Name', 'id': 'name', "selectable": True},
                    {"name": "Hyperparameter Type", "id": "param_type", 'presentation': 'dropdown'},
                    {"name": "Choices", "id": "choices", "selectable": True},
                    {"name": "Min Value", "id": "min", "selectable": True, "type": "numeric"},
                    {"name": "Max Value", "id": "max", "selectable": True, "type": "numeric"},
                    {"name": "Quantum", "id": "q", "selectable": True, "type": "numeric"},
                    {"name": "Data Type", "id": "dtype", 'presentation': 'dropdown'}
                ],
                editable=True,
                page_current=0,             # page number that user is on
                page_size=20,                # number of rows visible per page
                style_cell={                # ensure adequate header width when text is shorter than cell's text
                    'width': 'auto', 'height': 'auto', #'whiteSpace': 'normal'
                },
                style_cell_conditional=[    # align text columns to left. By default they are aligned to right
                    {
                        'if': {'column_id': c},
                        'textAlign': 'left'
                    } for c in ['name', 'param_type', "choices", "min", "max", "q", "dtype"]
                ],
                style_data={                # overflow cells' content into multiple lines
                    'whiteSpace': 'normal',
                    'height': 'auto'
                },
                dropdown={
                    'param_type': {
                        'options': [
                            {'label': 'categorical', 'value': 'hp_choice'},
                            {'label': 'uniform', 'value': 'hp_uniform'},
                            {'label': 'quantum uniform', 'value': 'hp_quniform'},
                            {'label': 'log uniform', 'value': 'hp_loguniform'},
                            {'label': 'quantum log uniform', 'value': 'hp_qloguniform'}
                        ],
                        'value':'hp_choice'
                    },
                    'dtype': {
                        'options': [
                            {'label': 'string', 'value': 'dtype_str'},
                            {'label': 'integer', 'value': 'dtype_int'},
                            {'label': 'float', 'value': 'dtype_float'}
                        ],
                        'value':'dtype_str'
                    }
                }
            )   

clone_name_form = dbc.FormGroup(
    [
        dbc.Label("Base Model Name", html_for="clone-name-row", width=3),
        dbc.Col(
            dbc.Input(
                type="text", id="clone-name-row", placeholder="Enter name", value = ""
            ),
            width=9,
        ),
    ],
    row=True,
)



# App layout ****************************************************************************
app.layout = dbc.Container([

    dbc.Navbar(
        [
            # Use row and col to control vertical alignment of logo / brand
            dbc.Row(
                [dbc.Col(dbc.NavbarBrand("Intermittent Human-in-the-Loop Cerebro", className="ml-2", style = {'font-size':'xx-large'}))],
                align="center",
                no_gutters=True,
            )
        ],
        color='#FF6F00',
        dark=True, 
        style = {'width': '100%'}
    ),
    
    
    # *************************************************************************************************
    #
    #     Model selection section
    #
    # *************************************************************************************************
    
    dbc.Row([dbc.Col(html.H3("Upload a model script file OR choose a popular model", className='text-center',
                             style={"margin-top": "15px"}))]),
    
    dbc.Row( [
        dbc.Col(
            dcc.Upload(
                id='upload-script',
                children = ['Drag and Drop or ',
                    html.A('Select a File', style ={ 'text-decoration': 'underline'})
                ], 
                style={
                    'width': '100%',
                    'height': '130px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center'
                },
                multiple=True
            ),
            className='column_left'
        ), 
        dbc.Col([
            html.H6("Select a popular model:"),
            dcc.Dropdown(id="slct_pop_model",
                 options=[
                     {"label": "MobileNet", "value": "MobileNet"},
                     {"label": "MLP", "value": "MLP"},
                     {"label": "BERT-base", "value": "Bert-Base"},
                     {"label": "ResNet-50", "value": "ResNet"},
                     {"label": "DistillBERT", "value": "Bert-Dil"}],
                 multi=False,
                 value="ResNet"
            ),
            html.Button('Select this model', id='slct_model', n_clicks=0, style={"margin-top": "15px"})
        ])
    ]),
    
    html.Div(id='script_response', children=[]),
    
    html.Hr(),
    
    # *************************************************************************************************
    #
    #     Experiment setup section
    #
    # *************************************************************************************************
    
    html.H3("Setup experiment"),
    
    html.Div(id='experiment-part', children=[
        dbc.Row([
            dbc.Col(
                [
                    dbc.Row([html.Div(id=Id, children=Describe, style= {'display': 'block'})]) for Id, Describe in zip(exp_describe_id, describe_list)
                ], width=8
            ),
            dbc.Col(
                [ 
                    dbc.Row([dcc.Input(
                        id="exp_{}".format(e),
                        # value=None,
                        type=t,
                        placeholder="{}:({})".format(e,t),
                        style={
                        'width': '240px',
                        'height': '25px'}
                    )])
                    for e,t in zip(EXPERIMENT_ELEMENT[:-1],EXPERIMENT_TYPE[:-1])
                ] +
                [
                    dbc.Row([dcc.Dropdown(id="model_slct_algorithm",
                         options=[
                             {"label": "GridSearch", "value": "GridSearch"},
                             {"label": "RandomSearch", "value": "RandomSearch"},
                             {"label": "HyperOpt", "value": "HyperOpt"}],
                         multi=False,
                         value="RandomSearch",
                         style={'width': "90%", 'height': '35px', 'display': 'block'}
                     )])
                ], width=4
            )
        ]),

        html.H5("Hyperparameter grid table:", style={"margin-top": "15px"}),

        dash_table.DataTable(
            id='datatable-params',
            data=[], #pop_model_params['ResNet'][1:],
            columns=[
                {"name": 'Name', 'id': 'name', "selectable": True},
                {"name": "Hyperparameter Type", "id": "param_type", 'presentation': 'dropdown'},
                {"name": "Choices", "id": "choices", "selectable": True},
                {"name": "Min Value", "id": "min", "selectable": True},
                {"name": "Max Value", "id": "max", "selectable": True},
                {"name": "Quantum", "id": "q", "selectable": True},
                {"name": "Data Type", "id": "dtype", 'presentation': 'dropdown'}
            ],
            editable=True,
            row_deletable=True,
            filter_action="native",     # allow filtering of data by user ('native') or not ('none')
            page_current=0,             # page number that user is on
            page_size=20,                # number of rows visible per page
            style_cell={                # ensure adequate header width when text is shorter than cell's text
                'width': 'auto', 'height': 'auto', #'whiteSpace': 'normal'
            },
            style_cell_conditional=[    # align text columns to left. By default they are aligned to right
                {
                    'if': {'column_id': c},
                    'textAlign': 'left'
                } for c in ['name', 'param_type', "choices", "min", "max", "q", "dtype"]
            ],
            style_data={                # overflow cells' content into multiple lines
                'whiteSpace': 'normal',
                'height': 'auto'
            },
            dropdown={
                'param_type': {
                    'options': [
                        {'label': 'categorical', 'value': 'hp_choice'},
                        {'label': 'uniform', 'value': 'hp_uniform'},
                        {'label': 'quantum uniform', 'value': 'hp_quniform'},
                        {'label': 'log uniform (base 10)', 'value': 'hp_loguniform'},
                        {'label': 'quantum log uniform (base 10)', 'value': 'hp_qloguniform'}
                    ],
                    'value':'hp_choice'
                },
                'dtype': {
                    'options': [
                        {'label': 'string', 'value': 'dtype_str'},
                        {'label': 'integer', 'value': 'dtype_int'},
                        {'label': 'float', 'value': 'dtype_float'}
                    ],
                    'value':'dtype_str'
                }
            }
        ),    

        html.Button('Add New Parameter(One row)', id='add_param', n_clicks=0, 
                    style={"margin-bottom": "15px", "margin-top": "15px"}),

        html.H5("Please check info above and launch the experiment by clicking the button below:"),
        html.Button('Launch This Experiment', id='add_exp', n_clicks=0, 
                        style = {'height': '40px', 'textAlign': 'center',
                                 'color': 'white', 'background-color': '#FF6F00', 'font-weight': 'bold'}),      
        html.Div(id='param_output', children=[])], style= {'display': 'block'}),
    html.Hr(),
 
    # *************************************************************************************************
    #
    #     Models inspection section
    #
    # *************************************************************************************************
    
    html.H3("Models"),
     
    html.Div(id="model_part", children=[
        html.H6("Input model name and operate on the model:"),
        dcc.Input(id="model_name_to_change", type="string", placeholder="Model Name",
                        style={'width': '200px', 'height': '35px', "margin-bottom": "5px"}),
        html.Br(), 
        dbc.Row([
            dbc.Col([html.Button('Resume', id='resume_', n_clicks=0, style={"margin-left": "5px", "margin-right": "2px"}),
                             #html.Button('Delete', id='delete_', n_clicks=0, style={"margin-left": "2px", "margin-right": "2px"}),
                             html.Button('Stop', id='stop_', n_clicks=0, style={"margin-left": "2px", "margin-right": "2px"}),
                             html.Button('Clone', id='clone_', n_clicks=0, style={"margin-left": "2px", "margin-right": "2px"})],
            ),
            dbc.Col([html.Button('Refresh', id='refresh_', n_clicks=0, style={"float": "right"})])
               
        ], style = {"margin-bottom": "15px"}),

        dbc.Modal(
            [
                dbc.ModalHeader("Clone Model"),
                dbc.ModalBody(id='modal_body', children=[clone_name_form, clone_checkbox, clone_table]),
                dbc.ModalFooter(
                    dbc.Button("Confirm", id="close_clone", n_clicks=0, className="ml-auto", 
                               style = {'color': 'white', 'background-color': '#FF6F00'})
                ),
            ],
            id="clone_modal",
            centered=True,
            is_open=False,
            style={"max-width": "none", "width": "90%"}
        ),

        dash_table.DataTable(
            id='datatable-interactivity',
            data=[],
            columns=[
                {"name": "Model Name", "id": "model_name", "selectable": True },
                {"name": "Experiment Name", "id": "exp_name", "selectable": True},
                {"name": "Parent", "id": "parent", "selectable": False},
                {"name": "Status", "id": "status", "selectable": False},
#                 {"name": "Learning Rate", "id": "learning_rate", "selectable": True},
#                 {"name": 'Batch Size', 'id': 'batch_size', "selectable": True},
#                 {"name": 'Model', 'id': 'model', "selectable": True},
#                 {"name": 'l2_Regularization', 'id': 'l2_reg', "selectable": True},
            ],
            editable=True,
            sort_action='native',
            filter_action="native",     # allow filtering of data by user ('native') or not ('none')
            page_current=0,             # page number that user is on
            page_size=100,                # number of rows visible per page
            style_cell={                # ensure adequate header width when text is shorter than cell's text
                'minWidth': 60, 'maxWidth': 120, 'width': 80
            },
            style_cell_conditional=[    # align text columns to left. By default they are aligned to right
                {
                    'if': {'column_id': c},
                    'textAlign': 'left'
                } for c in ["model_name", "exp_name", "parent","status"]
            ],
            style_data={                # overflow cells' content into multiple lines
                'whiteSpace': 'normal',
                'height': 'auto'
            }
        )], style= {'display': 'block'}),
    
    html.Hr(),
    html.Div(id='tensorboard', children=[tensorboard_frame]),
    html.Hr(),

])



def create_exp_metadata_dict(
    model_slct_algorithm,
    exp_name, 
    exp_description, 
    exp_max_num_models, 
    exp_feature_columns, 
    exp_label_columns, 
    exp_max_train_epochs,
    exp_data_store_prefix_path, 
    exp_executable_entrypoint,
    data):
    
    new_payload = {
      "name": exp_name,
      "description": exp_description if exp_description is not None else '',
      "model_selection_algorithm": model_slct_algorithm,
      "param_defs": data,
      "feature_columns": exp_feature_columns,
      "label_columns": exp_label_columns,
      "max_train_epochs": exp_max_train_epochs,
      "data_store_prefix_path": exp_data_store_prefix_path,
      "executable_entrypoint": exp_executable_entrypoint
    }
    
    if model_slct_algorithm != "GridSearch":
        new_payload['max_num_models'] = exp_max_num_models
        
    return new_payload

def create_clone_metadata_dict(exp, cloned_model_name, clone_mode, clone_table):
    new_payload = {
      "name": exp['name'] + '_c',
      "description": "cloned on " + cloned_model_name,
      "model_selection_algorithm": exp['model_selection_algorithm'],
      "param_defs": clone_table,
      "feature_columns": exp['feature_columns'],
      "label_columns": exp['label_columns'],
      "max_train_epochs": exp['max_train_epochs'],
      "data_store_prefix_path": exp['data_store_prefix_path'],
      "executable_entrypoint": exp['executable_entrypoint']
    }
    
    if clone_mode == 'warm':
        new_payload['name'] = new_payload['name'] + '_' + cloned_model_name
        new_payload['warm_start_from_cloned_model'] = True
        new_payload['clone_model_id'] = cloned_model_name
        
    if exp['model_selection_algorithm'] != "GridSearch":
        new_payload['model_selection_algorithm'] = exp['model_selection_algorithm']
        new_payload['max_num_models'] = exp['max_num_models']
        
    return new_payload

def extract_models(experiments, columns):
    models = []

    for exp in experiments:
        print(type(exp))
        exp_name = exp['name']
        
        print(exp_name)
        
        for param in exp['param_defs']:
            if param['name'] not in [c['name'] for c in columns]:
                columns.append({"name": param['name'], "id": param['name'], "selectable": True})
        
        for model in exp['models']:
            new_model = {}
            new_model['model_name'] = model['id']
            new_model['exp_name'] = exp_name
            new_model['parent'] = model['warm_start_model_id'] if model['warm_start_model_id'] is not None else '/' 
            new_model['status'] = model['status']
            for param in model['param_vals']:
                new_model[param['name']] = param['value']
            
            models.append(new_model)
        
    return models, columns

def update_model_table(columns):        
    model_get_url = "http://localhost:8889/api/experiments/"
    r = requests.get(model_get_url)
    exps = r.json()
    
    # print(r)
    # print(exps)
    # print(type(exps))
    
    return extract_models(exps, columns)

def datatable_to_numbers(datatable):
    for i in range(len(datatable)):
        if 'min' in datatable[i]:
            datatable[i]['min'] = int(datatable[i]['min'])
        if 'max' in datatable[i]:
            datatable[i]['max'] = int(datatable[i]['max'])
        if 'q' in datatable[i]:
            datatable[i]['q'] = int(datatable[i]['q'])

    return datatable

@app.callback([Output('upload-script', 'contents'),
              Output('script_response', 'children'),
              Output('experiment-part', 'style')],
              Input('upload-script', 'contents'),
              [State('upload-script', 'filename'),
               State('experiment-part', 'style')])
def upload_script(list_of_contents, list_of_names, exp_part_style):
    printout = None
    style= exp_part_style
    global canned_model
    if list_of_contents is not None:
        canned_model = False
        print("contents:", list_of_contents)
        print("filename:", list_of_names)
        content_type, content_string = list_of_contents[0].split(",")
        decoded = base64.b64decode(content_string)
        print("decoded")
        print(decoded)
        file_io = io.StringIO(decoded.decode("utf-8"))
        script_post_url = "http://localhost:8889/api/scripts/upload"
        print("list_of_names[0]")
        print(list_of_names[0])
        print("file_io")
        print(file_io)
        r = requests.post(script_post_url, files={'file':(list_of_names[0], file_io)})
        print("list of contents", list_of_contents)
        if r.status_code==201:
            print("A script has been successfully uploaded!")
            printout = "A script has been successfully uploaded!"
        style['display'] = 'block'
    return None, printout, style



@app.callback(
    [Output(component_id='datatable-params', component_property='data'),
     Output(component_id='add_param', component_property='n_clicks'), 
     Output(component_id='slct_model', component_property='n_clicks'),
     Output(component_id='datatable-params', component_property='editable'),
     Output('exp_executable_entrypoint', 'value')],
    [Input(component_id='add_param', component_property='n_clicks'),
     Input(component_id='slct_model', component_property='n_clicks')],
    [State('datatable-params', 'data'), 
     State('datatable-params', 'editable'), 
     State('slct_pop_model', 'value'),
     State('exp_executable_entrypoint', 'value')]
)
def add_param(add_param_n_clicks, slct_model_n_click, data, editable, pop_model, exp_executable_entrypoint):
    print('Button has been clicked in add_param.')   
    
    global canned_model 

    if add_param_n_clicks > 0:
        data.append({})

    
    if slct_model_n_click > 0:
        canned_model = True

        model_gen_file = pop_model + "_model_gen_script"
        
        with open(pop_model+"_model_gen_script.py") as f:
            string = f.read()
            
        file_io = io.StringIO(string)
        script_post_url = "http://localhost:8889/api/scripts/upload"
        r = requests.post(script_post_url, files={'file':(pop_model+"_model_gen_script.py", file_io)})
        
        data = pop_model_params[pop_model][1:]
        editable = True
        exp_executable_entrypoint = model_gen_file + ":estimator_gen_fn"
        
    return data, 0, 0, editable, exp_executable_entrypoint

@app.callback(
    [Output(component_id='datatable-interactivity', component_property='data'),
     Output(component_id='datatable-interactivity', component_property='columns'),
     Output(component_id='add_exp', component_property='n_clicks'),
     Output('clone_modal', 'is_open'),
     Output('clone_', 'n_clicks'),
     Output('close_clone', 'n_clicks'),
     Output('resume_', 'n_clicks'), 
     #Output('delete_', 'n_clicks'),
     Output('stop_', 'n_clicks'),
     Output('refresh_', 'n_clicks'),
     Output('model_part', 'style'),
     Output('exp_describe_max_model', 'style'),
     Output('exp_max_num_models', 'style'),
     Output('clone-name-row', 'value'),
     Output('clone_table', 'data')],
    [Input('model_slct_algorithm', 'value'),
     Input(component_id='add_exp', component_property='n_clicks'), 
     Input('clone_', 'n_clicks'), 
     Input('close_clone', 'n_clicks'),
     Input('resume_', 'n_clicks'), 
     #Input('delete_', 'n_clicks'),
     Input('stop_', 'n_clicks'), 
     Input('refresh_', 'n_clicks')],
    [State('datatable-params', 'data'), 
     State(component_id='datatable-interactivity', component_property='data'),
     State(component_id='datatable-interactivity', component_property='columns'),
     State('clone_modal', 'is_open'), 
     State('model_name_to_change', 'value'),
     State('tensorboard', 'children'), 
     State('model_part', 'style'),
     State('exp_describe_max_model', 'style'),
     State('exp_max_num_models', 'style'),
     State("exp_name", 'value'),
     State("exp_description", 'value'),
     State("exp_max_num_models", 'value'),
     State("exp_feature_columns", 'value'),
     State("exp_label_columns", 'value'),
     State("exp_max_train_epochs", 'value'),
     State("exp_data_store_prefix_path", 'value'),
     State("exp_executable_entrypoint", 'value'),
     State('modal_body', 'children'),
     State('clone-name-row', 'value'),
     State('clone_mode', 'value'),
     State('clone_table', 'data')
     ]
)  
def add_exp(# Input
            model_slct_algorithm, 
            n_clicks_add_exp, 
            clone_n_clicks, 
            close_clone_n_clicks, 
            resume_n_clicks,
            #delete_n_clicks, 
            stop_n_clicks, 
            refresh_n_clicks, 
            # State
            data, 
            return_data, 
            columns, 
            clone_modal_is_open, 
            model_name_to_change, 
            tensorboard, 
            model_part_style,
            exp_describe_max_model_style, 
            exp_max_num_models_style,
            exp_name, 
            exp_description, 
            exp_max_num_models,
            exp_feature_columns, 
            exp_label_columns, 
            exp_max_train_epochs,
            exp_data_store_prefix_path, 
            exp_executable_entrypoint,
            modal,
            cloned_model_name,
            clone_mode,
            clone_table):
    
    global canned_model 

    if model_slct_algorithm == "GridSearch":
        exp_describe_max_model_style['display'] = 'none'
        exp_max_num_models_style['display'] = 'none'
    else:
        exp_describe_max_model_style['display'] = 'block'
        exp_max_num_models_style['display'] = 'block'
    
    if n_clicks_add_exp > 0:
        exp_post_url = "http://localhost:8889/api/experiments/"
        
        exp_metadata = create_exp_metadata_dict(model_slct_algorithm, exp_name, exp_description, 
            exp_max_num_models, exp_feature_columns, exp_label_columns, exp_max_train_epochs,
            exp_data_store_prefix_path, exp_executable_entrypoint, data)
        
#         print(payload == exp_metadata)
#         print("payload:")
#         print(payload)
#         print("exp_metadata:")
#         print(exp_metadata)
        if canned_model:
            print("send MLP exp setup")
            r = requests.post(exp_post_url, json=MLP_payload)
        else:
            r = requests.post(exp_post_url, json=libsvm_payload)
        print(r)
        print(r.json())
        experiment_id = r.json()
        
        return_data, columns = update_model_table(columns)
        
        print(return_data)
        print(columns)
        
        # tensorboard.append(tensorboard_frame)
        
        return return_data, columns, 0, clone_modal_is_open, 0, 0, 0, 0, 0, {'display': 'block'}, \
                exp_describe_max_model_style, exp_max_num_models_style, model_name_to_change, clone_table

    if refresh_n_clicks > 0:

        return_data, columns = update_model_table(columns)
    
        print(return_data)
        print(columns)
        
        return return_data, columns, 0, clone_modal_is_open, 0, 0, 0, 0, 0, {'display': 'block'}, \
                exp_describe_max_model_style, exp_max_num_models_style, model_name_to_change, clone_table
        
    if clone_n_clicks > 0:
        model_get_url = "http://localhost:8889/api/models/" + model_name_to_change
        r = requests.get(model_get_url)
        mod = r.json()
        
        clone_table = []
        
        for param in mod['param_vals']:
            param_new = {
                'name': param['name'],
                "param_type": 'hp_choice',
                "choices": param['value'],
#                 "min": ,
#                 "max": ,
#                 "q" : ,
                "dtype" : param['dtype']
            }
            clone_table.append(param_new)
        
        return return_data, columns, 0, not clone_modal_is_open, 0, 0, 0, 0, 0, {'display': 'block'}, \
            exp_describe_max_model_style, exp_max_num_models_style, model_name_to_change, clone_table
    
    if close_clone_n_clicks > 0:
        model_get_url = "http://localhost:8889/api/models/" + cloned_model_name
        r = requests.get(model_get_url)
        mod = r.json()
        
        exp_id = mod['exp_id']
        
        exp_get_url = "http://localhost:8889/api/experiments/" + exp_id
        r = requests.get(exp_get_url)
        exp = r.json()
        
        # clone_table = datatable_to_numbers(clone_table)
        exp_metadata = create_clone_metadata_dict(exp, cloned_model_name, clone_mode, clone_table)
        
        exp_post_url = "http://localhost:8889/api/experiments/"
        print("clone model: ")
        print(exp_metadata)
        r = requests.post(exp_post_url, json=exp_metadata)
        experiment_id = r.json()
        print(r)
        print(experiment_id)
        
        return_data, columns = update_model_table(columns)
        
        # print(return_data)
        # print(columns)
        
        return return_data, columns, 0, not clone_modal_is_open, 0, 0, 0, 0, 0, {'display': 'block'}, \
            exp_describe_max_model_style, exp_max_num_models_style, model_name_to_change, clone_table
        
    if resume_n_clicks > 0:
        if model_name_to_change is not None:
            model_resume_url = "http://localhost:8889/api/models/resume/" + model_name_to_change
            r = requests.post(model_resume_url)
            print(r)
            print(r.json())
            return_data, columns = update_model_table(columns)        
        return return_data, columns, 0, clone_modal_is_open, 0, 0, 0, 0, 0, {'display': 'block'}, \
                exp_describe_max_model_style, exp_max_num_models_style, model_name_to_change, clone_table
        
    if stop_n_clicks > 0:
        if model_name_to_change is not None:
            model_stop_url = "http://localhost:8889/api/models/stop/" + model_name_to_change
#             print(model_stop_url)
            r = requests.post(model_stop_url)
#             print("stop results:")
#             print(r)
            return_data, columns = update_model_table(columns)
            
        return return_data, columns, 0, clone_modal_is_open, 0, 0, 0, 0, 0, {'display': 'block'}, \
                exp_describe_max_model_style, exp_max_num_models_style, model_name_to_change, clone_table
    
    
    return return_data, columns, 0, clone_modal_is_open, 0, 0, 0, 0, 0, model_part_style, \
            exp_describe_max_model_style, exp_max_num_models_style, model_name_to_change, clone_table



libsvm_payload = {
    "name": "Sample_libsvm",
    "description": "",
    "model_selection_algorithm": "GridSearch",
    "param_defs": [
        {
          "name": "batch_size",
          "param_type": "hp_choice",
          "choices": "96,128",
          "dtype": "dtype_int"
        },
        {
          "name": "lr",
          "param_type": "hp_choice",
          "choices": "0.01,0.001",
          "dtype": "dtype_float"
        }
    ],
    "feature_columns": "features",
    "label_columns": "label",
    "max_train_epochs": 10,
    "data_store_prefix_path": "/Users/Lee/Documents/MLsys/MS_Thesis/Trial/",
    "executable_entrypoint": "model_gen_script:estimator_gen_fn"
}

MLP_payload = {
    "name": "MLP",
    "description": "",
    "model_selection_algorithm": "RandomSearch",
    "max_num_models": 10,
    "param_defs": [
        {
            'name': 'lr', 
            'param_type': 'hp_loguniform',
            'min': -4, 
            'max': 1, 
            'dtype': 'dtype_float'
        },
        {
            'name': 'batch_size', 
            'param_type': 'hp_quniform', 
            'min': 64, 
            'max': 256, 
            'q': 32, 
            'dtype': 'dtype_int'
        },
        {
            'name': 'num_layers', 
            'param_type': 'hp_quniform',
            'min': 3, 
            'max': 6,
            'q': 1, 
            'dtype': 'dtype_int'
        }
    ],
    "feature_columns": "features",
    "label_columns": "label",
    "max_train_epochs": 10,
    "data_store_prefix_path": "/Users/Lee/Documents/MLsys/MS_Thesis/Trial/",
    "executable_entrypoint": "MLP_model_gen_script:estimator_gen_fn"
}


if __name__ == '__main__':
    host = socket.gethostbyname(socket.gethostname())
    print("host: ", host)
    app.run_server(debug=False, host=host, port=8051)