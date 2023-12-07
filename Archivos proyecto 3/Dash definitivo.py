import pandas as pd
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from sklearn.model_selection import train_test_split
import dash
from dash import dcc, html
from dash.dependencies import Input, Output 
import plotly.express as px 
import pickle
from pgmpy.inference import VariableElimination



df = pd.read_csv("/Users/sergioavendano/Desktop/Escritorio/Universidad/9semestre/Analitica/Proyecto3/3 ubicaciones.csv")

dataFrameDatos = pd.DataFrame()
nombreColumnas = {}

for columna in df.columns[0:-1]:

    valor = 0
    diccionario = {}
    for categoria in df[columna].unique():
        diccionario[categoria] = valor
        valor += 1
        nombreColumnas[columna] = diccionario




filename = "/Users/sergioavendano/Desktop/Escritorio/Universidad/9semestre/Analitica/Proyecto3/ResultadoshT.pkl"

file = open(filename, 'rb')
modeloPredictivo = pickle.load(file)
file.close()

# Promedio del puntaje global
average_punt_global = df['punt_global'].mean()

# Nombres columnas
column_name_mapping = {
    'periodo': 'Periodo',
    'cole_depto_ubicacion': 'Departamento',
    'fami_tieneinternet': 'Familia tiene internet?',
    'cole_jornada': 'Jornada',
    'cole_bilingue': 'El colegio es bilingue?',
    'fami_estratovivienda': 'Estrato'}



infer = VariableElimination(modeloPredictivo)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

#dataFrameDatos['puntaje_Global'] = df['punt_global']

app = dash.Dash(__name__,external_stylesheets=external_stylesheets)
server = app.server


# Diseño del dashboard
app.layout = html.Div([
    html.H1("Desempeño de estudiantes en la pruebas Saber 11"),
    html.H2("Llene los campos solicitados para realizar una predicción del perfil del estudiante deseado"),

    html.Div("Seleccione el estrato al cual pertenece el estudiante"),
    dcc.Dropdown(
        id = " estrato-dropdown",
        options = [{'label': key,'value': value} for key,value in nombreColumnas['fami_estratovivienda'].items()],
        value = "",
    ),

    html.Div("Seleccione si hay internet en su hogar"), 
    dcc.Dropdown(
        id="internet-dropdown", 
        options= [{'label': key, 'value': value} for key,value in nombreColumnas['fami_tieneinternet'].items()],
        value = "",
    ),  

    html.Div("Indique que tipo de jornada tiene el colegio del estudiante"), 
    dcc.Dropdown(
        id = "jornada-dropdown",
        options = [{'label': key, 'value': value} for key,value in nombreColumnas['cole_jornada'].items()],
        value="",
    ),

    html.Div("Indique si el colegio es bilingue"), 
    dcc.Dropdown(
        id = "bilingue-dropdown",
        options = [{'label': key, 'value': value} for key,value in nombreColumnas['cole_bilingue'].items()],
        value = "",
    ),

    html.Div("Indique en que departamento reside"), 
    dcc.Dropdown(
        id="departamento-dropdown",
        options=[{'label': key, 'value': value} for key,value in nombreColumnas['cole_depto_ubicacion'].items()],
        value="",
    ),
 
    html.H2("Con base en su selección de la preguntas anteriores, la predicción del puntaje en la prueba saber 11 es: "), 
    dcc.Textarea(
        id = 'prediction',
        value='La prediccion es:    Entre 300 y 400',
        style={'width': '30%', 'height': 50, "fontsize":"40px"},
        disabled = True
    ),

    html.H2("Acá puede observar la distribución de puntaje de acuerdo a una variable de interés que desee: "), 


    html.Div([
        html.Label("Selecciona la variable de su interes:"),
        dcc.Dropdown(
            id='dropdown-column',
            options=[{'label': column_name_mapping[col], 'value': col} for col in df.columns if col != 'punt_global'],
            value=df.columns[0]  
        )
    ]),
    
    dcc.Graph(id='histogram'),
    dcc.Graph(id='box-plot'),
    dcc.Graph(id='pie-chart'),

    #html.Button('Predecir Puntaje Global', id='button-prediccionarioion'),

    #dcc.Graph(id='grafico-pie')

])

@app.callback(
    Output('prediction', 'value'),
    [Input('opcion-internet', 'value'),
    Input('opcion-jornada', 'value'),
    Input('opcion-bilingue', 'value'),
    Input('opcion-estrato', 'value'),
    Input('opcion-departamento', 'value')]
)


@app.callback(
    [Output('histogram', 'figure'),
     Output('box-plot', 'figure'),
     Output('pie-chart', 'figure')],
    [Input('dropdown-column', 'value')]
)
def update_plots(selected_column):
    # Agrupar por la columna seleccionada y calcular el promedio del puntaje global
    grouped_data = df.groupby(selected_column)['punt_global'].mean().reset_index()
    

    histogram_fig = px.bar(grouped_data, x=selected_column, y='punt_global', 
                           title=f'Histograma Comparativo Promedio de Puntaje Global por {column_name_mapping[selected_column]}',
                           labels={'punt_global': 'Promedio de Puntaje Global', selected_column: column_name_mapping[selected_column]})
    
    # Linea promedio puntaje
    histogram_fig.add_hline(y=average_punt_global, line_dash="dash", line_color="red", 
                            annotation_text=f'Promedio General: {average_punt_global:.2f}', 
                            annotation_position="bottom right")
    

    box_plot_fig = px.box(df, x=selected_column, y='punt_global',
                          title=f'Box Plot de Puntaje Global por {column_name_mapping[selected_column]}',
                          labels={'punt_global': 'Puntaje Global', selected_column: column_name_mapping[selected_column]})
    

    pie_chart_fig = px.pie(df, names=selected_column, title=f'Distribución de {column_name_mapping[selected_column]}')
    
    return histogram_fig, box_plot_fig, pie_chart_fig

def actualizar_grafico(internet, jornada, bilingue, estrato, departamento_residencia):

    listaDesplegable = [internet, jornada, bilingue, estrato, departamento_residencia]
    columnas = ['fami_tieneinternet', 'cole_jornada', 'cole_bilingue', 'fami_estratovivienda', 'cole_depto_ubicacion']
    evi = {}
    for i in range(0,len(columnas)):
        evi[columnas[i]] = listaDesplegable[i]
    inferencia = infer.query(['puntaje_Global'],evidence = evi)
    valores = list(inferencia.values)
    proba = max(valores)
    target = valores.index(max(valores))

    if target == 1:
        respuesta = "Entre 0 y 200"
    elif target == 2:
        respuesta = "Entre 200 y 300"
    elif target == 3:
        respuesta = "Entre 300 y 400"
    else:
        respuesta = "Entre 400 y 500"

    respuestaFinal = "Usted podria sacar" + respuesta + "con probablidad de " + str(round(proba * 100)) + "%" 

    return respuestaFinal   

#@app.callback(
#    Output('graphBoxPlot', 'figure'),
#    Input('categoriaBoxPlot- dropdown', 'value')
#)

#def graficaBoxPlot(categoria):
#    if categoria is not None and categoria != "":
#        color_discrete_map = {"Box 1": "green", "Box 2" : "blue","Box 3": "orange", "Box 4": "red"}
#        figura = px.box(df, x-categoria, y="punt_global", points="all", title= "Distribución puntaje pruebas Saber 11")
#        figura.show()
#        return figura 
#    else:
#        return dash.no_update
    
#@app.callback(
#    Output('rubrocategoriaHist-dropdown', 'options'),
#    Input('categoriaHist-dropdown', 'value')
#)

#def generarListalistaDesplegable(categoria):
#    if not categoria:
#        return dash.no_update
#    listaDesplegable = nombreColumnas[categoria].keys()
#    opcionesActualizadas = [{'label': opcion, 'value': opcion} for opcion in listaDesplegable] 
#    return opcionesActualizadas

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=True)