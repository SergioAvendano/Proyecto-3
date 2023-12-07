# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 15:12:25 2023

@author: j.torresp
"""

import pandas as pd
from pgmpy.models import BayesianModel
import networkx as nx
import matplotlib.pyplot as plt

# Leer datos
df = pd.read_excel("C:/Users/j.torresp/Documents/AndesU/AnaliticaProyecto3/DatosP3Final.xlsx")


from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import K2Score
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.estimators import BicScore
from pgmpy.models import Bayesianetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import scipy
from sklearn.model_selection import train_test_split
import pyparsing
import torch
import statsmodels
import tqdm
import joblib
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import ConfusionMatrixDisplay
from pgmpy.estimators import ParameterEstimator



variables = df[['periodo', 'cole_depto_ubicacion', 'fami_tieneinternet', 
             'cole_jornada', 'cole_bilingue', 'fami_estratovivienda', 
             'punt_global']]

train,test = train_test_split(df, test_size=0.2,random_state= 42, stratify=variables['punt_global'])

import mlflow
import mlflow.sklearn


Experimento=mlflow.set_experiment('experimentoK2')


indegree=100000


with mlflow.start_run (experiment_id=Experimento.experiment_id):

    scoring_method = K2Score(data=train)
    esth = HillClimbSearch(data=train)
    estimated_model = esth.estimate(
        scoring_method=scoring_method, max_indegree=indegree, max_iter=int(1e4)
    )
    
    
    estimated_model= BayesianNetwork(estimated_model)
    estimated_model.fit(data=train, estimator = MaximumLikelihoodEstimator)
    
    
    
    bic_score= BicScore(train)
    
    
    inference = VariableElimination(estimated_model)
    
    
    # Definir las variables observadas y no observadas
    
    variables = ['periodo', 'cole_depto_ubicacion', 'fami_tieneinternet', 
                 'cole_jornada', 'cole_bilingue', 'fami_estratovivienda', 
                 'punt_global']

    
    observed_variables = ['periodo', 'cole_depto_ubicacion', 'fami_tieneinternet', 'cole_jornada', 'cole_bilingue', 'fami_estratovivienda']
    target_variable = 'punt_global'
    
    
    # Convertir las columnas a un diccionario
    evidence_dict = train[observed_variables].to_dict(orient='records')[0]
    
    # Convertir a BayesianNetwork
    bayesian_model = BayesianNetwork(estimated_model)
    
    predicted_values = inference.map_query(variables=[target_variable], evidence=train[observed_variables].iloc[0].to_dict())
    
    # Obtener las etiquetas reales
    true_values = train[target_variable]
    
    # Convertir a listas
    true_values_list = list(true_values)
    predicted_values_list = [predicted_values[target_variable].item()]  # Convertir el valor predicho a un tipo de datos nativo
    
    # Calcular la precisión
    accuracy = accuracy_score(true_values_list, [predicted_values_list[0] for _ in range(len(true_values_list))])
   
    
    
    
    
#parametros
mlflow.log_param("max_indegree", indegree)

#Registro modelo
mlflow.log_artifact (estimated_model)



#Metrics
mlflow.log_metric("BIC", bic_score)
mlflow.log_metric("K2", scoring_method)
mlflow.log_metric("Accuracy", accuracy)









