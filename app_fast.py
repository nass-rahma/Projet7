from fastapi import FastAPI
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import pickle
import uvicorn
import shap
import nest_asyncio

nest_asyncio.apply()

# Create a FastAPI instance
app = FastAPI()

# Loading the model and data
model = pickle.load(open('mlflow_model/model.pkl', 'rb'))
data = pd.read_csv('test_df.csv')
data_train = pd.read_csv('train_df.csv')


data_scaled = data.copy()

data_train_scaled = data_train.copy()


explainer = shap.TreeExplainer(model['classifier'])


# Functions
@app.get('/')
def welcome():
    """
    Welcome message.
    :param: None
    :return: Message (string).
    """
    return 'Hello API'


@app.get('/{client_id}')
def check_client_id(client_id: int):
    """
    Customer search in the database
    :param: client_id (int)
    :return: message (string).
    """
    if client_id in list(data['SK_ID_CURR']):
        return True
    else:
        return False


@app.get('/prediction/{client_id}')
def get_prediction(client_id: int):
    """
    Calculates the probability of default for a client.
    :param: client_id (int)
    :return: probability of default (float).
    """
    client_data = data[data['SK_ID_CURR'] == client_id]
    info_client = client_data.drop('SK_ID_CURR', axis=1)
    prediction = model.predict_proba(info_client)[0][1]
    return prediction



@app.get('/shaplocal/{client_id}')
def shap_values_local(client_id: int):
    """
    Calcule les SHAP values pour un client spécifique (local).
    
    """
    # 1) Extraire la ligne du client
    client_data = data_scaled[data_scaled['SK_ID_CURR'] == client_id]
    if client_data.empty:
        return {'error': 'Client ID not found.'}

    # 2) Retirer la colonne SK_ID_CURR
    X_client = client_data.drop('SK_ID_CURR', axis=1)

    # 3) Calculer les shap_values. 
    
    shap_vals_list = explainer.shap_values(X_client)

    # 4) On suppose qu'on veut les contributions de la classe 1
    #    => shap_vals_list[1] si c'est bien une liste de deux arrays
    if isinstance(shap_vals_list, list) and len(shap_vals_list) == 2:
        shap_val_class1 = shap_vals_list[1]  # shape (1, nb_features) si X_client a 1 ligne
        local_shap_vector = shap_val_class1[0]  # on prend la première (et unique) ligne
        base_value = explainer.expected_value[1]  # base value pour la classe 1
    else:
        
        shap_val_class1 = shap_vals_list
        
        local_shap_vector = shap_val_class1[0]
        
        base_value = explainer.expected_value

    # Conversion en listes Python pour la sérialisation
    shap_values_list = local_shap_vector.tolist()
    data_values_list = X_client.values[0].tolist()
    feature_names_list = X_client.columns.tolist()

    result = {
        'shap_values': shap_values_list,   # contributions locales
        'base_value': float(base_value) if isinstance(base_value, (int, float)) else base_value,
        'data': data_values_list,          # valeurs de l'observation
        'feature_names': feature_names_list
    }
    return result


@app.get('/shap/')
def shap_values():
    """ Calcul les shap values de l'ensemble du jeu de données
    :param:
    :return: shap values
    """
    # explainer = shap.TreeExplainer(model['classifier'])
    shap_val = explainer.shap_values(data_scaled.drop('SK_ID_CURR', axis=1))
    return {'shap_values_0': shap_val[0].tolist(),
            'shap_values_1': shap_val[1].tolist()}


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)