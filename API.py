#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, request
from flask_restx import Api, Resource, fields, reqparse
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
api = Api(
    app, 
    version='1.0', 
    title='Movie Genre Prediction API',
    description='Movie Genre Prediction API'
)

ns = api.namespace('predict', description='Movie Genre')

# Cargar el modelo y el vectorizador
model = joblib.load('genre_classification_model.pkl')
vect_tfidf = joblib.load('tfidf_vectorizer.pkl')

# Definición de campos de respuesta
resource_fields = api.model('Resource', {
    'predictions': fields.List(fields.Integer),
})

# Definición de argumentos para la API
parser = reqparse.RequestParser()
parser.add_argument('plots', type=str, action='append', required=True, help='Plots of the movies')

# Definición de la clase para la API
@ns.route('/')
class MovieGenreApi(Resource):

    @api.expect(parser)
    @api.marshal_with(resource_fields)
    def post(self):
        args = parser.parse_args()

        # Preprocesar y transformar las sinopsis de las películas
        X_tfidf = vect_tfidf.transform(args['plots'])
        X_tfidf_dense = X_tfidf.toarray()
        
        # Hacer predicciones
        predictions = model.predict(X_tfidf_dense)
        
        # Convertir predicciones a una lista
        predictions_list = predictions.tolist()
        
        return {'predictions': predictions_list}, 200

# Inicia el servidor Flask
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)

