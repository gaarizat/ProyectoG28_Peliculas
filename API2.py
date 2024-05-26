#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, request
from flask_restx import Api, Resource, fields, reqparse
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from flask_cors import CORS
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string

app = Flask(__name__)
CORS(app)
api = Api(
    app, 
    version='1.0', 
    title='Movie Genre Prediction API',
    description='Movie Genre Prediction API'
)

ns = api.namespace('predict', description='Movie Genre')

# Definir una función para preprocesar el texto
def preprocess_text(text):
    wordnet_lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Convertir el texto a minúsculas
    text = text.lower()
    # Eliminar signos de puntuación
    text = ''.join([char for char in text if char not in string.punctuation])
    # Eliminar stopwords
    words = text.split()
    text = ' '.join([word for word in words if word not in stop_words])
    # Lematizar palabras
    text = ' '.join([wordnet_lemmatizer.lemmatize(word) for word in text.split()])
    
    return text

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
        preprocessed_plots = [preprocess_text(plot) for plot in args['plots']]
        X_tfidf = vect_tfidf.transform(preprocessed_plots)
        X_tfidf_dense = X_tfidf.toarray()
        
        # Hacer predicciones
        predictions = model.predict(X_tfidf_dense)
        
        # Convertir predicciones a una lista
        predictions_list = predictions.tolist()
        
        return {'predictions': predictions_list}, 200

# Inicia el servidor Flask
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)


# In[ ]:




