from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from waitress import serve
import pandas as pd
import joblib
import os

# Path to files on web server replacing curr_dir with BASE_DIR
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_files_dir = os.path.join(BASE_DIR, "model_files")
vectorizer_files_dir = os.path.join(BASE_DIR, "vectorizer_files")
model_file = "randfor_model_200_3.joblib"
vectorizer_file = "tfidf_vectorizer.joblib"

# Path to Heroku's /tmp directory
UPLOAD_FOLDER = "/tmp"

# Declare Flask App
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Limiting file size to 100 MB
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/uploader', methods=['POST', 'GET'])
def upload():
    model = joblib.load(os.path.join(model_files_dir, model_file))
    vectorizer = joblib.load(os.path.join(vectorizer_files_dir, vectorizer_file))

    # Get csv_file
    data_file = request.files['csv_file']
    if data_file.filename.split('.')[1].lower() == 'csv':
        filename = secure_filename(data_file.filename)
        data_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    else:
        return "<h1>Wrong file type</h1>"
    df = pd.read_csv(os.path.join(UPLOAD_FOLDER, data_file.filename))

    # Reformat data file for model
    df['title'] = df['title'].fillna('notitle')
    df['author'] = df['author'].fillna('noauthor')
    df['text'] = df['text'].fillna('notext')
    df['text'] = df['text'].str.replace('\\n',' ')
    df['text'] = df['text'].str.replace(r'<.*?>',' ', regex=True)
    df['merged'] = df['title'] + ' ' + df['author'] + ' ' + df['text']

    X_tfidfvec = vectorizer.transform(df['merged'])

    # Run model
    y_pred = model.predict(X_tfidfvec)
    
    # Store results
    df['predicted'] = y_pred
    df['predicted'] = df['predicted'].map({0: "Reliable News", 1: "Fake News"})

    # Clean up dataframe for webpage
    if 'label' in list(df.columns):
        df['label'] = df['label'].map({0: "Reliable News", 1: "Fake News"})
        df = df.rename(columns={'label':'actual'})
        df = df[['id', 'title', 'author', 'text', 'actual', 'predicted']]
    else: 
        df = df[['id', 'title', 'author', 'text', 'predicted']]
    
    df['text'] = df['text'].str.slice(0,100) + '[truncated]'

    return render_template("predictor.html", tables = [df.to_html()], titles = df.columns.values) 

# Run Web App
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    # WSGI production
    serve(app, host='0.0.0.0', port=port)    

    # Flask testing
    #app.run(debug=True,
    #        host="0.0.0.0",
    #        port=port)
