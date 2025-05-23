from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session, send_file, send_from_directory
import psycopg2
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
import io
import bcrypt
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import logging
import time
import retrying
import warnings

# Suppress pandas psycopg2 warning
warnings.filterwarnings("ignore", category=UserWarning, message="pandas only supports SQLAlchemy connectable")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'jodhu@12')

# Database configuration
db_config = {
    'host': os.getenv('DATABASE_HOST', 'dpg-d0mq9qe3jp1c738j58dg-a'),
    'user': os.getenv('DATABASE_USER', 'root'),
    'password': os.getenv('DATABASE_PASSWORD', 'btRdOQDtI8AJxjjyf9L1Kw5fiMGxxzPD'),
    'dbname': os.getenv('DATABASE_NAME', 'product_reviews_de1l'),
    'port': os.getenv('DATABASE_PORT', '5432')
}

# NLP Model variables
model = None
vectorizer = None
reviewed_comments = []

def init_nltk():
    """Initialize NLTK resources with retries."""
    try:
        nltk_data_path = '/opt/render/nltk_data'
        if not os.path.exists(nltk_data_path):
            os.makedirs(nltk_data_path, exist_ok=True)
        nltk.data.path.append(nltk_data_path)
        resources = ['stopwords', 'wordnet', 'omw-1.4']
        
        @retrying.retry(stop_max_attempt_number=3, wait_fixed=2000)
        def download_resource(resource):
            try:
                nltk.data.find(f'corpora/{resource}')
                logger.info(f"NLTK resource '{resource}' already exists")
            except LookupError:
                logger.info(f"Downloading NLTK resource '{resource}' to {nltk_data_path}")
                nltk.download(resource, download_dir=nltk_data_path, quiet=True)
                logger.info(f"Successfully downloaded '{resource}'")
        
        for resource in resources:
            download_resource(resource)
        
        try:
            stopwords.words('english')
            logger.info("NLTK stopwords loaded successfully")
        except LookupError:
            logger.warning("NLTK stopwords not found after download attempts")
    except Exception as e:
        logger.error(f"Failed to initialize NLTK resources: {e}")

def init_db():
    """Initialize database tables."""
    try:
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(50) UNIQUE,
                email VARCHAR(100) UNIQUE,
                password VARCHAR(100)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reviews (
                id SERIAL PRIMARY KEY,
                review_text TEXT,
                label VARCHAR(50)
            )
        ''')
        conn.commit()
        logger.info("Database tables initialized successfully")
    except psycopg2.Error as e:
        logger.error(f"Failed to initialize database: {e}")
        raise
    finally:
        cursor.close()
        conn.close()

# Initialize database and NLTK
init_db()
init_nltk()

def preprocess_text(text):
    """Preprocess text with fallback if stopwords are unavailable."""
    lemmatizer = WordNetLemmatizer()
    try:
        stop_words = set(stopwords.words('english'))
    except LookupError:
        logger.warning("Stopwords not found, skipping stopword removal")
        stop_words = set()
    words = [lemmatizer.lemmatize(word.lower()) for word in text.split() if word.lower() not in stop_words]
    return ' '.join(words)

def login_required(f):
    def wrap(*args, **kwargs):
        if 'logged_in' not in session:
            flash('Please login to access this page', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    wrap.__name__ = f.__name__
    return wrap

@app.route('/')
def index():
    return render_template('index.html', logged_in='logged_in' in session)

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'logged_in' in session:
        return redirect(url_for('upload'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password'].encode('utf-8')
        try:
            conn = psycopg2.connect(**db_config)
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE username = %s', (username,))
            user = cursor.fetchone()
            if user and bcrypt.checkpw(password, user[3].encode('utf-8')):
                session['logged_in'] = True
                session['username'] = username
                return redirect(url_for('upload'))
            flash('Invalid credentials', 'error')
        except psycopg2.Error as e:
            logger.error(f"Database error during login: {e}")
            flash('Database error, please try again later', 'error')
        finally:
            cursor.close()
            conn.close()
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'logged_in' in session:
        return redirect(url_for('upload'))
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return render_template('register.html')
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        try:
            conn = psycopg2.connect(**db_config)
            cursor = conn.cursor()
            cursor.execute('INSERT INTO users (username, email, password) VALUES (%s, %s, %s)', 
                           (username, email, hashed_password))
            conn.commit()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except psycopg2.Error as e:
            logger.error(f"Database error during registration: {e}")
            flash('Username or email already exists', 'error')
        finally:
            cursor.close()
            conn.close()
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session.pop('username', None)
    flash('You have been logged out', 'success')
    return redirect(url_for('index'))

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)
        if file and file.filename.endswith('.csv'):
            try:
                df = pd.read_csv(file)
                logger.info(f"Uploaded CSV: {len(df)} rows, columns: {df.columns.tolist()}")
                conn = psycopg2.connect(**db_config)
                cursor = conn.cursor()
                cursor.execute('DELETE FROM reviews')
                conn.commit()
                for _, row in df.iterrows():
                    cursor.execute('INSERT INTO reviews (id, review_text, label) VALUES (%s, %s, %s)',
                                   (row['id'], row['review_text'], row['label']))
                conn.commit()
                flash('File uploaded successfully', 'success')
                return redirect(url_for('preview'))
            except (psycopg2.Error, pd.errors.ParserError) as e:
                logger.error(f"Error during upload: {e}")
                flash('Error uploading file, please check the CSV format', 'error')
            finally:
                cursor.close()
                conn.close()
    return render_template('upload.html', username=session.get('username'))

@app.route('/preview')
@login_required
def preview():
    try:
        conn = psycopg2.connect(**db_config)
        df = pd.read_sql('SELECT * FROM reviews', conn)
        logger.info(f"Preview: {len(df)} rows retrieved")
        if df.empty:
            flash('No dataset uploaded. Please upload a dataset to proceed.', 'error')
            return redirect(url_for('upload'))
        table_data = {
            'name': 'Reviews Preview',
            'columns': df.columns.tolist(),
            'data': df.values.tolist()
        }
        return render_template('preview.html', tables=[table_data], username=session.get('username'))
    except psycopg2.Error as e:
        logger.error(f"Database error during preview: {e}")
        flash('Database error, please try again later', 'error')
        return redirect(url_for('upload'))
    finally:
        conn.close()

@app.route('/analyze')
@login_required
def analyze():
    global model, vectorizer
    try:
        start_time = time.time()
        logger.info("Starting model training")
        conn = psycopg2.connect(**db_config)
        df = pd.read_sql('SELECT * FROM reviews', conn)
        logger.info(f"Retrieved {len(df)} reviews in {time.time() - start_time:.2f} seconds")
        if df.empty:
            flash('No reviews available for training', 'error')
            return redirect(url_for('upload'))
        if len(df) > 20:
            logger.warning(f"Dataset size ({len(df)}) may cause timeouts. Consider using a smaller dataset.")
            flash('Dataset is large. For faster training, use a CSV with 10-20 rows.', 'warning')

        logger.info("Preprocessing text")
        preprocess_start = time.time()
        df['processed_text'] = df['review_text'].apply(preprocess_text)
        logger.info(f"Preprocessing completed in {time.time() - preprocess_start:.2f} seconds")

        logger.info("Vectorizing text")
        vectorize_start = time.time()
        vectorizer = TfidfVectorizer(min_df=1, max_df=0.9, ngram_range=(1, 1))
        X = vectorizer.fit_transform(df['processed_text'])
        logger.info(f"Vectorization completed in {time.time() - vectorize_start:.2f} seconds")

        df['label_mapped'] = df['label'].str.lower().apply(
            lambda x: 'Positive' if x in ['positive', 'accurate'] else 'Neutral' if x == 'neutral' else 'Negative'
        )
        y = df['label_mapped'].map({'Positive': 1, 'Neutral': 0, 'Negative': 0})
        logger.info(f"Labels: {df['label_mapped'].value_counts().to_dict()}")

        if len(y.unique()) < 2:
            flash('Error: Dataset contains only one class. Please upload a CSV with diverse labels.', 'error')
            return redirect(url_for('upload'))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logger.info(f"Train split: {X_train.shape[0]} samples, Test split: {X_test.shape[0]} samples")

        if len(pd.Series(y_train).unique()) < 2:
            flash('Error: Training split contains only one class. Try uploading a larger dataset.', 'error')
            return redirect(url_for('upload'))

        logger.info("Training model")
        train_start = time.time()
        param_grid = {
            'estimator__C': [1],
            'estimator__penalty': ['l2'],
            'estimator__loss': ['squared_hinge']
        }
        base_model = LinearSVC(random_state=42, max_iter=1000, dual=False)
        calibrated_model = CalibratedClassifierCV(base_model, method='sigmoid', cv=2)
        grid_search = GridSearchCV(calibrated_model, param_grid, cv=2, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        logger.info(f"Model training completed in {time.time() - train_start:.2f} seconds")

        model = grid_search.best_estimator_
        accuracy = model.score(X_test, y_test)
        logger.info(f"Total training time: {time.time() - start_time:.2f} seconds. Test accuracy: {accuracy:.2f}")
        flash(f'Model trained successfully. Test accuracy: {accuracy:.2f}', 'success')
        return redirect(url_for('review'))
    except TimeoutError:
        logger.error("Model training timed out")
        flash('Training timed out. Try a smaller dataset (10-20 rows).', 'error')
        return redirect(url_for('upload'))
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        flash(f'Analysis failed: {str(e)}', 'error')
        return redirect(url_for('upload'))
    finally:
        conn.close()

@app.route('/review', methods=['GET', 'POST'])
@login_required
def review():
    global model, vectorizer, reviewed_comments
    if request.method == 'POST':
        comment = request.form.get('comment', '').strip()
        if not comment:
            flash('Please enter a comment to analyze', 'error')
            return render_template('review.html', username=session.get('username'))

        if not model or not vectorizer:
            flash('Model not trained yet. Please train the model first.', 'error')
            return redirect(url_for('analyze'))

        try:
            processed_comment = preprocess_text(comment)
            comment_vec = vectorizer.transform([processed_comment])
            prob_array = model.predict_proba(comment_vec)[0]
            probability = float(prob_array[1] * 100)

            classification = (
                'EXTREMIST' if probability < 40 else
                'MODERATE' if probability <= 70 else
                'ACCURATE'
            )

            reviewed_comments.append({
                'comment': comment,
                'probability': probability,
                'classification': classification
            })

            return render_template(
                'review.html',
                probability=probability,
                classification=classification,
                comment=comment,
                reviewed_comments=reviewed_comments,
                username=session.get('username')
            )
        except Exception as e:
            logger.error(f"Error processing review: {e}")
            flash(f'Error processing review: {str(e)}', 'error')
            return render_template('review.html', username=session.get('username'))
    return render_template('review.html', reviewed_comments=reviewed_comments, username=session.get('username'))

@app.route('/download_csv_report')
@login_required
def download_csv_report():
    global reviewed_comments
    if not reviewed_comments:
        flash('No reviews available to download.', 'error')
        return redirect(request.referrer or url_for('dashboard'))
    
    try:
        df = pd.DataFrame(reviewed_comments)
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        return send_file(
            io.BytesIO(csv_buffer.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name='sentiment_analysis_report.csv'
        )
    except Exception as e:
        logger.error(f"Error generating CSV report: {e}")
        flash(f'Error generating CSV report: {str(e)}', 'error')
        return redirect(request.referrer or url_for('dashboard'))

@app.route('/download_pdf_report')
@login_required
def download_pdf_report():
    global reviewed_comments
    if not reviewed_comments:
        flash('No reviews available to download.', 'error')
        return redirect(request.referrer or url_for('dashboard'))
    
    try:
        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
        elements = []
        styles = getSampleStyleSheet()
        title = Paragraph("Sentiment Analysis Report", styles['Heading1'])
        elements.append(title)
        data = [['Comment', 'Probability (%)', 'Classification']]
        for review in reviewed_comments:
            data.append([
                review['comment'][:50] + '...' if len(review['comment']) > 50 else review['comment'],
                f"{review['probability']:.2f}",
                review['classification']
            ])
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(table)
        doc.build(elements)
        pdf_buffer.seek(0)
        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name='sentiment_analysis_report.pdf'
        )
    except Exception as e:
        logger.error(f"Error generating PDF report: {e}")
        flash(f'Error generating PDF report: {str(e)}', 'error')
        return redirect(request.referrer or url_for('dashboard'))

@app.route('/dashboard')
@login_required
def dashboard():
    try:
        conn = psycopg2.connect(**db_config)
        df = pd.read_sql('SELECT * FROM reviews', conn)
        total_reviews = len(reviewed_comments)
        extremist_count = sum(1 for review in reviewed_comments if review['classification'] == 'EXTREMIST')
        moderate_count = sum(1 for review in reviewed_comments if review['classification'] == 'MODERATE')
        accurate_count = sum(1 for review in reviewed_comments if review['classification'] == 'ACCURATE')
        data = {
            'total_reviews': total_reviews,
            'extremist_count': extremist_count,
            'moderate_count': moderate_count,
            'accurate_count': accurate_count,
            'dataset_size': len(df) if not df.empty else 0
        }
        return render_template('dashboard.html', data=data, username=session.get('username'))
    except psycopg2.Error as e:
        logger.error(f"Database error during dashboard: {e}")
        flash('Database error, please try again later', 'error')
        return redirect(url_for('upload'))
    finally:
        conn.close()

@app.route('/dataset_metrics')
@login_required
def dataset_metrics():
    try:
        conn = psycopg2.connect(**db_config)
        df = pd.read_sql('SELECT * FROM reviews', conn)
        if df.empty:
            return render_template('dataset_metrics.html', data={'positive_percentage': 0.0, 'neutral_percentage': 0.0, 'negative_percentage': 0.0}, username=session.get('username'))
        df['label_mapped'] = df['label'].str.lower().apply(
            lambda x: 'Positive' if x in ['positive', 'accurate'] else 'Neutral' if x == 'neutral' else 'Negative'
        )
        total_reviews = len(df)
        positive_count = len(df[df['label_mapped'] == 'Positive'])
        neutral_count = len(df[df['label_mapped'] == 'Neutral'])
        negative_count = len(df[df['label_mapped'] == 'Negative'])
        positive_percentage = (positive_count / total_reviews) * 100 if total_reviews > 0 else 0.0
        neutral_percentage = (neutral_count / total_reviews) * 100 if total_reviews > 0 else 0.0
        negative_percentage = (negative_count / total_reviews) * 100 if total_reviews > 0 else 0.0
        data = {
            'positive_percentage': positive_percentage,
            'neutral_percentage': neutral_percentage,
            'negative_percentage': negative_percentage
        }
        return render_template('dataset_metrics.html', data=data, username=session.get('username'))
    except psycopg2.Error as e:
        logger.error(f"Database error during dataset metrics: {e}")
        flash('Database error, please try again later', 'error')
        return redirect(url_for('upload'))
    finally:
        conn.close()

@app.route('/review_history')
@login_required
def review_history():
    global reviewed_comments
    return render_template('review_history.html', reviews=reviewed_comments, username=session.get('username'))

@app.route('/extremist_analysis')
@login_required
def extremist_analysis():
    global reviewed_comments
    if not reviewed_comments:
        flash('No reviews available for extremist group analysis', 'error')
        return redirect(url_for('review'))
    total_reviews = len(reviewed_comments)
    extremist_count = sum(1 for review in reviewed_comments if review['classification'] == 'EXTREMIST')
    extremist_percentage = (extremist_count / total_reviews) * 100 if total_reviews > 0 else 0
    if extremist_percentage > 70:
        likelihood = "High"
        likelihood_percentage = 85
    elif extremist_percentage > 40:
        likelihood = "Moderate"
        likelihood_percentage = 60
    elif extremist_percentage > 10:
        likelihood = "Low"
        likelihood_percentage = 30
    else:
        likelihood = "Very Low"
        likelihood_percentage = 5
    analysis_result = {
        'total_reviews': total_reviews,
        'extremist_count': extremist_count,
        'extremist_percentage': round(extremist_percentage, 2),
        'likelihood': likelihood,
        'likelihood_percentage': likelihood_percentage
    }
    return render_template('extremist_analysis.html', analysis=analysis_result, username=session.get('username'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)), debug=True)
