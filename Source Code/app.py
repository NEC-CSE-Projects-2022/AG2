import os
import torch
import torch.nn as nn
import numpy as np
import sqlite3
from flask import Flask, render_template, request, redirect, url_for, session, flash, g
import joblib

# ==============================
# Model Definition
# ==============================
class CNN_BiLSTM_Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, cnn_out_channels=32, kernel_size=3):
        super(CNN_BiLSTM_Attention, self).__init__()
        self.cnn = nn.Conv1d(in_channels=1, out_channels=cnn_out_channels,
                             kernel_size=kernel_size, padding=kernel_size // 2)
        self.bi_lstm = nn.LSTM(input_size=cnn_out_channels, hidden_size=hidden_dim,
                               bidirectional=True, batch_first=True)
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, 3)  # 3 classes

    def forward(self, x):
        x = x.unsqueeze(1)                 # (batch, 1, features)
        x = self.cnn(x)                     # (batch, channels, features)
        x = x.permute(0, 2, 1)              # (batch, features, channels)
        lstm_out, _ = self.bi_lstm(x)
        attn_weights = torch.softmax(self.attn(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        output = self.fc(context)
        return output

# ==============================
# Flask Setup
# ==============================
app = Flask(__name__, template_folder="templates")
app.secret_key = "supersecretkey"  # needed for sessions

import time

@app.context_processor
def inject_version():
    return dict(version=time.time())

DATABASE = "users.db"
MODEL_PATH = "cnn_bilstm_attention().pth"
SCALER_PATH = "scaler.pkl"
INPUT_DIM = 53
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# Database Helper
# ==============================
def get_db():
    db = getattr(g, "_database", None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
    return db

def init_db():
    with app.app_context():
        db = get_db()
        cursor = db.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        """)
        db.commit()
@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, "_database", None)
    if db:
        db.close()
# ==============================
# Load Model & Scaler
# ==============================
model = CNN_BiLSTM_Attention(input_dim=INPUT_DIM).to(device)
model_loaded = False

if os.path.exists(MODEL_PATH):
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        model_loaded = True
        print("Model loaded from", MODEL_PATH)
        

    except Exception as e:
        print("Warning: failed to load model:", e)
else:
    print(f"Warning: model file '{MODEL_PATH}' not found. Using untrained model.")

scaler = None
if os.path.exists(SCALER_PATH):
    try:
        scaler = joblib.load(SCALER_PATH)
        print("Scaler loaded from", SCALER_PATH)
    except Exception as e:
        print("Warning: failed to load scaler:", e)

# ==============================
# Preprocess Input
# ==============================
def preprocess_user_input(form):
    # Existing features
    weather = form.get("Weather", "Clear")
    try:
        speed = float(form.get("Speed", "0"))
    except ValueError:
        speed = 0.0
    day = form.get("DayOfWeek", "Monday")
    light = form.get("LightCondition", "Daylight")
    road = form.get("RoadSurface", "Dry")

    features = np.zeros(INPUT_DIM, dtype=float)
    
    # Mappings
    weather_map = {"Clear": 0, "Rainy": 1, "Snowy": 2}
    day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
    light_map = {"Daylight": 0, "Dark": 1}
    road_map = {"Dry": 0, "Wet": 1, "Snow/Ice": 2}

    features[0] = weather_map.get(weather, 0)
    features[1] = speed
    features[2] = day_map.get(day, 0)
    features[3] = light_map.get(light, 0)
    features[4] = road_map.get(road, 0)

    if scaler:
        try:
            features = scaler.transform(features.reshape(1, -1))[0]
        except Exception as e:
            print("Scaler transform failed:", e)

    return features.reshape(1, -1)

# ==============================
# Fallback Predict
# ==============================
def validate_input(form):
    """Basic validation to detect unrelated or malformed inputs.
    Returns (is_valid: bool, message: str).
    """
    # Required categorical maps must match those used in preprocess
    weather_map = {"Clear": 0, "Rainy": 1, "Snowy": 2}
    day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
               "Friday": 4, "Saturday": 5, "Sunday": 6}
    light_map = {"Daylight": 0, "Dark": 1}
    road_map = {"Dry": 0, "Wet": 1, "Snow/Ice": 2}

    # Check speed
    try:
        if float(form.get("Speed", "")) < 0: return False, "Speed must be positive."
    except: return False, "Speed must be a number."

    # Check categories (protect against tampered input)
    if form.get("Weather", "") not in weather_map: return False, "Weather value not recognized."
    if form.get("DayOfWeek", "") not in day_map: return False, "Day value not recognized."
    if form.get("LightCondition", "") not in light_map: return False, "Light value not recognized."
    if form.get("RoadSurface", "") not in road_map: return False, "Road value not recognized."

    # Passed basic checks
    return True, ""

def fallback_predict(features):
    speed = float(features[0, 1])
    if speed >= 80:
        probs = np.array([0.05, 0.20, 0.75])
    elif speed >= 50:
        probs = np.array([0.10, 0.75, 0.15])
    elif speed >= 20:
        probs = np.array([0.70, 0.25, 0.05])
    else:
        probs = np.array([0.85, 0.14, 0.01])
    return probs

# ==============================
# Initialize database for predictions
# ==============================
def init_prediction_db():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        input_data TEXT NOT NULL,
                        prediction TEXT NOT NULL,
                        confidence REAL,
                        source_file TEXT, 
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )''')
    
    # Check if columns exist (migrations)
    cursor.execute("PRAGMA table_info(predictions)")
    columns = [info[1] for info in cursor.fetchall()]
    
    if "confidence" not in columns:
        cursor.execute("ALTER TABLE predictions ADD COLUMN confidence REAL")
    
    if "source_file" not in columns:
        cursor.execute("ALTER TABLE predictions ADD COLUMN source_file TEXT")
        
    conn.commit()
    conn.close()

# Initialize the prediction database
init_prediction_db()
# ==============================
# Helper for History Display 
# ==============================
def format_input_summary(form_data):
    """
    Returns a readable string summary of the input features.
    Example: "Speed: 85 | Weather: Rainy | Light: Dark | Road: Wet"
    """
    speed = form_data.get("Speed", "0")
    weather = form_data.get("Weather", "Unknown")
    day = form_data.get("DayOfWeek", "Unknown")
    light = form_data.get("LightCondition", "Unknown")
    road = form_data.get("RoadSurface", "Unknown")
    
    return f"Speed: {speed} | Weather: {weather} | Light: {light} | Road: {road} | Day: {day}"

# ==============================
# Routes
# ==============================
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/Result")
def Result():
    return render_template("Result.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        db = get_db()
        cursor = db.cursor()
        try:
            cursor.execute("INSERT INTO users(username, password) VALUES (?,?)", (username, password))
            db.commit()
            flash("Registration successful! Please login.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Username already exists!", "danger")
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        db = get_db()
        cursor = db.cursor()
        cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        user = cursor.fetchone()
        if user:
            session["logged_in"] = True
            session["user"] = username
            flash("Login successful!", "success")
            return redirect(url_for("index"))
        else:
            flash("Invalid credentials!", "danger")
    return render_template("login.html")
@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("home"))
@app.route("/index")
def index():
    if not session.get("logged_in"):
        flash("Please login first.", "warning")
        return redirect(url_for("login"))
    return render_template("index.html")
@app.route("/history")
def history():
    if not session.get("logged_in"):
        flash("Please login first.", "warning")
        return redirect(url_for("login"))
    
    # Fetch all predictions to display on the history page
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    # Ensure we select source_file as well (it might be NULL for old records)
    cursor.execute('SELECT input_data, prediction, confidence, timestamp, source_file FROM predictions ORDER BY timestamp DESC')
    predictions = cursor.fetchall()
    conn.close()

    return render_template("History.html", predictions=predictions)

@app.route("/predict", methods=["POST"])
def predict():
    if not session.get("logged_in"):
        flash("Please login first.", "warning")
        return redirect(url_for("login"))
    try:
        # Validate input first to avoid answering unrelated/malformed data
        is_valid, message = validate_input(request.form)
        if not is_valid:
            flash(f"Input rejected: {message}", "warning")
            return render_template("index.html")

        X = preprocess_user_input(request.form)
        
        # Create readable summary
        input_summary = format_input_summary(request.form)
        
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        if not model_loaded:
            probs = fallback_predict(X)
            pred_class = int(np.argmax(probs))
        else:
            with torch.no_grad():
                output = model(X_tensor)
                probs = torch.softmax(output, dim=1).cpu().numpy()[0]
                pred_class = int(np.argmax(probs))
                if np.ptp(probs) < 1e-4:
                    probs = fallback_predict(X)
                    pred_class = int(np.argmax(probs))
        # Confidence threshold: if model isn't confident, don't return a hard prediction
        max_prob = float(np.max(probs))
        classes = ["Slight", "Serious", "Fatal"]
        CONFIDENCE_THRESHOLD = 0.60

        if max_prob < CONFIDENCE_THRESHOLD:
            # Low confidence — treat as unrelated / ambiguous
            flash((f"Model not confident enough to give a prediction (confidence={max_prob:.2f}). "
                   "Please check your input or try a related example."), "warning")
            # Optionally show probabilities for transparency, but don't show the class
            return render_template("index.html", probabilities=probs.tolist())

        result = classes[pred_class]
        confidence_percent = round(max_prob * 100, 2)

        # Save prediction to database
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('INSERT INTO predictions (input_data, prediction, confidence, source_file) VALUES (?, ?, ?, ?)',
                       (input_summary, result, confidence_percent, "Manual Input"))
        conn.commit()
        conn.close()

        return render_template("prediction_result.html",
                               prediction=result,
                               confidence=f"{confidence_percent}"
                               )
    except Exception as e:
        return render_template("index.html",
                               prediction_text=f"Error: {str(e)}")

# ==============================
# Bulk Prediction
# ==============================
import pandas as pd

@app.route("/bulk_predict", methods=["POST"])
def bulk_predict():
    if not session.get("logged_in"):
        flash("Please login first.", "warning")
        return redirect(url_for("login"))

    files = request.files.getlist("file")
    if not files or (len(files) == 1 and files[0].filename == ""):
        flash("No files selected", "warning")
        return redirect(url_for("index"))

    total_records = 0
    processed_files = 0
    
    # Connect to DB once
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    try:
        for file in files:
            if file.filename == "":
                continue
                
            source_filename = file.filename # This often contains relative path with directory
            
            if file.filename.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.filename.endswith('.xlsx'):
                df = pd.read_excel(file)
            else:
                # Skip non-data files silently or log warning
                continue
            
            processed_files += 1

            for index, row in df.iterrows():
                form_data = {
                    "Speed": str(row.get("Speed", 0)),
                    "Weather": row.get("Weather", "Clear"),
                    "DayOfWeek": row.get("DayOfWeek", "Monday"),
                    "LightCondition": row.get("LightCondition", "Daylight"),
                    "RoadSurface": row.get("RoadSurface", "Dry")
                }
                
                # Preprocess
                X = preprocess_user_input(form_data)
                
                # Create readable summary
                input_summary = format_input_summary(form_data)

                X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
                
                # Predict
                if not model_loaded:
                    probs = fallback_predict(X)
                else:
                    with torch.no_grad():
                        output = model(X_tensor)
                        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
                        if np.ptp(probs) < 1e-4:
                            probs = fallback_predict(X)
                
                pred_class = int(np.argmax(probs))
                classes = ["Slight", "Serious", "Fatal"]
                prediction = classes[pred_class]
                max_prob = float(np.max(probs))
                confidence_percent = round(max_prob * 100, 2)
                
                # Save to DB
                cursor.execute('INSERT INTO predictions (input_data, prediction, confidence, source_file) VALUES (?, ?, ?, ?)',
                            (input_summary, prediction, confidence_percent, source_filename))
                
                total_records += 1

        conn.commit()
    except Exception as e:
        flash(f"Error processing files: {str(e)}", "danger")
        if conn:
            conn.close()
        return redirect(url_for("index"))
    finally:
        if conn:
            conn.close()
    
    if processed_files > 0:
        flash(f"Processed {total_records} records from {processed_files} files.", "success")
        return redirect(url_for("history"))
    else:
         flash("No valid CSV/Excel files found in selection.", "warning")
         return redirect(url_for("index"))

# ==============================
# Main
# ==============================
if __name__ == "__main__":
    init_db()
    torch.set_num_threads(1)
    app.run(debug=True)

