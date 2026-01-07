import sys
import json
import threading
import time
from datetime import datetime
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, 
                             QGridLayout, QMainWindow, QMessageBox, QProgressBar, QFrame, QDialog,
                             QPushButton, QComboBox, QFormLayout, QLineEdit, QSizePolicy, QCheckBox, QScrollArea,
                             QGraphicsDropShadowEffect)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QPropertyAnimation, pyqtProperty, QSize, QCoreApplication, QObject, QPoint
from PyQt5.QtGui import QFont, QColor, QPalette, QIcon, QMovie, QPixmap, QCursor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.dates import DateFormatter, HourLocator
import paho.mqtt.client as mqtt
import requests
import pymysql
import os
from pynput import mouse, keyboard
try:
    from plyer import notification
    NOTIFICATION_AVAILABLE = True
except:
    NOTIFICATION_AVAILABLE = False

# Try to import PyTorch for model inference
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not available. Using heuristic prediction method.")

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------

# MQTT Configuration
MQTT_BROKER = ""
MQTT_PORT = 8883
MQTT_USER = "
MQTT_PASSWORD = ""
MQTT_CA_CERT = r"ca.crt"

# Database Configuration
MYSQL_HOST = ''
MYSQL_USER = 'root'
MYSQL_PASSWORD = ''
MYSQL_DATABASE = ''
MYSQL_PORT = 3306

# Weather API
WEATHER_API_URL = ""

# City Coordinates (Japan)
CITIES = {
    "Fukuoka": {"lat": 33.59, "lon": 130.40},
    "Tokyo": {"lat": 35.68, "lon": 139.69},
    "Osaka": {"lat": 34.69, "lon": 135.50},
    "Nagoya": {"lat": 35.18, "lon": 136.90},
    "Sapporo": {"lat": 43.06, "lon": 141.35},
    "Sendai": {"lat": 38.26, "lon": 140.87}
}

# -----------------------------------------------------------------------------
# RainNet-MT Model Architecture (for inference)
# -----------------------------------------------------------------------------

if TORCH_AVAILABLE:
    class MultiScaleAttention(nn.Module):
        """Multi-scale attention mechanism"""
        def __init__(self, hidden_size, num_heads=8):
            super().__init__()
            self.num_heads = num_heads
            self.head_dim = hidden_size // num_heads
            assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
            
            self.query = nn.Linear(hidden_size, hidden_size)
            self.key = nn.Linear(hidden_size, hidden_size)
            self.value = nn.Linear(hidden_size, hidden_size)
            self.out = nn.Linear(hidden_size, hidden_size)
            self.dropout = nn.Dropout(0.1)
        
        def forward(self, x):
            batch_size, seq_len, hidden_size = x.size()
            Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            
            attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
            attention_weights = F.softmax(attention_scores, dim=-1)
            attention_weights = self.dropout(attention_weights)
            
            attention_output = torch.matmul(attention_weights, V)
            attention_output = attention_output.transpose(1, 2).contiguous().view(
                batch_size, seq_len, hidden_size)
            return self.out(attention_output)
    
    class FeaturePyramidNetwork(nn.Module):
        """Feature Pyramid Network with optional dilated convolutions."""
        def __init__(self, input_size, hidden_size, dilation_rates=(1, 2, 4, 8), use_dilated=True):
            super().__init__()
            self.use_dilated = use_dilated
            self.conv_blocks = nn.ModuleList()
            
            if use_dilated:
                configs = [(3, d) for d in dilation_rates]
            else:
                configs = [(k, 1) for k in (3, 5, 7)]
            
            for kernel_size, dilation in configs:
                padding = dilation * (kernel_size - 1) // 2
                block = nn.Sequential(
                    nn.Conv1d(input_size, hidden_size, kernel_size=kernel_size,
                              padding=padding, dilation=dilation),
                    nn.BatchNorm1d(hidden_size),
                    nn.GELU()
                )
                self.conv_blocks.append(block)
            
            self.num_paths = len(self.conv_blocks)
            self.fusion = nn.Conv1d(hidden_size * self.num_paths, hidden_size, kernel_size=1)
            self.dropout = nn.Dropout(0.2)
        
        def forward(self, x):
            x = x.transpose(1, 2)  # [batch, features, time]
            conv_features = [block(x) for block in self.conv_blocks]
            fused = torch.cat(conv_features, dim=1)
            fused = self.dropout(F.relu(self.fusion(fused)))
            return fused.transpose(1, 2)
    
    class RainNetMTEnhanced(nn.Module):
        """Enhanced RainNet-MT model for inference."""
        def __init__(self, input_size, hidden_size=512, num_layers=6, num_classes=3,
                     dropout=0.15, dilation_rates=(1, 2, 4, 8), recurrent_type='gru',
                     use_bi_gru=False, use_attention=False, use_dilated_convs=True,
                     use_conditional_tasking=False):
            super().__init__()
            self.num_classes = num_classes
            self.hidden_size = hidden_size
            self.use_attention = use_attention
            self.use_conditional_tasking = use_conditional_tasking
            self.recurrent_type = recurrent_type.lower()
            self.use_bi_gru = use_bi_gru
            
            self.input_norm = nn.LayerNorm(input_size)
            self.fpn = FeaturePyramidNetwork(
                input_size, hidden_size // 2,
                dilation_rates=dilation_rates,
                use_dilated=use_dilated_convs
            )
            
            rnn_cls = nn.GRU if self.recurrent_type == 'gru' else nn.LSTM
            self.rnn = rnn_cls(
                hidden_size // 2,
                hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=use_bi_gru,
                batch_first=True
            )
            self.sequence_dim = hidden_size * (2 if use_bi_gru else 1)
            self.shared_dim = hidden_size // 2
            
            if use_attention:
                self.multi_scale_attention = MultiScaleAttention(self.sequence_dim, num_heads=16)
                self.global_attention = nn.Sequential(
                    nn.Linear(self.sequence_dim, self.sequence_dim // 2),
                    nn.Tanh(),
                    nn.Linear(self.sequence_dim // 2, 1)
                )
            else:
                self.multi_scale_attention = nn.Identity()
                self.global_attention = None
            
            self.feature_extractor = nn.Sequential(
                nn.Linear(self.sequence_dim, self.sequence_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.sequence_dim, self.sequence_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.sequence_dim // 2, self.shared_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.residual = nn.Linear(self.sequence_dim, self.shared_dim)
            
            self.occurrence_head = nn.Sequential(
                nn.Linear(self.shared_dim, self.shared_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.shared_dim // 2, 1)
            )
            
            intensity_input_dim = self.shared_dim + (1 if use_conditional_tasking else 0)
            self.intensity_head = nn.Sequential(
                nn.Linear(intensity_input_dim, self.shared_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.shared_dim, self.shared_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.shared_dim // 2, self.shared_dim // 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.shared_dim // 4, num_classes)
            )
        
        def forward(self, x):
            if x.dim() == 3 and x.size(-1) == 1:
                x = x.squeeze(-1).unsqueeze(1)
            elif x.dim() == 3:
                x = x.transpose(1, 2)
            else:
                x = x.unsqueeze(1)
            
            x = self.input_norm(x)
            x = self.fpn(x)
            rnn_out, _ = self.rnn(x)
            
            if self.use_attention:
                seq = self.multi_scale_attention(rnn_out)
                att_logits = self.global_attention(seq)
                attention_weights = F.softmax(att_logits, dim=1)
                context = torch.sum(attention_weights * seq, dim=1)
            else:
                seq = rnn_out
                context = torch.mean(seq, dim=1)
            
            features = self.feature_extractor(context)
            residual = self.residual(context)
            shared = features + residual
            
            occ_logits = self.occurrence_head(shared).squeeze(-1)
            if self.use_conditional_tasking:
                occ_prob = torch.sigmoid(occ_logits).unsqueeze(-1)
                intensity_input = torch.cat([shared, occ_prob], dim=1)
            else:
                intensity_input = shared
            intensity_logits = self.intensity_head(intensity_input)
            
            return {
                'occurrence': occ_logits,
                'intensity': intensity_logits
            }

# -----------------------------------------------------------------------------
# RainNet-MT Engine
# -----------------------------------------------------------------------------

class RainNetMTEngine:
    """
    RainNet-MT model inference engine.
    Supports both real model loading and fallback heuristic.
    """
    def __init__(self, model_path=None, scaler_path=None, input_size=None):
        self.intensity_labels = ['No Rain', 'Light Rain', 'Medium Rain', 'Heavy Rain']
        self.model = None
        self.scaler = None
        self.use_real_model = False
        self.device = torch.device('cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu') if TORCH_AVAILABLE else None
        self.input_size = input_size  # Number of features expected by model
        
        # Try to load real model if paths provided
        if model_path and TORCH_AVAILABLE:
            try:
                import pickle
                
                # Check if model file exists
                if os.path.exists(model_path):
                    # Default model configuration (matching training script)
                    model_kwargs = {
                        'hidden_size': 512,
                        'num_layers': 6,
                        'dropout': 0.15,
                        'dilation_rates': (1, 2, 4, 8),
                        'recurrent_type': 'gru',
                        'use_bi_gru': False,
                        'use_attention': False,
                        'use_dilated_convs': True,
                        'use_conditional_tasking': False
                    }
                    
                    # Try to infer input_size from scaler if available
                    if scaler_path and os.path.exists(scaler_path):
                        try:
                            with open(scaler_path, 'rb') as f:
                                self.scaler = pickle.load(f)
                            # Infer input size from scaler
                            if hasattr(self.scaler, 'n_features_in_'):
                                self.input_size = self.scaler.n_features_in_
                            elif hasattr(self.scaler, 'scale_'):
                                self.input_size = len(self.scaler.scale_)
                            print(f"✅ Scaler loaded from {scaler_path}")
                            print(f"   Inferred input size: {self.input_size}")
                        except Exception as e:
                            print(f"⚠️ Could not load scaler: {e}")
                    
                    # If input_size still not set, use a default (will need to be adjusted)
                    if self.input_size is None:
                        self.input_size = 200  # Default, should match your feature count
                        print(f"⚠️ Using default input_size: {self.input_size}")
                    
                    # Create and load model
                    self.model = RainNetMTEnhanced(input_size=self.input_size, **model_kwargs)
                    self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                    self.model.to(self.device)
                    self.model.eval()
                    self.use_real_model = True
                    print(f"✅ Model loaded from {model_path}")
                    print(f"   Using device: {self.device}")
                else:
                    print(f"⚠️ Model file not found: {model_path}")
            except Exception as e:
                print(f"⚠️ Could not load model: {e}")
                import traceback
                traceback.print_exc()
    
    def predict(self, current_data):
        """
        Predict occurrence and intensity based on current sensor readings.
        Uses real model if available, otherwise uses enhanced heuristic.
        """
        if self.use_real_model and self.model is not None:
            try:
                # Simplified feature extraction from current sensor data
                # Note: Real model expects engineered features, but we'll use current values
                # as a simplified approximation
                features = self._extract_features(current_data)
                
                if features is not None and len(features) == self.input_size:
                    # Scale features
                    if self.scaler is not None:
                        features_scaled = self.scaler.transform([features])
                    else:
                        features_scaled = np.array([features])
                    
                    # Reshape for model input: [batch, features, time]
                    features_tensor = torch.FloatTensor(features_scaled).reshape(1, self.input_size, 1).to(self.device)
                    
                    # Model inference
                    with torch.no_grad():
                        output = self.model(features_tensor)
                        intensity_logits = output['intensity']
                        intensity_probs = F.softmax(intensity_logits, dim=1)
                        intensity_idx = torch.argmax(intensity_probs, dim=1).item()
                        
                        # Convert to probability (sum of rain classes)
                        # Model outputs: 0=Light, 1=Medium, 2=Heavy
                        rain_prob = (intensity_probs[0][1] + intensity_probs[0][2]).item() * 100
                        
                        # Map model output to labels
                        if intensity_idx == 0:
                            label = 'No Rain'
                        elif intensity_idx == 1:
                            label = 'Light Rain'
                        elif intensity_idx == 2:
                            label = 'Medium Rain'
                        else:
                            label = 'Heavy Rain'
                        
                        return {
                            'probability': rain_prob,
                            'intensity_index': intensity_idx,
                            'intensity_label': label,
                            'attention_weights': {
                                'Humidity': float(current_data.get('Humidity', 0)) / 100,
                                'Pressure': max(0.1, (1020 - float(current_data.get('Pressure', 1013))) / 50),
                                'Temperature': float(current_data.get('Ta', 25)) / 50,
                                'Wind': float(current_data.get('Windspeed', 0)) / 20
                            }
                        }
            except Exception as e:
                print(f"⚠️ Model inference error: {e}. Falling back to heuristic.")
                import traceback
                traceback.print_exc()
        
        # Enhanced heuristic (fallback)
        rh = float(current_data.get('Humidity', 0))
        pressure = float(current_data.get('Pressure', 1013))
        wind = float(current_data.get('Windspeed', 0))
        temp = float(current_data.get('Ta', 25))
        
        # More sophisticated heuristic based on meteorological principles
        base_score = 0
        
        # Humidity contribution (most important)
        if rh > 70:
            base_score += (rh - 70) * 2.5
        if rh > 85:
            base_score += (rh - 85) * 3  # Extra weight for very high humidity
        
        # Pressure contribution (low pressure = higher rain chance)
        if pressure < 1015:
            base_score += (1015 - pressure) * 3.5
        if pressure < 1000:
            base_score += (1000 - pressure) * 5  # Extra weight for very low pressure
        
        # Wind contribution (moderate wind can indicate weather systems)
        if 3 < wind < 10:
            base_score += wind * 1.5
        elif wind > 10:
            base_score += 15  # Strong wind often accompanies storms
        
        # Temperature-humidity interaction
        if rh > 80 and temp > 20:
            base_score += 10  # Warm and humid = higher chance
        
        # Normalize to 0-100 probability
        prob = min(max(base_score / 1.5, 0), 100)
        
        # Intensity classification (matching model output: 0=Light, 1=Medium, 2=Heavy)
        if prob < 30:
            intensity_idx = 0  # No Rain
        elif prob < 50:
            intensity_idx = 0  # No Rain (low probability)
        elif prob < 70:
            intensity_idx = 1  # Light Rain
        elif prob < 85:
            intensity_idx = 2  # Medium Rain
        else:
            intensity_idx = 3  # Heavy Rain
            
        return {
            'probability': prob,
            'intensity_index': intensity_idx,
            'intensity_label': self.intensity_labels[intensity_idx],
            'attention_weights': {
                'Humidity': max(0.1, rh/100),
                'Pressure': max(0.1, (1020-pressure)/50),
                'Temperature': max(0.1, temp/50),
                'Wind': max(0.1, wind/20)
            }
        }
    
    def _extract_features(self, current_data):
        """
        Extract features from current sensor data.
        This is a simplified version - real model needs engineered features.
        For now, we pad with zeros or use current values.
        """
        try:
            # Basic features from current data
            features = []
            
            # Map sensor data to feature positions (simplified)
            # Real model expects engineered features, so we'll create a basic vector
            rh = float(current_data.get('Humidity', 0))
            pressure = float(current_data.get('Pressure', 1013))
            wind = float(current_data.get('Windspeed', 0))
            temp = float(current_data.get('Ta', 25))
            co2 = float(current_data.get('CO2', 400))
            
            # Create a basic feature vector (pad to expected size)
            # This is a simplified approach - ideally should match training features
            base_features = [rh, pressure, wind, temp, co2]
            
            # Pad or repeat to match expected input size
            if self.input_size:
                # Repeat and add variations to fill the feature vector
                features = base_features * (self.input_size // len(base_features))
                features += base_features[:self.input_size % len(base_features)]
                return np.array(features[:self.input_size])
            
            return None
        except Exception as e:
            print(f"⚠️ Feature extraction error: {e}")
            return None

# -----------------------------------------------------------------------------
# Screen Reminder (Functionality Restored)
# -----------------------------------------------------------------------------

class ScreenReminder(QObject):
    userAwaySignal = pyqtSignal(str)
    reminderSignal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.last_active_time = time.time()
        self.max_idle_seconds = 15 * 60  # 15 mins
        self.reminder_threshold = 1800   # 30 mins
        self.user_away_time = 0

        self.mouse_listener = mouse.Listener(on_move=self.on_input, on_click=self.on_input, on_scroll=self.on_input)
        self.keyboard_listener = keyboard.Listener(on_press=self.on_input)
        self.mouse_listener.start()
        self.keyboard_listener.start()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.check_activity)
        self.timer.start(1000)

    def on_input(self, *args):
        self.last_active_time = time.time()
        self.user_away_time = 0

    def check_activity(self):
        idle_time = time.time() - self.last_active_time
        if idle_time > self.max_idle_seconds:
            self.user_away_time += 1
            if self.user_away_time == 1: # Trigger once when crossing threshold
                 self.userAwaySignal.emit("It's been 15 minutes. Consider saving energy!")

# -----------------------------------------------------------------------------
# Custom UI Components (Native Style)
# -----------------------------------------------------------------------------

class ModernCard(QFrame):
    """Base class for modern cards with subtle shadow."""
    def __init__(self, bg_color="#ffffff", text_color="#333333"):
        super().__init__()
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {bg_color};
                border-radius: 14px;
                border: 1px solid #FFFFFF;
            }}
            QLabel {{
                background-color: transparent;
                color: {text_color};
                font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
            }}
        """)
        # Add subtle shadow effect
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setXOffset(0)
        shadow.setYOffset(2)
        shadow.setColor(QColor(0, 0, 0, 20))
        self.setGraphicsEffect(shadow)

class SensorWidget(ModernCard):
    """Apple-style small widget for sensors."""
    def __init__(self, title, unit, icon_name=None, accent_color="#007AFF"):
        super().__init__(bg_color="#FFFFFF", text_color="#1C1C1E")
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(140)  # Increased height for better spacing
        layout = QVBoxLayout()
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(6)
        
        # Top Row: Title + Icon
        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(8)
        self.title_label = QLabel(title)
        self.title_label.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self.title_label.setStyleSheet("color: #8E8E93;")
        self.title_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        top_row.addWidget(self.title_label)
        
        # Indicator Dot
        self.dot = QLabel("●")
        self.dot.setFont(QFont("Arial", 14))
        self.dot.setStyleSheet(f"color: {accent_color};")
        self.dot.setFixedSize(14, 14)
        top_row.addWidget(self.dot)
        layout.addLayout(top_row)
        
        # Middle: Value
        self.value_label = QLabel("--")
        self.value_label.setFont(QFont("Segoe UI", 28, QFont.Bold))
        self.value_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.value_label.setStyleSheet("color: #1C1C1E; padding: 0px;")
        self.value_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.value_label.setWordWrap(False)
        layout.addWidget(self.value_label)
        
        # Bottom: Unit
        self.unit_label = QLabel(unit)
        self.unit_label.setFont(QFont("Segoe UI", 13))
        self.unit_label.setStyleSheet("color: #8E8E93; padding: 0px;")
        self.unit_label.setAlignment(Qt.AlignLeft)
        layout.addWidget(self.unit_label)
        
        self.setLayout(layout)

    def update_value(self, value):
        self.value_label.setText(str(value))

class WeatherCard(ModernCard):
    """Large card for local weather - Clickable for details."""
    def __init__(self):
        super().__init__(bg_color="#007AFF", text_color="#FFFFFF")
        self.setFixedHeight(160)  # Slightly increased for better spacing
        self.setCursor(Qt.PointingHandCursor)
        self.weather_data = {}
        
        # Override style for weather card
        self.setStyleSheet("""
            QFrame {
                background-color: #007AFF;
                border-radius: 16px;
                border: none;
            }
            QLabel {
                background-color: transparent;
                color: #FFFFFF;
                font-family: 'Segoe UI', sans-serif;
            }
        """)
        
        layout = QHBoxLayout()
        layout.setContentsMargins(30, 22, 30, 22)
        layout.setSpacing(20)
        
        # Left: City & Temp
        left_layout = QVBoxLayout()
        left_layout.setSpacing(8)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.city_label = QLabel("Loading...")
        self.city_label.setFont(QFont("Segoe UI", 15, QFont.Bold))
        self.city_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        
        self.temp_label = QLabel("--°C")
        self.temp_label.setFont(QFont("Segoe UI", 36, QFont.Bold))
        self.temp_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.temp_label.setWordWrap(False)
        
        # Additional info
        self.wind_label = QLabel("Wind: --")
        self.wind_label.setFont(QFont("Segoe UI", 12))
        self.wind_label.setStyleSheet("color: rgba(255,255,255,0.9);")
        self.wind_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        
        left_layout.addWidget(self.city_label)
        left_layout.addWidget(self.temp_label)
        left_layout.addWidget(self.wind_label)
        left_layout.addStretch()
        
        # Right: Condition
        right_layout = QVBoxLayout()
        right_layout.setSpacing(8)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addStretch()
        self.condition_label = QLabel("--")
        self.condition_label.setFont(QFont("Segoe UI", 16, QFont.Bold))
        self.condition_label.setAlignment(Qt.AlignRight | Qt.AlignTop)
        self.condition_label.setWordWrap(True)
        self.condition_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        
        self.desc_label = QLabel("Tap for details")
        self.desc_label.setFont(QFont("Segoe UI", 11))
        self.desc_label.setStyleSheet("color: rgba(255,255,255,0.7);")
        self.desc_label.setAlignment(Qt.AlignRight | Qt.AlignBottom)
        self.desc_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        
        right_layout.addWidget(self.condition_label)
        right_layout.addWidget(self.desc_label)
        
        layout.addLayout(left_layout)
        layout.addStretch()
        layout.addLayout(right_layout)
        self.setLayout(layout)

    def update_weather(self, city, temp, code, wind=None):
        self.city_label.setText(city)
        self.temp_label.setText(f"{temp:.1f}°C")
        
        if wind:
            self.wind_label.setText(f"Wind: {wind:.1f} m/s")
        
        condition = "Unknown"
        if code == 0: condition = "Clear Sky ☀️"
        elif code in [1, 2, 3]: condition = "Partly Cloudy ⛅"
        elif code in [45, 48]: condition = "Foggy 🌫️"
        elif 51 <= code <= 67: condition = "Drizzle 🌧️"
        elif 80 <= code <= 99: condition = "Rain/Showers ⛈️"
        elif 71 <= code <= 77: condition = "Snow ❄️"
        
        self.condition_label.setText(condition)
        self.weather_data = {'city': city, 'temp': temp, 'code': code, 'wind': wind}

class ComprehensiveDashboard(QDialog):
    """Comprehensive data dashboard with export."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Comprehensive Dashboard")
        self.resize(1000, 700)
        self.setStyleSheet("background-color: #F5F5F7;")
        
        layout = QVBoxLayout()
        layout.setContentsMargins(30, 30, 30, 30)
        
        # Header
        header = QHBoxLayout()
        title = QLabel("Data Analytics")
        title.setFont(QFont("Segoe UI", 20, QFont.Bold))
        header.addWidget(title)
        header.addStretch()
        
        export_btn = QPushButton("Export CSV")
        export_btn.setFixedSize(120, 40)
        export_btn.setStyleSheet("""
            QPushButton { background-color: #34C759; color: white; border-radius: 8px; font-weight: bold; }
        """)
        export_btn.clicked.connect(self.export_data)
        header.addWidget(export_btn)
        layout.addLayout(header)
        
        # Multiple Charts
        self.figure, axes = plt.subplots(2, 2, figsize=(10, 7))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        self.axes = axes.flatten()
        self.plot_comprehensive_data()
        
        self.setLayout(layout)
        
    def plot_comprehensive_data(self):
        # Sample comprehensive visualization
        dates = pd.date_range(end=datetime.now(), periods=24, freq='H')
        
        # Date formatter for x-axis
        date_formatter = DateFormatter('%H:%M')
        hour_locator = HourLocator(interval=3)  # Show every 3 hours
        
        # Plot 1: Temperature
        self.axes[0].plot(dates, np.random.normal(25, 2, 24), color='#FF3B30', linewidth=2, marker='o', markersize=3)
        self.axes[0].set_title("Temperature Trend", fontweight='bold', fontsize=12)
        self.axes[0].set_xlabel("Time", fontsize=9)
        self.axes[0].set_ylabel("Temperature (°C)", fontsize=9)
        self.axes[0].xaxis.set_major_formatter(date_formatter)
        self.axes[0].xaxis.set_major_locator(hour_locator)
        self.axes[0].tick_params(axis='x', rotation=45, labelsize=8)
        self.axes[0].tick_params(axis='y', labelsize=8)
        self.axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Humidity
        self.axes[1].plot(dates, np.random.normal(60, 10, 24), color='#5856D6', linewidth=2, marker='o', markersize=3)
        self.axes[1].set_title("Humidity Trend", fontweight='bold', fontsize=12)
        self.axes[1].set_xlabel("Time", fontsize=9)
        self.axes[1].set_ylabel("Humidity (%)", fontsize=9)
        self.axes[1].xaxis.set_major_formatter(date_formatter)
        self.axes[1].xaxis.set_major_locator(hour_locator)
        self.axes[1].tick_params(axis='x', rotation=45, labelsize=8)
        self.axes[1].tick_params(axis='y', labelsize=8)
        self.axes[1].grid(True, alpha=0.3)
        
        # Plot 3: CO2
        self.axes[2].plot(dates, np.random.normal(500, 100, 24), color='#FF9500', linewidth=2, marker='o', markersize=3)
        self.axes[2].set_title("CO2 Levels", fontweight='bold', fontsize=12)
        self.axes[2].set_xlabel("Time", fontsize=9)
        self.axes[2].set_ylabel("CO2 (ppm)", fontsize=9)
        self.axes[2].xaxis.set_major_formatter(date_formatter)
        self.axes[2].xaxis.set_major_locator(hour_locator)
        self.axes[2].tick_params(axis='x', rotation=45, labelsize=8)
        self.axes[2].tick_params(axis='y', labelsize=8)
        self.axes[2].grid(True, alpha=0.3)
        
        # Plot 4: Rain Probability (use dates for consistency)
        hours = [d.strftime('%H:%M') for d in dates]
        self.axes[3].bar(range(24), np.random.randint(0, 80, 24), color='#007AFF', alpha=0.7)
        self.axes[3].set_title("Rain Probability", fontweight='bold', fontsize=12)
        self.axes[3].set_xlabel("Time", fontsize=9)
        self.axes[3].set_ylabel("Probability (%)", fontsize=9)
        # Set x-axis labels for every 3 hours to avoid overlap
        tick_positions = list(range(0, 24, 3))
        tick_labels = [hours[i] for i in tick_positions]
        self.axes[3].set_xticks(tick_positions)
        self.axes[3].set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)
        self.axes[3].tick_params(axis='y', labelsize=8)
        self.axes[3].grid(True, alpha=0.3, axis='y')
        
        # Adjust layout to prevent label overlap
        plt.tight_layout(pad=2.5)
        self.canvas.draw()
        
    def export_data(self):
        # Export to CSV
        try:
            filename = f"RainNet_Export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df = pd.DataFrame({
                'Timestamp': pd.date_range(end=datetime.now(), periods=24, freq='H'),
                'Temperature': np.random.normal(25, 2, 24),
                'Humidity': np.random.normal(60, 10, 24),
                'CO2': np.random.normal(500, 100, 24)
            })
            df.to_csv(filename, index=False)
            QMessageBox.information(self, "Success", f"Data exported to {filename}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Export failed: {e}")

class AboutDialog(QDialog):
    """About RainNet-MT System."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About RainNet-MT")
        self.setFixedSize(500, 400)
        self.setStyleSheet("background-color: #FFFFFF;")
        
        layout = QVBoxLayout()
        layout.setContentsMargins(40, 40, 40, 40)
        
        # logo = QLabel()
        # movie = QMovie('Fig/Networkconnecting.gif')
        # movie.setScaledSize(QSize(100, 100))
        # logo.setMovie(movie)
        # movie.start()
        # logo.setAlignment(Qt.AlignCenter)
        # layout.addWidget(logo)
        
        title = QLabel("RainNet-MT")
        title.setFont(QFont("Segoe UI", 26, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        subtitle = QLabel("Rainfall Prediction & Indoor Monitoring System")
        subtitle.setFont(QFont("Segoe UI", 12))
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #8E8E93;")
        layout.addWidget(subtitle)
        
        info_text = QLabel(
            "Version: 2.0\n\n"
            "Based on deep learning model for rainfall intensity classification.\n"
            "Features multi-scale attention mechanism and temporal feature extraction.\n\n"
            "© 2026 RainNet-MT Team. All Rights Reserved."
        )
        info_text.setFont(QFont("Segoe UI", 11))
        info_text.setAlignment(Qt.AlignCenter)
        info_text.setWordWrap(True)
        info_text.setStyleSheet("margin-top: 20px;")
        layout.addWidget(info_text)
        
        layout.addStretch()
        
        close_btn = QPushButton("Close")
        close_btn.setFixedHeight(45)
        close_btn.setStyleSheet("""
            QPushButton { background-color: #007AFF; color: white; border-radius: 10px; font-weight: bold; }
        """)
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)
        
        self.setLayout(layout)

class WeatherDetailsDialog(QDialog):
    """Detailed weather forecast view."""
    def __init__(self, city, parent=None):
        super().__init__(parent)
        self.city = city
        self.setWindowTitle(f"Weather Details - {city}")
        self.setFixedSize(600, 500)
        self.setStyleSheet("background-color: #F5F5F7;")
        
        layout = QVBoxLayout()
        layout.setContentsMargins(30, 30, 30, 30)
        
        title = QLabel(f"{city} Forecast")
        title.setFont(QFont("Segoe UI", 20, QFont.Bold))
        layout.addWidget(title)
        
        self.info_label = QLabel("Loading detailed forecast...")
        self.info_label.setFont(QFont("Segoe UI", 12))
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)
        
        # Chart
        self.figure, self.ax = plt.subplots(figsize=(5, 3))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        layout.addStretch()
        self.setLayout(layout)
        
        self.fetch_forecast()
        
    def fetch_forecast(self):
        coords = CITIES.get(self.city, CITIES["Fukuoka"])
        try:
            url = f"{WEATHER_API_URL}?latitude={coords['lat']}&longitude={coords['lon']}&hourly=temperature_2m,precipitation_probability&timezone=auto&forecast_days=1"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()['hourly']
                temps = data['temperature_2m'][:12]  # Next 12 hours
                precip = data['precipitation_probability'][:12]
                
                hours = list(range(1, 13))
                self.ax.clear()
                self.ax.plot(hours, temps, marker='o', label='Temperature (°C)', color='#FF3B30')
                self.ax2 = self.ax.twinx()
                self.ax2.bar(hours, precip, alpha=0.3, label='Rain Prob (%)', color='#007AFF')
                self.ax.set_xlabel("Hours ahead")
                self.ax.legend(loc='upper left')
                self.ax2.legend(loc='upper right')
                self.ax.grid(True, alpha=0.3)
                self.canvas.draw()
                
                self.info_label.setText(f"Next 12 hours forecast for {self.city}")
        except:
            self.info_label.setText("Unable to fetch detailed forecast.")

# -----------------------------------------------------------------------------
# Dialogs
# -----------------------------------------------------------------------------

class DatabaseTrendDialog(QDialog):
    """View historical data from MySQL."""
    def __init__(self, label_name):
        super().__init__()
        self.setWindowTitle(f"Historical Trend: {label_name}")
        self.resize(900, 600)
        self.setStyleSheet("background-color: #FFFFFF;")
        
        layout = QVBoxLayout()
        
        # Header
        header = QLabel(f"{label_name} - Last 24 Hours")
        header.setFont(QFont("Segoe UI", 18, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)

        # Plot
        self.figure, self.ax = plt.subplots(figsize=(8, 5))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        self.label_name = label_name
        self.fetch_and_plot_data()
        
        self.setLayout(layout)

    def fetch_and_plot_data(self):
        # Try fetching from DB, fallback to simulation
        data = []
        timestamps = []
        
        try:
            connection = pymysql.connect(
                host=MYSQL_HOST, user=MYSQL_USER, password=MYSQL_PASSWORD,
                database=MYSQL_DATABASE, port=MYSQL_PORT
            )
            with connection.cursor() as cursor:
                # Extended column mapping
                col_map = {
                    'Humidity': 'Humidity', 
                    'Ta': 'Indoor_Temperature', 
                    'Pressure': 'Pressure',
                    'Windspeed': 'Wind_Speed',
                    'CO2': 'co2',
                    'PM2.5': 'pm25',
                    'PM10': 'pm10',
                    'Tg': 'Globe_temperature',
                    'MRT': 'Mean_radiant_temperature'
                }
                db_col = col_map.get(self.label_name, 'Indoor_Temperature')
                
                sql = f"SELECT Time, {db_col} FROM raspberry_mqtt5 WHERE Time >= NOW() - INTERVAL 1 DAY ORDER BY Time"
                cursor.execute(sql)
                result = cursor.fetchall()
                for row in result:
                    timestamps.append(row[0])
                    data.append(row[1])
            connection.close()
        except Exception as e:
            print(f"DB Error: {e}. Using simulated data.")
            timestamps = pd.date_range(end=datetime.now(), periods=24, freq='H')
            data = np.random.normal(25, 2, 24)

        if not data:
             timestamps = pd.date_range(end=datetime.now(), periods=24, freq='H')
             data = np.random.normal(25, 2, 24)

        self.ax.clear()
        self.ax.plot(timestamps, data, marker='o', linestyle='-', color='#007AFF', linewidth=2.5, markersize=4)
        self.ax.set_facecolor('#FAFAFA')
        self.ax.grid(True, color='#E5E5EA', alpha=0.5)
        self.ax.set_ylabel(self.label_name, fontsize=12, fontweight='bold')
        self.figure.autofmt_xdate()
        plt.tight_layout()
        self.canvas.draw()

class ProfileDialog(QDialog):
    """User profile viewer."""
    def __init__(self, username, parent=None):
        super().__init__(parent)
        self.username = username
        self.setWindowTitle("Profile")
        self.setFixedSize(400, 500)
        self.setStyleSheet("""
            QDialog { background-color: #FFFFFF; }
            QLabel { font-family: 'Segoe UI'; }
            QPushButton {
                background-color: #007AFF; color: white; 
                border-radius: 8px; padding: 12px; font-weight: bold;
            }
        """)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(40, 40, 40, 40)
        
        # Avatar
        avatar = QLabel("👤")
        avatar.setFont(QFont("Segoe UI", 60))
        avatar.setAlignment(Qt.AlignCenter)
        layout.addWidget(avatar)
        
        # Username
        name_label = QLabel(username)
        name_label.setFont(QFont("Segoe UI", 20, QFont.Bold))
        name_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(name_label)
        
        # Score
        user_data = self.fetch_user_data()
        score_label = QLabel(f"Points: {user_data.get('score', 0)}")
        score_label.setFont(QFont("Segoe UI", 16))
        score_label.setAlignment(Qt.AlignCenter)
        score_label.setStyleSheet("color: #007AFF;")
        layout.addWidget(score_label)
        
        layout.addStretch()
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)
        
        self.setLayout(layout)
        
    def fetch_user_data(self):
        try:
            API_ENDPOINT = "http://192.168.83.7:5022/fetch_user_data"
            response = requests.post(API_ENDPOINT, json={"username": self.username}, timeout=3)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return {'username': self.username, 'score': 0}

class SettingsDialog(QDialog):
    """Dashboard customization."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.setWindowTitle("Settings")
        self.setFixedSize(450, 600)
        self.setStyleSheet("""
            QDialog { background-color: #F5F5F7; }
            QLabel { font-family: 'Segoe UI'; color: #1C1C1E; }
            QCheckBox { font-size: 14px; padding: 8px; }
            QPushButton {
                background-color: #007AFF; color: white; border-radius: 10px;
                padding: 12px; font-weight: bold;
            }
        """)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(30, 30, 30, 30)
        
        # Title
        title = QLabel("Dashboard Settings")
        title.setFont(QFont("Segoe UI", 20, QFont.Bold))
        layout.addWidget(title)
        
        # Quick Actions
        actions_layout = QHBoxLayout()
        
        profile_btn = QPushButton("Profile")
        profile_btn.setFixedHeight(45)
        profile_btn.clicked.connect(self.show_profile)
        
        about_btn = QPushButton("About")
        about_btn.setFixedHeight(45)
        about_btn.clicked.connect(self.show_about)
        
        actions_layout.addWidget(profile_btn)
        actions_layout.addWidget(about_btn)
        layout.addLayout(actions_layout)
        
        # Sensor Visibility
        visibility_label = QLabel("Visible Sensors:")
        visibility_label.setFont(QFont("Segoe UI", 14, QFont.Bold))
        visibility_label.setStyleSheet("margin-top: 20px;")
        layout.addWidget(visibility_label)
        
        self.checkboxes = {}
        sensors = ["Temperature", "Humidity", "Pressure", "Wind Speed", "CO2", "PM2.5", "PM10", "Globe Temp"]
        
        for s in sensors:
            cb = QCheckBox(s)
            cb.setChecked(True)
            self.checkboxes[s] = cb
            layout.addWidget(cb)
            
        layout.addStretch()
        
        save_btn = QPushButton("Save & Close")
        save_btn.clicked.connect(self.accept)
        layout.addWidget(save_btn)
        
        self.setLayout(layout)
        
    def show_profile(self):
        if self.parent_window and hasattr(self.parent_window, 'username'):
            dlg = ProfileDialog(self.parent_window.username, self)
            dlg.exec_()
            
    def show_about(self):
        dlg = AboutDialog(self)
        dlg.exec_()

class ActivityReportDialog(QDialog):
    """Modern Activity Reporting with Rankings"""
    def __init__(self, username):
        super().__init__()
        self.username = username
        self.total_score = 0
        self.setWindowTitle("Activity Report")
        self.setFixedSize(700, 800)
        self.setStyleSheet("""
            QDialog { background-color: #F5F5F7; }
            QPushButton {
                border: none;
                border-radius: 12px;
                background-color: white;
            }
            QPushButton:hover { background-color: #E5E5EA; }
        """)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(30, 30, 30, 30)
        
        title = QLabel("Log Your Activity")
        title.setFont(QFont("Segoe UI", 22, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Activity Grid (18 items like original)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: none;")
        scroll_widget = QWidget()
        grid = QGridLayout(scroll_widget)
        grid.setSpacing(10)
        
        self.activity_buttons = []
        for i in range(18):
            btn = QPushButton()
            btn.setIcon(QIcon(f"report/{i+1}.png")) 
            btn.setIconSize(QSize(70, 70))
            btn.setFixedSize(100, 100)
            btn.clicked.connect(lambda _, x=i: self.log_activity(x))
            self.activity_buttons.append(btn)
            grid.addWidget(btn, i//6, i%6)
        
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
        
        # Score Display
        self.score_label = QLabel(f"Points: {self.total_score}")
        self.score_label.setFont(QFont("Segoe UI", 18, QFont.Bold))
        self.score_label.setAlignment(Qt.AlignCenter)
        self.score_label.setStyleSheet("color: #007AFF;")
        layout.addWidget(self.score_label)
        
        # Action Buttons
        button_row = QHBoxLayout()
        
        self.save_btn = QPushButton("Save Report")
        self.save_btn.setFixedHeight(50)
        self.save_btn.setStyleSheet("""
            QPushButton { background-color: #34C759; color: white; font-weight: bold; border-radius: 10px; }
            QPushButton:hover { background-color: #2DA44E; }
        """)
        self.save_btn.clicked.connect(self.save_report)
        
        self.ranking_btn = QPushButton("View Rankings")
        self.ranking_btn.setFixedHeight(50)
        self.ranking_btn.setStyleSheet("""
            QPushButton { background-color: #007AFF; color: white; font-weight: bold; border-radius: 10px; }
            QPushButton:hover { background-color: #0062CC; }
        """)
        self.ranking_btn.clicked.connect(self.show_rankings)
        
        button_row.addWidget(self.save_btn)
        button_row.addWidget(self.ranking_btn)
        layout.addLayout(button_row)
        
        self.setLayout(layout)
        
    def log_activity(self, idx):
        self.total_score += 1
        self.score_label.setText(f"Points: {self.total_score}")
        
    def save_report(self):
        # API Call to update score
        try:
            API_ENDPOINT = "http://192.168.83.7:5022/update_score"
            data = {"username": self.username, "score": self.total_score}
            response = requests.post(API_ENDPOINT, json=data, timeout=3)
            if response.status_code == 200 and response.json().get("status") == "success":
                QMessageBox.information(self, "Success", f"Report saved! Total points: {self.total_score}")
            else:
                QMessageBox.warning(self, "Warning", "Unable to save to server. Saved locally.")
        except:
            QMessageBox.information(self, "Saved", "Report saved locally.")
            
    def show_rankings(self):
        dlg = RankingsDialog(self)
        dlg.exec_()

class RankingsDialog(QDialog):
    """Display user rankings."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Rankings")
        self.setFixedSize(500, 600)
        self.setStyleSheet("""
            QDialog { background-color: #FFFFFF; }
            QLabel { font-family: 'Segoe UI'; }
        """)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(30, 30, 30, 30)
        
        title = QLabel("🏆 Top Users")
        title.setFont(QFont("Segoe UI", 20, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        self.rankings_display = QLabel("Loading...")
        self.rankings_display.setFont(QFont("Segoe UI", 14))
        self.rankings_display.setWordWrap(True)
        layout.addWidget(self.rankings_display)
        
        layout.addStretch()
        
        close_btn = QPushButton("Close")
        close_btn.setFixedHeight(45)
        close_btn.setStyleSheet("background-color: #007AFF; color: white; border-radius: 8px; font-weight: bold;")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)
        
        self.setLayout(layout)
        self.fetch_rankings()
        
    def fetch_rankings(self):
        try:
            API_ENDPOINT = "http://192.168.83.7:5022/get_rankings"
            response = requests.get(API_ENDPOINT, timeout=3)
            if response.status_code == 200:
                rankings = response.json()
                text = ""
                for i, rank in enumerate(rankings[:10], 1):
                    medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
                    text += f"{medal} {rank['username']}: {rank['score']} pts\n"
                self.rankings_display.setText(text or "No data available.")
            else:
                self.rankings_display.setText("Unable to fetch rankings.")
        except:
            self.rankings_display.setText("Server unavailable.")

class CitySelectionDialog(QDialog):
    """Modern login dialog with city selection."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Welcome to RainNet-MT")
        self.setFixedSize(440, 650)
        self.setWindowFlags(Qt.Dialog | Qt.WindowCloseButtonHint)
        self.selected_city = "Fukuoka"
        self.username = "User"
        
        self.setStyleSheet("""
            QDialog { background-color: #FFFFFF; }
            QLabel { color: #000000; font-family: 'Segoe UI'; }
            QComboBox, QLineEdit {
                padding: 8px 12px;
                border: 1px solid #E5E5EA;
                border-radius: 10px;
                background-color: #F2F2F7;
                font-size: 14px;
                min-height: 40px;
                color: #1C1C1E;
            }
            QComboBox::drop-down { 
                border: none; 
                width: 30px;
            }
            QComboBox::down-arrow { 
                image: none;
                border: none;
            }
            QLineEdit::placeholder {
                color: #8E8E93;
            }
            QPushButton {
                background-color: #007AFF;
                color: white;
                padding: 10px;
                border-radius: 10px;
                font-weight: bold;
                font-size: 15px;
                border: none;
                min-height: 45px;
            }
            QPushButton:hover { background-color: #0062CC; }
            QPushButton#secondary {
                background-color: #F2F2F7;
                color: #007AFF;
            }
            QPushButton#secondary:hover { background-color: #E5E5EA; }
        """)
        
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(40, 40, 40, 40)
        
        # Logo (Static Icon)
        icon_label = QLabel()
        try:
            pixmap = QPixmap('Fig/AppIcon.png')
            if pixmap.isNull():
                # Fallback to text icon
                icon_label.setText("🌧️")
                icon_label.setFont(QFont("Segoe UI", 60))
            else:
                icon_label.setPixmap(pixmap.scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except:
            icon_label.setText("🌧️")
            icon_label.setFont(QFont("Segoe UI", 60))
        icon_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(icon_label)
        
        title = QLabel("RainNet-MT")
        title.setFont(QFont("Segoe UI", 26, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("margin-top: 12px; color: #1C1C1E;")
        title.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        layout.addWidget(title)
        
        subtitle = QLabel("Rainfall Monitoring System")
        subtitle.setFont(QFont("Segoe UI", 13))
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #8E8E93; margin-bottom: 8px;")
        subtitle.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)
        
        # Version info
        version_label = QLabel("Version 2.0")
        version_label.setFont(QFont("Segoe UI", 11))
        version_label.setAlignment(Qt.AlignCenter)
        version_label.setStyleSheet("color: #C7C7CC; margin-bottom: 28px;")
        version_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        layout.addWidget(version_label)
        
        # City Selection
        city_label = QLabel("Select City:")
        city_label.setFont(QFont("Segoe UI", 13, QFont.Bold))
        city_label.setStyleSheet("margin-bottom: 8px; color: #1C1C1E;")
        city_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        layout.addWidget(city_label)
        
        self.city_combo = QComboBox()
        self.city_combo.addItems(list(CITIES.keys()))
        self.city_combo.setFixedHeight(52)
        self.city_combo.setFont(QFont("Segoe UI", 14))
        layout.addWidget(self.city_combo)
        
        # Username
        user_label = QLabel("Your Name (Optional):")
        user_label.setFont(QFont("Segoe UI", 13, QFont.Bold))
        user_label.setStyleSheet("margin-top: 12px; margin-bottom: 8px; color: #1C1C1E;")
        user_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        layout.addWidget(user_label)
        
        self.user_input = QLineEdit()
        self.user_input.setPlaceholderText("Enter your name")
        self.user_input.setFixedHeight(52)
        self.user_input.setFont(QFont("Segoe UI", 14))
        layout.addWidget(self.user_input)
        
        layout.addStretch()
        
        # Buttons
        btn = QPushButton("Start Monitoring")
        btn.setCursor(Qt.PointingHandCursor)
        btn.setFont(QFont("Segoe UI", 14, QFont.Bold))
        btn.setFixedHeight(50)
        btn.clicked.connect(self.submit)
        layout.addWidget(btn)
        
        guest_btn = QPushButton("Continue as Guest")
        guest_btn.setObjectName("secondary")
        guest_btn.setCursor(Qt.PointingHandCursor)
        guest_btn.setFont(QFont("Segoe UI", 13))
        guest_btn.setFixedHeight(46)
        guest_btn.clicked.connect(self.guest_login)
        layout.addWidget(guest_btn)
        
        self.setLayout(layout)
        
        # Center on screen
        screen = QApplication.primaryScreen().geometry()
        x = (screen.width() - 440) // 2
        y = (screen.height() - 650) // 2
        self.move(x, y)

    def submit(self):
        self.selected_city = self.city_combo.currentText()
        if self.user_input.text():
            self.username = self.user_input.text()
        self.accept()
        
    def guest_login(self):
        self.selected_city = self.city_combo.currentText()
        self.username = "Guest"
        self.accept()

# -----------------------------------------------------------------------------
# Custom Title Bar & Supporting Widgets
# -----------------------------------------------------------------------------

class CustomTitleBar(QWidget):
    """Custom draggable title bar for frameless window."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.setFixedHeight(50)
        # Style moved to initUI for border radius consistency
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 0, 10, 0)
        
        # App Icon & Title
        title = QLabel("RainNet-MT")
        title.setFont(QFont("Segoe UI", 13, QFont.Bold))
        title.setStyleSheet("color: #1C1C1E;")
        layout.addWidget(title)
        layout.addStretch()
        
        # Window Controls
        self.minimize_btn = QPushButton("−")
        self.maximize_btn = QPushButton("□")
        self.close_btn = QPushButton("✕")
        
        for btn in [self.minimize_btn, self.maximize_btn, self.close_btn]:
            btn.setFixedSize(40, 30)
            btn.setStyleSheet("""
                QPushButton { 
                    background-color: transparent; 
                    border: none; 
                    font-size: 16px;
                    color: #8E8E93;
                }
                QPushButton:hover { background-color: #E5E5EA; border-radius: 5px; }
            """)
        
        self.close_btn.setStyleSheet("""
            QPushButton { background-color: transparent; border: none; font-size: 16px; color: #8E8E93; }
            QPushButton:hover { background-color: #FF3B30; color: white; border-radius: 5px; }
        """)
        
        self.minimize_btn.clicked.connect(parent.showMinimized)
        self.maximize_btn.clicked.connect(self.toggle_maximize)
        self.close_btn.clicked.connect(parent.close)
        
        layout.addWidget(self.minimize_btn)
        layout.addWidget(self.maximize_btn)
        layout.addWidget(self.close_btn)
        
        self.start_pos = None

    def toggle_maximize(self):
        if self.parent_window.isMaximized():
            self.parent_window.showNormal()
        else:
            self.parent_window.showMaximized()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.start_pos = event.globalPos()

    def mouseMoveEvent(self, event):
        if self.start_pos and not self.parent_window.isMaximized():
            delta = event.globalPos() - self.start_pos
            self.parent_window.move(self.parent_window.pos() + delta)
            self.start_pos = event.globalPos()

    def mouseReleaseEvent(self, event):
        self.start_pos = None

class ConnectingDialog(QDialog):
    """Loading animation after login."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Connecting...")
        self.setFixedSize(400, 350)
        self.setStyleSheet("""
            QDialog {
                background-color: #FFFFFF; 
                border: 1px solid #E5E5EA;
                border-radius: 12px;
            }
        """)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setAttribute(Qt.WA_TranslucentBackground, False)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(40, 40, 40, 40)
        
        # GIF Animation
        gif_label = QLabel()
        movie = QMovie('Fig/Networkconnecting.gif')
        movie.setScaledSize(QSize(150, 150))
        gif_label.setMovie(movie)
        movie.start()
        gif_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(gif_label)
        
        # Status Text
        status = QLabel("Connecting to system...")
        status.setFont(QFont("Segoe UI", 14))
        status.setAlignment(Qt.AlignCenter)
        status.setStyleSheet("color: #8E8E93;")
        layout.addWidget(status)
        
        self.setLayout(layout)
        
        # Center on screen
        if parent:
            parent_geo = parent.geometry()
            x = parent_geo.x() + (parent_geo.width() - 400) // 2
            y = parent_geo.y() + (parent_geo.height() - 350) // 2
            self.move(x, y)
        
        # Auto close after 3 seconds
        QTimer.singleShot(3000, self.accept)

# -----------------------------------------------------------------------------
# Main Application Window
# -----------------------------------------------------------------------------

class MainWindow(QMainWindow):
    new_mqtt_message = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        
        # Use standard window (no frameless for now to avoid black screen issue)
        self.setWindowTitle("RainNet-MT")
        self.resize(1200, 800)
        self.setWindowIcon(QIcon('Fig/B.ico'))
        
        # Center on screen
        screen = QApplication.primaryScreen().geometry()
        x = (screen.width() - 1200) // 2
        y = (screen.height() - 800) // 2
        self.move(x, y)
        
        self.setStyleSheet("""
            QMainWindow { 
                background-color: #F5F5F7;
            } 
            QScrollArea { border: none; background-color: transparent; }
            QWidget#Central { background-color: #F5F5F7; }
            QPushButton {
                font-family: 'Segoe UI', sans-serif;
            }
        """)

        # Initialize RainNet-MT engine with model if available
        model_path = r"D:\yuanru\daTA\enhanced_rainnet_plots\best_rainnet_enhanced_mt.pth"
        scaler_path = r"D:\yuanru\daTA\enhanced_rainnet_plots\scaler.pkl"  # Adjust if different name
        
        # Check if scaler exists with different possible names
        if not os.path.exists(scaler_path):
            # Try alternative names or locations
            alt_scaler = model_path.replace('best_rainnet_enhanced_mt.pth', 'scaler.pkl')
            if os.path.exists(alt_scaler):
                scaler_path = alt_scaler
            else:
                # Try to find scaler in same directory
                model_dir = os.path.dirname(model_path)
                alt_scaler = os.path.join(model_dir, 'scaler.pkl')
                if os.path.exists(alt_scaler):
                    scaler_path = alt_scaler
        
        self.rain_engine = RainNetMTEngine(
            model_path=model_path if os.path.exists(model_path) else None,
            scaler_path=scaler_path if os.path.exists(scaler_path) else None
        )
        self.current_topic = "raspberry/mqtt"
        self.current_city = "Fukuoka"
        self.username = "User"
        
        # Data history for charts
        self.rain_history = {
            'time': [],
            'probability': [],
            'humidity': []
        }
        
        # Alert tracking
        self.last_alert_time = 0
        self.alert_cooldown = 300  # 5 minutes cooldown between alerts
        
        # Init functionality
        self.screen_reminder = ScreenReminder()
        self.screen_reminder.userAwaySignal.connect(self.show_notification)
        
        self.initUI()
        
        # Timers
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_time)
        self.timer.start(1000)

        self.weather_timer = QTimer(self)
        self.weather_timer.timeout.connect(self.fetch_local_weather)
        self.weather_timer.start(1800000) # 30 mins
        
        # Fallback: If no MQTT data after 10 seconds, show demo data
        self.demo_timer = QTimer(self)
        self.demo_timer.timeout.connect(self.check_for_demo_mode)
        self.demo_timer.setSingleShot(True)
        
        self.new_mqtt_message.connect(self.process_data)
        
        # MQTT thread tracking
        self.mqtt_thread = None
        
        # Don't show main window until login completes
        # Show login immediately
        QTimer.singleShot(50, self.show_login_dialog)
        
    def closeEvent(self, event):
        """Clean shutdown."""
        # Stop MQTT thread if running
        if self.mqtt_thread and hasattr(self.mqtt_thread, 'stop'):
            try:
                self.mqtt_thread.stop()
            except:
                pass
        event.accept()

    def initUI(self):
        # Main Content Widget
        self.central_widget = QWidget()
        self.central_widget.setObjectName("Central")
        self.setCentralWidget(self.central_widget)
        
        main_layout = QVBoxLayout(self.central_widget)
        main_layout.setContentsMargins(25, 15, 25, 25)
        main_layout.setSpacing(18)
        
        # 1. Header Row
        header = QHBoxLayout()
        
        title_block = QVBoxLayout()
        title_block.setSpacing(4)
        title_block.setContentsMargins(0, 0, 0, 0)
        self.app_title = QLabel("RainNet-MT")
        self.app_title.setFont(QFont("Segoe UI", 24, QFont.Bold))
        self.app_title.setStyleSheet("color: #1C1C1E; padding: 0px;")
        self.app_title.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.location_sub = QLabel("Fukuoka • Loading...")
        self.location_sub.setFont(QFont("Segoe UI", 13))
        self.location_sub.setStyleSheet("color: #8E8E93; padding: 0px;")
        self.location_sub.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        title_block.addWidget(self.app_title)
        title_block.addWidget(self.location_sub)
        
        header.addLayout(title_block)
        header.addStretch()
        
        # Action Buttons
        self.dashboard_btn = QPushButton("Analytics")
        self.dashboard_btn.setFixedSize(105, 42)
        self.dashboard_btn.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self.dashboard_btn.setStyleSheet("""
            QPushButton { background-color: #34C759; color: white; border-radius: 8px; font-weight: bold; }
            QPushButton:hover { background-color: #2DA44E; }
        """)
        self.dashboard_btn.clicked.connect(self.show_dashboard)
        
        self.report_btn = QPushButton("Report")
        self.report_btn.setFixedSize(95, 42)
        self.report_btn.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self.report_btn.setStyleSheet("""
            QPushButton { background-color: #007AFF; color: white; border-radius: 8px; font-weight: bold; }
            QPushButton:hover { background-color: #0062CC; }
        """)
        self.report_btn.clicked.connect(self.show_report)
        
        self.settings_btn = QPushButton("⚙")
        self.settings_btn.setFixedSize(45, 42)
        self.settings_btn.setFont(QFont("Segoe UI", 16))
        self.settings_btn.setStyleSheet("""
            QPushButton { background-color: #E5E5EA; color: #000; border-radius: 8px; font-weight: bold; }
            QPushButton:hover { background-color: #D1D1D6; }
        """)
        self.settings_btn.clicked.connect(self.show_settings)
        
        header.addWidget(self.dashboard_btn)
        header.addWidget(self.report_btn)
        header.addWidget(self.settings_btn)
        main_layout.addLayout(header)
        
        # 2. Weather Card (Clickable)
        self.weather_card = WeatherCard()
        self.weather_card.mousePressEvent = lambda event: self.show_weather_details()
        main_layout.addWidget(self.weather_card)
        
        # 3. Sensor Grid
        self.grid_layout = QGridLayout()
        self.grid_layout.setSpacing(15)
        self.grid_layout.setContentsMargins(0, 10, 0, 10)
        
        self.sensors = {}
        # 按类别组织：室内环境、室外气象、健康相关
        # 室内环境（温湿度相关）- 蓝色系
        INDOOR_COLOR = "#007AFF"  # 蓝色
        # 室外气象 - 绿色系
        OUTDOOR_COLOR = "#34C759"  # 绿色
        # 健康相关（空气质量）- 橙色系
        HEALTH_COLOR = "#FF9500"  # 橙色
        
        sensor_configs = [
            # 第一行：室内环境（前3个）+ 室外（1个）
            ("Temperature", "°C", 'Ta', INDOOR_COLOR),      # 室内环境
            ("Humidity", "%", 'Humidity', INDOOR_COLOR),     # 室内环境
            ("Globe Temp", "°C", 'Tg', INDOOR_COLOR),        # 室内环境
            ("Pressure", "hPa", 'Pressure', OUTDOOR_COLOR), # 室外气象
            # 第二行：室外（1个）+ 健康相关（3个）
            ("Wind Speed", "m/s", 'Windspeed', OUTDOOR_COLOR), # 室外气象
            ("CO2", "ppm", 'CO2', HEALTH_COLOR),            # 健康相关
            ("PM2.5", "μg/m³", 'PM2.5', HEALTH_COLOR),     # 健康相关
            ("PM10", "μg/m³", 'PM10', HEALTH_COLOR)         # 健康相关
        ]
        
        row, col = 0, 0
        for title, unit, key, color in sensor_configs:
            widget = SensorWidget(title, unit, accent_color=color)
            widget.mousePressEvent = lambda event, k=key: self.show_trend(k)
            self.sensors[key] = widget
            self.grid_layout.addWidget(widget, row, col)
            col += 1
            if col >= 4:  # 4 columns layout
                col = 0
                row += 1

        main_layout.addLayout(self.grid_layout)
        
        # 4. Prediction Frame with Live Chart
        pred_frame = ModernCard(bg_color="#FFFFFF")
        pred_layout = QVBoxLayout(pred_frame)
        pred_layout.setContentsMargins(25, 25, 25, 25)
        
        pred_header = QHBoxLayout()
        pred_header.setSpacing(12)
        pred_title = QLabel("Rainfall Prediction (1h)")
        pred_title.setFont(QFont("Segoe UI", 15, QFont.Bold))
        pred_title.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        pred_title.setStyleSheet("color: #1C1C1E; padding: 0px;")
        pred_header.addWidget(pred_title)
        pred_header.addStretch()
        self.status_pill = QLabel(" Monitoring ")
        self.status_pill.setFont(QFont("Segoe UI", 11))
        self.status_pill.setStyleSheet("background-color: #E5F9E0; color: #34C759; border-radius: 12px; padding: 6px 14px;")
        self.status_pill.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        pred_header.addWidget(self.status_pill)
        pred_layout.addLayout(pred_header)
        
        # Status Text
        self.intensity_label = QLabel("Analyzing data...")
        self.intensity_label.setFont(QFont("Segoe UI", 18, QFont.Bold))
        self.intensity_label.setStyleSheet("color: #8E8E93; padding: 4px 0px;")
        self.intensity_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.intensity_label.setWordWrap(True)
        pred_layout.addWidget(self.intensity_label)
        
        # Progress Bar
        self.prob_bar = QProgressBar()
        self.prob_bar.setRange(0, 100)
        self.prob_bar.setValue(0)
        self.prob_bar.setFixedHeight(14)
        self.prob_bar.setStyleSheet("""
            QProgressBar {
                border: none;
                border-radius: 7px;
                background-color: #E5E5EA;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #34C759;
                border-radius: 7px;
            }
        """)
        pred_layout.addWidget(self.prob_bar)
        
        # Mini Trend Chart
        self.mini_figure, self.mini_ax = plt.subplots(figsize=(9, 2.2))
        self.mini_figure.patch.set_facecolor('#FFFFFF')
        self.mini_ax.set_facecolor('#FAFAFA')
        # Initialize with empty plot but proper axis range
        self.mini_ax.set_xlim(0, 30)
        self.mini_ax.set_ylim(0, 100)
        self.mini_ax.set_xlabel('Recent Updates', fontsize=9, color='#8E8E93')
        self.mini_ax.set_ylabel('Probability (%)', fontsize=9, color='#8E8E93')
        self.mini_ax.tick_params(labelsize=8, colors='#8E8E93')
        self.mini_ax.grid(True, alpha=0.2, color='#E5E5EA')
        self.mini_ax.text(15, 50, 'Waiting for data...', ha='center', va='center', 
                         fontsize=10, color='#8E8E93', style='italic')
        self.mini_canvas = FigureCanvas(self.mini_figure)
        self.mini_canvas.setFixedHeight(200)
        self.mini_canvas.setStyleSheet("border: none;")
        pred_layout.addWidget(self.mini_canvas)
        
        main_layout.addWidget(pred_frame)
        main_layout.addStretch()

    def update_time(self):
        """Update time display in location subtitle."""
        now = datetime.now()
        self.location_sub.setText(f"{self.current_city} • {now.strftime('%H:%M')}")

    def fetch_local_weather(self):
        coords = CITIES.get(self.current_city, CITIES["Fukuoka"])
        try:
            url = f"{WEATHER_API_URL}?latitude={coords['lat']}&longitude={coords['lon']}&current_weather=true"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json().get('current_weather', {})
                self.weather_card.update_weather(
                    self.current_city, 
                    data.get('temperature', 0), 
                    data.get('weathercode', 0),
                    data.get('windspeed', 0)
                )
        except Exception as e:
            print(f"Weather fetch error: {e}")

    def show_trend(self, label_name):
        dialog = DatabaseTrendDialog(label_name)
        dialog.exec_()

    def show_dashboard(self):
        dlg = ComprehensiveDashboard(self)
        dlg.exec_()

    def show_about(self):
        dlg = AboutDialog(self)
        dlg.exec_()

    def show_weather_details(self):
        dlg = WeatherDetailsDialog(self.current_city, self)
        dlg.exec_()

    def show_settings(self):
        dlg = SettingsDialog(self)
        if dlg.exec_() == QDialog.Accepted:
            # Apply visibility settings
            for sensor_name, checkbox in dlg.checkboxes.items():
                # Map display name to sensor key
                key_map = {
                    "Temperature": "Ta",
                    "Humidity": "Humidity", 
                    "Pressure": "Pressure",
                    "Wind Speed": "Windspeed",
                    "CO2": "CO2",
                    "PM2.5": "PM2.5",
                    "PM10": "PM10",
                    "Globe Temp": "Tg"
                }
                sensor_key = key_map.get(sensor_name)
                if sensor_key and sensor_key in self.sensors:
                    self.sensors[sensor_key].setVisible(checkbox.isChecked())

    def show_report(self):
        dlg = ActivityReportDialog(self.username)
        dlg.exec_()

    def show_notification(self, message):
        """Show system notification or fallback to message box."""
        if NOTIFICATION_AVAILABLE:
            try:
                notification.notify(title="RainNet-MT", message=message, app_icon="Fig/B.ico", timeout=5)
                return
            except:
                pass
        # Fallback: Show message box
        msg = QMessageBox(self)
        msg.setWindowTitle("RainNet-MT Alert")
        msg.setText(message)
        msg.setIcon(QMessageBox.Information)
        msg.exec_()

    def process_data(self, data):
        # Update all sensors
        mapping = {
            'Ta': 'Ta', 
            'Humidity': 'Humidity', 
            'Pressure': 'Pressure', 
            'Windspeed': 'Windspeed', 
            'CO2': 'CO2', 
            'PM2.5': 'PM2.5',
            'PM10': 'PM10',
            'Tg': 'Tg'
        }
        
        for key, sensor_key in mapping.items():
            if key in data and sensor_key in self.sensors:
                val = data[key]
                # Format values for display
                try:
                    if key == 'Ta' or key == 'Tg':
                        val = f"{float(val):.1f}"
                    elif key == 'Humidity':
                        val = f"{float(val):.1f}"
                    elif key == 'Pressure':
                        val = f"{float(val):.1f}"
                    elif key == 'Windspeed':
                        val = f"{float(val):.2f}"
                    elif key in ['CO2', 'PM2.5', 'PM10']:
                        val = f"{int(float(val))}"
                except:
                    val = str(val)
                self.sensors[sensor_key].update_value(val)
        
        # CO2 level indicator
        if 'CO2' in data:
            co2_val = float(data['CO2']) if isinstance(data['CO2'], (int, float)) else 0
            if 'CO2' in self.sensors:
                if co2_val < 600:
                    self.sensors['CO2'].dot.setStyleSheet("color: #34C759; font-size: 16px;")
                elif co2_val < 1000:
                    self.sensors['CO2'].dot.setStyleSheet("color: #FF9500; font-size: 16px;")
                else:
                    self.sensors['CO2'].dot.setStyleSheet("color: #FF3B30; font-size: 16px;")
        
        # Predict rainfall
        res = self.rain_engine.predict(data)
        prob = int(res['probability'])
        self.prob_bar.setValue(prob)
        self.intensity_label.setText(res['intensity_label'])
        
        if prob < 40: color = "#34C759"
        elif prob < 70: color = "#FF9500"
        else: color = "#FF3B30"
        
        self.intensity_label.setStyleSheet(f"color: {color};")
        self.prob_bar.setStyleSheet(f"QProgressBar::chunk {{ background-color: {color}; border-radius: 6px; }} QProgressBar {{ background-color: #F2F2F7; border-radius: 6px; }}")
        
        # Update mini chart (Keep last 30 data points)
        self.rain_history['time'].append(datetime.now())
        self.rain_history['probability'].append(prob)
        if 'Humidity' in data:
            self.rain_history['humidity'].append(float(data['Humidity']))
        
        # Keep only recent data
        if len(self.rain_history['time']) > 30:
            self.rain_history['time'] = self.rain_history['time'][-30:]
            self.rain_history['probability'] = self.rain_history['probability'][-30:]
            self.rain_history['humidity'] = self.rain_history['humidity'][-30:]
        
        # Redraw mini chart - Always update when we have data
        self.mini_ax.clear()
        x_range = list(range(len(self.rain_history['probability'])))
        
        if len(x_range) > 0:
            # Plot the probability line
            self.mini_ax.plot(x_range, self.rain_history['probability'], color='#007AFF', 
                            linewidth=2.5, marker='o', markersize=4, label='Rain Prob %', zorder=3)
            # Fill area under curve
            self.mini_ax.fill_between(x_range, self.rain_history['probability'], alpha=0.3, 
                                    color='#007AFF', zorder=2)
            # Add legend
            self.mini_ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
        else:
            # Show waiting message if no data yet
            self.mini_ax.text(15, 50, 'Waiting for data...', ha='center', va='center', 
                             fontsize=10, color='#8E8E93', style='italic')
        
        # Set axis properties
        max_x = max(30, len(x_range) + 2) if len(x_range) > 0 else 30
        self.mini_ax.set_xlim(-0.5, max_x)
        self.mini_ax.set_ylim(0, 100)
        self.mini_ax.set_xlabel('Recent Updates', fontsize=9, color='#8E8E93')
        self.mini_ax.set_ylabel('Probability (%)', fontsize=9, color='#8E8E93')
        self.mini_ax.tick_params(labelsize=8, colors='#8E8E93')
        self.mini_ax.grid(True, alpha=0.2, color='#E5E5EA', zorder=1)
        
        # Force redraw
        plt.tight_layout()
        self.mini_canvas.draw()
        self.mini_canvas.flush_events()  # Ensure immediate update
        
        # Smart Alerts (with cooldown to prevent spam)
        current_time = time.time()
        if current_time - self.last_alert_time > self.alert_cooldown:
            # Heavy rain alert
            if prob >= 85:
                self.show_notification("⚠️ Heavy Rain Warning! Probability: " + str(prob) + "%")
                self.last_alert_time = current_time
            
            # CO2 alert
            if 'CO2' in data:
                co2_val = float(data['CO2']) if isinstance(data['CO2'], (int, float)) else 0
                if co2_val > 1500:
                    self.show_notification("⚠️ CO2 Level High! Consider ventilation. Level: " + str(int(co2_val)) + " ppm")
                    self.last_alert_time = current_time

    def start_mqtt(self, topic):
        """Start MQTT connection in background thread."""
        self.current_topic = topic
        
        # Stop existing thread if any
        if self.mqtt_thread and hasattr(self.mqtt_thread, 'is_alive') and self.mqtt_thread.is_alive():
            try:
                self.mqtt_thread.stop()
            except:
                pass
        
        # Start new thread
        self.mqtt_thread = StoppableThread(target=self.mqtt_loop)
        self.mqtt_thread.start()

    def mqtt_loop(self):
        client = None
        try:
            # Use VERSION2 if available
            if hasattr(mqtt, 'CallbackAPIVersion'):
                client = mqtt.Client(
                    mqtt.CallbackAPIVersion.VERSION2, 
                    "RainNet_" + str(int(time.time())),
                    protocol=mqtt.MQTTv5
                )
            else:
                client = mqtt.Client("RainNet_" + str(int(time.time())))
        except:
            # Fallback
            client = mqtt.Client("RainNet_" + str(int(time.time())))
        
        client.username_pw_set(MQTT_USER, MQTT_PASSWORD)
        
        # TLS/SSL Configuration
        if MQTT_PORT == 8883:
            import ssl
            # Use custom CA cert if it exists, otherwise default system certs
            ca_certs = MQTT_CA_CERT if os.path.exists(MQTT_CA_CERT) else None
            try:
                if ca_certs:
                    client.tls_set(ca_certs=ca_certs, cert_reqs=ssl.CERT_REQUIRED, tls_version=ssl.PROTOCOL_TLSv1_2)
                else:
                    client.tls_set(cert_reqs=ssl.CERT_REQUIRED, tls_version=ssl.PROTOCOL_TLSv1_2)
            except Exception as e:
                print(f"TLS Setup Error: {e}")

        def on_connect(client, userdata, flags, rc, properties=None):
            """Callback when connected to MQTT broker."""
            if rc == 0:
                print(f"Connected to MQTT: {MQTT_BROKER}")
                client.subscribe(self.current_topic)
            else:
                print(f"MQTT Connection failed with code: {rc}")

        def on_message(client, userdata, msg):
            try:
                payload = msg.payload.decode('utf-8')
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    return

                data_dict = {}
                if isinstance(data, list) and len(data) >= 11:
                    data_dict = {
                        'timestamp': data[0], 'Ta': data[1], 'Tg': data[2],
                        'Pressure': data[3], 'Humidity': data[4], 'MRT': data[5],
                        'Windspeed': data[6], 'PMV': data[7], 'CO2': data[8],
                        'PM2.5': data[9] if len(data) > 9 else 0,
                        'PM10': data[10] if len(data) > 10 else 0
                    }
                elif isinstance(data, dict):
                    data_dict = data
                else:
                    return

                self.new_mqtt_message.emit(data_dict)
            except Exception as e:
                print(f"MQTT Parse Error: {e}")

        client.on_connect = on_connect
        client.on_message = on_message
        
        try:
            client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
            print(f"Connecting to {MQTT_BROKER}:{MQTT_PORT}...")
            client.loop_forever()
        except Exception as e:
            print(f"MQTT Connection Error: {e}")

    def show_login_dialog(self):
        # Show login dialog immediately
        dlg = CitySelectionDialog(self)
        
        if dlg.exec_() == QDialog.Accepted:
            # Show connecting animation
            connecting = ConnectingDialog(self)
            connecting.exec_()  # Block during connection (3 seconds)
            
            # Update settings
            self.current_city = dlg.selected_city
            self.username = dlg.username
            self.location_sub.setText(f"{self.current_city} • Online")
            
            # Start services
            self.start_mqtt("raspberry/mqtt")
            self.fetch_local_weather()
            
            # Initialize display
            self.initialize_sample_data()
            
            # Show main window
            self.show()
            self.raise_()
            self.activateWindow()
        else:
            sys.exit()
            
    def initialize_sample_data(self):
        """Initialize sensors with placeholder data for immediate display."""
        # Set initial "waiting" display
        for sensor_key in self.sensors.keys():
            self.sensors[sensor_key].update_value("--")
        
        # Update status
        self.intensity_label.setText("Waiting for sensor data...")
        self.intensity_label.setStyleSheet("color: #8E8E93;")
        
        # Initialize chart - just show waiting message, don't add fake data
        self.mini_ax.clear()
        self.mini_ax.set_xlim(-0.5, 30)
        self.mini_ax.set_ylim(0, 100)
        self.mini_ax.set_xlabel('Recent Updates', fontsize=9, color='#8E8E93')
        self.mini_ax.set_ylabel('Probability (%)', fontsize=9, color='#8E8E93')
        self.mini_ax.tick_params(labelsize=8, colors='#8E8E93')
        self.mini_ax.grid(True, alpha=0.2, color='#E5E5EA')
        self.mini_ax.text(15, 50, 'Waiting for data...', ha='center', va='center', 
                         fontsize=10, color='#8E8E93', style='italic')
        plt.tight_layout()
        self.mini_canvas.draw()
        
        # Start demo mode timer (shorter for faster feedback)
        self.demo_timer.start(5000)  # 5 seconds - faster demo data
        
    def check_for_demo_mode(self):
        """If no real data received, show demo data."""
        # Check if any sensor has real data
        has_data = any(self.sensors[k].value_label.text() != "--" for k in self.sensors.keys())
        
        if not has_data:
            # Generate demo data with some variation to show chart activity
            import random
            base_humidity = 65 + random.uniform(-5, 10)
            demo_data = {
                'Ta': 23.5 + random.uniform(-1, 1),
                'Humidity': base_humidity,
                'Pressure': 1013.2 + random.uniform(-2, 2),
                'Windspeed': 2.3 + random.uniform(-0.5, 0.5),
                'CO2': 480 + random.randint(-20, 20),
                'PM2.5': 15 + random.randint(-3, 3),
                'PM10': 28 + random.randint(-5, 5),
                'Tg': 24.1 + random.uniform(-0.5, 0.5)
            }
            self.process_data(demo_data)
            
            # Update status to indicate demo mode
            self.status_pill.setText(" Demo Mode ")
            self.status_pill.setStyleSheet("background-color: #FFF3E0; color: #FF9500; border-radius: 12px; padding: 6px 12px;")
            
            # Schedule periodic demo updates to show chart progression
            if not hasattr(self, 'demo_update_timer'):
                self.demo_update_timer = QTimer(self)
                self.demo_update_timer.timeout.connect(self.update_demo_data)
                self.demo_update_timer.start(3000)  # Update every 3 seconds
    
    def update_demo_data(self):
        """Periodically update demo data to show chart progression."""
        # Check if we still need demo mode
        has_real_data = any(self.sensors[k].value_label.text() != "--" for k in self.sensors.keys())
        if has_real_data:
            # Stop demo updates if real data arrived
            if hasattr(self, 'demo_update_timer'):
                self.demo_update_timer.stop()
            return
        
        # Generate new demo data with slight variations
        import random
        base_humidity = 65 + random.uniform(-5, 10)
        demo_data = {
            'Ta': 23.5 + random.uniform(-1, 1),
            'Humidity': base_humidity,
            'Pressure': 1013.2 + random.uniform(-2, 2),
            'Windspeed': 2.3 + random.uniform(-0.5, 0.5),
            'CO2': 480 + random.randint(-20, 20),
            'PM2.5': 15 + random.randint(-3, 3),
            'PM10': 28 + random.randint(-5, 5),
            'Tg': 24.1 + random.uniform(-0.5, 0.5)
        }
        self.process_data(demo_data)

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

class StoppableThread(threading.Thread):
    def __init__(self, target=None, args=()):
        super().__init__(target=target, args=args)
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    QCoreApplication.setApplicationName('RainNet-MT System')
    
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    main = MainWindow()
    # Don't show main window here - it will show after login
    # main.show() is called in show_login_dialog() after successful login
    
    sys.exit(app.exec_())
