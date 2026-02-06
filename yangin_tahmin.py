# -*- coding: utf-8 -*-

import streamlit as st
import requests
import numpy as np
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation, PillowWriter
import time
from datetime import datetime, timedelta
import os
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

try:
    import rasterio
    from rasterio.transform import rowcol
    from rasterio.windows import Window
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    st.error("❌ rasterio yüklü değil! pip install rasterio")

# ==================== YAPILANDIRMA ====================
st.set_page_config(
    page_title="🚨 Operasyonel Yangın Yönetimi v7.0",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Keys
API_KEYS = [
    "c1af9673bcc98f462db39ee5ffbb13e5"
]
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", API_KEYS[0])

# Dosya yolları - Programla aynı klasörde arar
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
CORINE_PATH = os.path.join(SCRIPT_DIR, "corine_tr_gercek_2018.tif")

# CORINE sınıf tanımları
CORINE_FUEL_MAP = {
    111: {"name": "Sürekli Kentsel Yapı", "fuel_load": 0.1, "sav": 2000, "moisture": 15, "flammability": 0.1},
    112: {"name": "Süreksiz Kentsel Yapı", "fuel_load": 0.5, "sav": 2500, "moisture": 15, "flammability": 0.3},
    121: {"name": "Sanayi/Ticaret", "fuel_load": 0.2, "sav": 2000, "moisture": 10, "flammability": 0.2},
    131: {"name": "Maden Çıkarma", "fuel_load": 0.1, "sav": 1500, "moisture": 5, "flammability": 0.1},
    141: {"name": "Kentsel Yeşil Alan", "fuel_load": 1.0, "sav": 3000, "moisture": 25, "flammability": 0.4},
    142: {"name": "Spor/Dinlenme", "fuel_load": 0.8, "sav": 3000, "moisture": 25, "flammability": 0.3},
    211: {"name": "Sulanmayan Tarım", "fuel_load": 1.5, "sav": 4000, "moisture": 20, "flammability": 0.6},
    212: {"name": "Sulanan Tarım", "fuel_load": 1.2, "sav": 4500, "moisture": 30, "flammability": 0.4},
    213: {"name": "Pirinç Tarlası", "fuel_load": 0.8, "sav": 5000, "moisture": 40, "flammability": 0.2},
    221: {"name": "Bağ", "fuel_load": 1.8, "sav": 3500, "moisture": 20, "flammability": 0.7},
    222: {"name": "Meyve Bahçesi", "fuel_load": 2.0, "sav": 3500, "moisture": 22, "flammability": 0.7},
    223: {"name": "Zeytin Bahçesi", "fuel_load": 2.2, "sav": 3000, "moisture": 18, "flammability": 0.8},
    231: {"name": "Çayır/Otlak", "fuel_load": 1.5, "sav": 5000, "moisture": 25, "flammability": 0.7},
    241: {"name": "Yıllık/Sürekli Tarım", "fuel_load": 1.3, "sav": 4000, "moisture": 25, "flammability": 0.5},
    242: {"name": "Karmaşık Tarım", "fuel_load": 1.6, "sav": 3800, "moisture": 23, "flammability": 0.6},
    243: {"name": "Tarım/Doğal Vejetasyon", "fuel_load": 2.0, "sav": 3500, "moisture": 20, "flammability": 0.7},
    244: {"name": "Tarım/Orman", "fuel_load": 2.5, "sav": 3200, "moisture": 18, "flammability": 0.8},
    311: {"name": "Geniş Yapraklı Orman", "fuel_load": 3.5, "sav": 2800, "moisture": 15, "flammability": 0.9},
    312: {"name": "İğne Yapraklı Orman", "fuel_load": 4.5, "sav": 2500, "moisture": 12, "flammability": 1.0},
    313: {"name": "Karışık Orman", "fuel_load": 4.0, "sav": 2600, "moisture": 13, "flammability": 0.95},
    321: {"name": "Doğal Çayırlık", "fuel_load": 2.0, "sav": 4500, "moisture": 22, "flammability": 0.7},
    322: {"name": "Fundalık", "fuel_load": 3.0, "sav": 3000, "moisture": 15, "flammability": 0.9},
    323: {"name": "Sert Yapraklı Vejetasyon", "fuel_load": 3.2, "sav": 2800, "moisture": 14, "flammability": 0.95},
    324: {"name": "Orman-Fundalık Geçişi", "fuel_load": 3.5, "sav": 2700, "moisture": 14, "flammability": 0.95},
    331: {"name": "Kumsallar/Kumullar", "fuel_load": 0.2, "sav": 5000, "moisture": 5, "flammability": 0.1},
    332: {"name": "Çıplak Kayalık", "fuel_load": 0.1, "sav": 1000, "moisture": 2, "flammability": 0.05},
    333: {"name": "Seyrek Vejetasyon", "fuel_load": 0.8, "sav": 4000, "moisture": 10, "flammability": 0.4},
    334: {"name": "Yanmış Alanlar", "fuel_load": 0.5, "sav": 5000, "moisture": 8, "flammability": 0.3},
    335: {"name": "Buzullar/Kar", "fuel_load": 0.0, "sav": 1000, "moisture": 100, "flammability": 0.0},
    411: {"name": "İç Sulak Alanlar", "fuel_load": 1.5, "sav": 5500, "moisture": 50, "flammability": 0.2},
    412: {"name": "Turbalıklar", "fuel_load": 2.0, "sav": 4000, "moisture": 60, "flammability": 0.3},
    421: {"name": "Tuzlu Bataklıklar", "fuel_load": 1.2, "sav": 5000, "moisture": 55, "flammability": 0.2},
    422: {"name": "Tuzlalar", "fuel_load": 0.3, "sav": 3000, "moisture": 40, "flammability": 0.1},
    423: {"name": "Gelgit Bölgeleri", "fuel_load": 0.8, "sav": 4500, "moisture": 50, "flammability": 0.2},
    511: {"name": "Su Yolları", "fuel_load": 0.0, "sav": 1000, "moisture": 100, "flammability": 0.0},
    512: {"name": "Su Yüzeyleri", "fuel_load": 0.0, "sav": 1000, "moisture": 100, "flammability": 0.0},
    521: {"name": "Kıyı Lagünleri", "fuel_load": 0.5, "sav": 4000, "moisture": 70, "flammability": 0.1},
    522: {"name": "Halicler", "fuel_load": 0.3, "sav": 3500, "moisture": 75, "flammability": 0.1},
    523: {"name": "Denizler/Okyanuslar", "fuel_load": 0.0, "sav": 1000, "moisture": 100, "flammability": 0.0},
}

class SimulationConfig:
    TIME_STEP = 5  # dakika
    PIXEL_SIZE = 0.1  # km

# ==================== VERİ YÜKLEME ====================

@st.cache_resource
def load_corine_data():
    """Gerçek CORINE verisini yükle"""
    if not os.path.exists(CORINE_PATH):
        st.error(f"❌ CORINE dosyası bulunamadı: {CORINE_PATH}")
        return None, None
    
    try:
        dataset = rasterio.open(CORINE_PATH)
        st.success(f"✅ CORINE yüklendi: {dataset.width}x{dataset.height}, {dataset.crs}")
        return dataset, dataset.profile
    except Exception as e:
        st.error(f"❌ CORINE yükleme hatası: {e}")
        return None, None

def extract_real_terrain(corine_dataset, lat, lon, grid_size=(100, 100)):
    """
    Gerçek CORINE verisinden bölgesel arazi bilgisi çıkar
    """
    if corine_dataset is None:
        # Fallback: simüle veri
        return np.random.randint(311, 314, size=grid_size), np.random.uniform(0, 30, size=grid_size)
    
    try:
        # Koordinatı piksel indeksine çevir
        row, col = rowcol(corine_dataset.transform, lon, lat)
        
        # Grid boyutunun yarısı
        half_h = grid_size[0] // 2
        half_w = grid_size[1] // 2
        
        # Window oluştur
        window = Window(
            col_off=max(0, col - half_w),
            row_off=max(0, row - half_h),
            width=grid_size[1],
            height=grid_size[0]
        )
        
        # Veriyi oku
        local_corine = corine_dataset.read(1, window=window)
        
        # Eğer boyut uymazsa resize et
        if local_corine.shape != grid_size:
            from scipy.ndimage import zoom
            zoom_factor = (grid_size[0] / local_corine.shape[0], 
                          grid_size[1] / local_corine.shape[1])
            local_corine = zoom(local_corine, zoom_factor, order=0)
        
        # Eğim simüle et (gerçek DEM olmadığı için)
        # TODO: DEM dosyası eklendiğinde burası güncellenecek
        local_slope = np.random.uniform(0, 25, size=grid_size)
        
        return local_corine, local_slope
        
    except Exception as e:
        st.warning(f"⚠️ CORINE okuma hatası: {e}. Simüle veri kullanılıyor.")
        return np.random.randint(311, 314, size=grid_size), np.random.uniform(0, 30, size=grid_size)

# ==================== HAVA DURUMU ====================

def get_weather_data(lat, lon):
    """Gerçek zamanlı hava durumu - OpenWeather API"""
    url = f"http://api.openweathermap.org/data/2.5/weather"
    params = {
        'lat': lat,
        'lon': lon,
        'appid': OPENWEATHER_API_KEY,
        'units': 'metric',
        'lang': 'tr'
    }
    
    try:
        response = requests.get(url, params=params, timeout=5)
        data = response.json()
        
        return {
            'sıcaklık': data['main']['temp'],
            'nem': data['main']['humidity'],
            'rüzgar_hız': data['wind']['speed'],
            'rüzgar_yön': data['wind'].get('deg', 0),
            'durum': data['weather'][0]['description'],
            'basınç': data['main']['pressure'],
            'görüş_mesafesi': data.get('visibility', 10000) / 1000,  # km
            'bulut': data.get('clouds', {}).get('all', 0),  # %
        }
    except Exception as e:
        st.warning(f"⚠️ Hava durumu API hatası: {e}. Varsayılan değerler kullanılıyor.")
        return {
            'sıcaklık': 30.0,
            'nem': 35,
            'rüzgar_hız': 5.0,
            'rüzgar_yön': 270,
            'durum': 'Simülasyon verisi',
            'basınç': 1013,
            'görüş_mesafesi': 10,
            'bulut': 20
        }

def get_location_name(lat, lon):
    """Konum adı"""
    url = f"http://api.openweathermap.org/geo/1.0/reverse"
    params = {
        'lat': lat,
        'lon': lon,
        'appid': OPENWEATHER_API_KEY,
        'limit': 1
    }
    
    try:
        response = requests.get(url, params=params, timeout=5)
        data = response.json()
        if data:
            return {
                'il': data[0].get('state', 'Türkiye'),
                'ilçe': data[0].get('name', 'Bilinmiyor'),
                'ülke': data[0].get('country', 'TR')
            }
    except:
        pass
    
    return {'il': 'Türkiye', 'ilçe': 'Seçilen Konum', 'ülke': 'TR'}

# ==================== MATEMATİKSEL MODELLER ====================

class FireModels:
    """Yangın yayılım modelleri"""
    
    @staticmethod
    def rothermel_with_corine(wind_speed, slope_deg, corine_class, 
                               temperature=25, humidity=50):
        """
        Rothermel modeli + Gerçek CORINE yakıt özellikleri
        """
        # CORINE'den yakıt özellikleri
        fuel_props = CORINE_FUEL_MAP.get(corine_class, CORINE_FUEL_MAP[312])  # Varsayılan: iğne yapraklı orman
        
        fuel_load = fuel_props['fuel_load']
        fuel_sav = fuel_props['sav']
        base_moisture = fuel_props['moisture']
        
        # Nem etkisiyle yakıt nemi
        fuel_moisture = base_moisture * (humidity / 50)
        
        # Rothermel hesaplaması
        sigma = fuel_sav
        moisture_damping = max(0, 1 - 2.59 * (fuel_moisture / 100))
        mineral_damping = 0.174 * (0.19 ** (-0.8189))
        
        IR = 0.0591 + 2.926 * (sigma ** -1.5) * fuel_load * moisture_damping * mineral_damping
        IR = max(0, IR)
        
        # Rüzgar faktörü
        U_mph = wind_speed * 2.237
        C = 7.47 * np.exp(-0.133 * sigma ** 0.55)
        B = 0.02526 * sigma ** 0.54
        E = 0.715 * np.exp(-3.59e-4 * sigma)
        
        beta = 0.0012
        beta_op = 3.348 * sigma ** (-0.8189)
        
        phi_w = C * (U_mph ** B) * ((beta / beta_op) ** (-E))
        phi_w = max(0, min(15, phi_w))
        
        # Eğim faktörü
        slope_rad = np.radians(slope_deg)
        phi_s = 5.275 * (beta ** -0.3) * (np.tan(slope_rad) ** 2)
        phi_s = max(0, min(10, phi_s))
        
        # Aşırı koşul çarpanı
        multiplier = 1.0
        if wind_speed > 15:
            multiplier *= (1 + (wind_speed - 15) * 0.1)
        if humidity < 30:
            multiplier *= (1 + (30 - humidity) * 0.02)
        if temperature > 30:
            multiplier *= (1 + (temperature - 30) * 0.03)
        
        # Yanabilirlik faktörü
        flammability = fuel_props['flammability']
        
        # Final hız
        xi = 0.174 * (sigma ** -0.19)
        R0 = IR * xi / (192.0 + 7.9095 * fuel_moisture)
        R = R0 * (1 + phi_w + phi_s) * multiplier * flammability
        
        return max(0, R)

# ==================== SİMÜLASYON ====================

def simulate_fire_operational(start_pos, grid_size, weather, slope, corine, 
                               steps=30, time_horizon_hours=24,
                               spread_multiplier=3.0, use_random_seed=True):
    """
    Operasyonel yangın simülasyonu
    
    Args:
        spread_multiplier: Yayılım hızı çarpanı (1.0-10.0)
        use_random_seed: True ise her seferinde aynı sonuç (seed=42)
    """
    history = []
    intensity_history = []
    time_stamps = []
    
    # Random seed ayarla
    if use_random_seed:
        np.random.seed(42)  # Tutarlı sonuçlar
    else:
        np.random.seed(None)  # Rastgele sonuçlar
    
    # Başlangıç
    grid = np.zeros(grid_size, dtype=float)
    intensity = np.zeros(grid_size, dtype=float)
    
    grid[start_pos] = 1.0
    intensity[start_pos] = 1.0
    
    history.append(grid.copy())
    intensity_history.append(intensity.copy())
    time_stamps.append(datetime.now())
    
    # Rüzgar yönü vektörü
    wind_dir_rad = np.radians(weather['rüzgar_yön'])
    wind_vec = (np.sin(wind_dir_rad), np.cos(wind_dir_rad))
    
    # Yayılım agresiflik faktörü (daha görünür yayılım için)
    SPREAD_MULTIPLIER = spread_multiplier  # Parametre olarak alındı
    MIN_SPREAD_RATE = 0.5    # Minimum yayılım garantisi
    
    # Simülasyon döngüsü
    for step in range(steps):
        new_grid = grid.copy()
        new_intensity = intensity.copy()
        burning_cells = np.argwhere(grid > 0.3)  # Eşik düşürüldü: 0.5 → 0.3
        
        if len(burning_cells) == 0:
            # Yangın sönmüş, tarihi tekrarla
            history.append(grid.copy())
            intensity_history.append(intensity.copy())
            time_stamps.append(datetime.now() + timedelta(minutes=(step+1)*SimulationConfig.TIME_STEP))
            continue
        
        for cell in burning_cells:
            y, x = cell
            
            # Bu hücrenin CORINE sınıfı ve eğimi
            corine_class = int(corine[y, x])
            cell_slope = slope[y, x]
            
            # Yayılım hızı hesapla
            spread_rate = FireModels.rothermel_with_corine(
                weather['rüzgar_hız'],
                cell_slope,
                corine_class,
                weather['sıcaklık'],
                weather['nem']
            )
            
            # Minimum yayılım garantisi
            spread_rate = max(MIN_SPREAD_RATE, spread_rate)
            
            # Komşu hücrelere yayılım
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    
                    ny, nx = y + dy, x + dx
                    
                    if 0 <= ny < grid_size[0] and 0 <= nx < grid_size[1]:
                        if new_grid[ny, nx] < 0.5:
                            # Komşu hücrenin yakılabilirliği
                            neighbor_corine = int(corine[ny, nx])
                            neighbor_fuel = CORINE_FUEL_MAP.get(neighbor_corine, 
                                                                CORINE_FUEL_MAP[312])
                            neighbor_flammability = neighbor_fuel['flammability']
                            
                            # Rüzgar yönü etkisi
                            dir_vec = (dx, dy)
                            wind_alignment = (dir_vec[0] * wind_vec[0] + 
                                            dir_vec[1] * wind_vec[1])
                            wind_factor = 1.0 + max(0, wind_alignment) * 1.5  # Artırıldı: 0.5 → 1.5
                            
                            # Mesafe faktörü (köşegen daha zor yanar)
                            distance = np.sqrt(dx**2 + dy**2)
                            distance_factor = 1.0 / distance
                            
                            # Yayılma olasılığı - GELİŞTİRİLMİŞ
                            spread_prob = (spread_rate * 0.05 * SPREAD_MULTIPLIER * 
                                         wind_factor * distance_factor * 
                                         neighbor_flammability)
                            spread_prob = min(0.98, spread_prob)  # Max %98
                            
                            if np.random.random() < spread_prob:
                                new_grid[ny, nx] = 1.0
                                new_intensity[ny, nx] = spread_rate * neighbor_flammability
        
        # Yanmakta olan hücrelerin yoğunluğunu azalt (tükenme)
        burning_mask = grid > 0.8
        new_intensity[burning_mask] *= 0.95  # %5 azalma
        
        grid = new_grid
        intensity = new_intensity
        
        history.append(grid.copy())
        intensity_history.append(intensity.copy())
        time_stamps.append(datetime.now() + timedelta(minutes=(step+1)*SimulationConfig.TIME_STEP))
    
    return history, intensity_history, time_stamps

# ==================== MÜDAHALE PLANLAMA ====================

class InterventionPlanner:
    """Müdahale planı oluşturucu"""
    
    @staticmethod
    def calculate_fire_perimeter(fire_grid):
        """Yangın çevre çizgisini bul"""
        from scipy import ndimage
        
        # Kenarları bul
        structure = np.array([[1,1,1],[1,1,1],[1,1,1]])
        dilated = ndimage.binary_dilation(fire_grid > 0.5, structure=structure)
        perimeter = dilated & ~(fire_grid > 0.5)
        
        return perimeter
    
    @staticmethod
    def find_firebreak_locations(fire_grid, corine_grid, distance_km=0.5):
        """
        Yangın engelleme hatları (firebreak) öner
        
        Returns:
            list of dicts: [{'lat': ..., 'lon': ..., 'priority': ...}, ...]
        """
        perimeter = InterventionPlanner.calculate_fire_perimeter(fire_grid)
        perimeter_points = np.argwhere(perimeter)
        
        # distance_km kadar uzaktaki noktaları bul
        distance_pixels = int(distance_km / SimulationConfig.PIXEL_SIZE)
        
        firebreak_locations = []
        
        for point in perimeter_points[::5]:  # Her 5 noktadan 1'ini al (performans için)
            y, x = point
            
            # Önündeki noktayı hesapla
            # TODO: Rüzgar yönüne göre optimize et
            fb_y = min(fire_grid.shape[0]-1, y + distance_pixels)
            fb_x = x
            
            # CORINE sınıfına göre öncelik
            corine_class = int(corine_grid[fb_y, fb_x])
            priority = CORINE_FUEL_MAP.get(corine_class, {}).get('flammability', 0.5)
            
            firebreak_locations.append({
                'grid_y': fb_y,
                'grid_x': fb_x,
                'priority': priority,
                'corine_class': corine_class
            })
        
        # Önceliğe göre sırala
        firebreak_locations.sort(key=lambda x: x['priority'], reverse=True)
        
        return firebreak_locations[:10]  # En önemli 10'u
    
    @staticmethod
    def find_retardant_drop_zones(fire_grid, intensity_grid, wind_direction):
        """
        Retardant (söndürücü kimyasal) dökülecek noktalar
        
        Returns:
            list of dicts with GPS coordinates and priority
        """
        # En yoğun yangın noktalarını bul
        high_intensity = intensity_grid > np.percentile(intensity_grid[intensity_grid > 0], 75)
        
        hot_spots = np.argwhere(high_intensity)
        
        drop_zones = []
        
        for spot in hot_spots[::3]:  # Her 3 noktadan 1'i
            y, x = spot
            
            # Rüzgar yönünde biraz ileride dökülmeli
            wind_rad = np.radians(wind_direction)
            offset_y = int(3 * np.cos(wind_rad))
            offset_x = int(3 * np.sin(wind_rad))
            
            drop_y = max(0, min(fire_grid.shape[0]-1, y + offset_y))
            drop_x = max(0, min(fire_grid.shape[1]-1, x + offset_x))
            
            drop_zones.append({
                'grid_y': drop_y,
                'grid_x': drop_x,
                'intensity': float(intensity_grid[y, x]),
                'priority': 'YÜKSEK' if intensity_grid[y, x] > 0.7 else 'ORTA'
            })
        
        # Yoğunluğa göre sırala
        drop_zones.sort(key=lambda x: x['intensity'], reverse=True)
        
        return drop_zones[:15]  # En kritik 15 nokta
    
    @staticmethod
    def convert_grid_to_gps(grid_y, grid_x, center_lat, center_lon, grid_size):
        """
        Grid koordinatını GPS'e çevir
        """
        # Grid merkezini bul
        center_grid_y = grid_size[0] // 2
        center_grid_x = grid_size[1] // 2
        
        # Offset hesapla (her piksel ~0.001 derece, yaklaşık 100m)
        offset_y = (grid_y - center_grid_y) * 0.001
        offset_x = (grid_x - center_grid_x) * 0.001
        
        gps_lat = center_lat - offset_y  # Kuzey-güney ters
        gps_lon = center_lon + offset_x
        
        return gps_lat, gps_lon

# ==================== GÖRSELLEŞTIRME ====================

def create_intervention_map(center_lat, center_lon, fire_grid, 
                           firebreak_locs, retardant_zones, grid_size):
    """
    Müdahale haritası oluştur
    """
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13)
    
    # Yangın alanı
    fire_points = np.argwhere(fire_grid > 0.5)
    for point in fire_points[::5]:
        lat, lon = InterventionPlanner.convert_grid_to_gps(
            point[0], point[1], center_lat, center_lon, grid_size
        )
        folium.CircleMarker(
            location=[lat, lon],
            radius=3,
            color='red',
            fill=True,
            fillColor='red',
            fillOpacity=0.6,
            popup='Yangın Alanı'
        ).add_to(m)
    
    # Firebreak konumları
    for idx, fb in enumerate(firebreak_locs, 1):
        lat, lon = InterventionPlanner.convert_grid_to_gps(
            fb['grid_y'], fb['grid_x'], center_lat, center_lon, grid_size
        )
        folium.Marker(
            location=[lat, lon],
            popup=f"🔨 Firebreak #{idx}<br>Öncelik: {fb['priority']:.2f}<br>Arazi: {CORINE_FUEL_MAP.get(fb['corine_class'], {}).get('name', 'Bilinmiyor')}",
            icon=folium.Icon(color='blue', icon='minus', prefix='fa')
        ).add_to(m)
    
    # Retardant dökülecek noktalar
    for idx, zone in enumerate(retardant_zones, 1):
        lat, lon = InterventionPlanner.convert_grid_to_gps(
            zone['grid_y'], zone['grid_x'], center_lat, center_lon, grid_size
        )
        folium.Marker(
            location=[lat, lon],
            popup=f"✈️ Retardant #{idx}<br>Öncelik: {zone['priority']}<br>Yoğunluk: {zone['intensity']:.2f}",
            icon=folium.Icon(color='orange', icon='plane', prefix='fa')
        ).add_to(m)
    
    return m

# ==================== ANA UYGULAMA ====================

def main():
    st.markdown("""
    <div style='background: linear-gradient(135deg, #c0392b 0%, #e74c3c 100%); 
                padding: 30px; border-radius: 15px; margin-bottom: 30px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.3);'>
        <h1 style='color: white; text-align: center; margin: 0; font-size: 42px;'>
            OPERASYONEL YANGIN YÖNETİM SİSTEMİ
        </h1>
        <p style='color: white; text-align: center; margin: 10px 0 0 0; font-size: 18px;'>
            • Adım Adım Müdahale Planı •
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # CORINE yükle
    corine_dataset, corine_profile = load_corine_data()
    
    if corine_dataset is None:
        st.error("❌ CORINE verisi yüklenemedi. Sistem çalışamaz.")
        return
    
    with st.sidebar:
        st.markdown("### ⚙️ OPERASYONEL AYARLAR")
        
        grid_size_option = st.selectbox(
            "📏 Grid boyutu",
            ["50x50 (Hızlı Test)", "100x100 (Normal)", "150x150 (Detaylı)"],
            index=1
        )
        grid_size = {
            "50x50 (Hızlı Test)": (50, 50),
            "100x100 (Normal)": (100, 100),
            "150x150 (Detaylı)": (150, 150)
        }[grid_size_option]
        
        sim_hours = st.slider("⏱️ Simülasyon süresi (saat)", 1, 24, 6)
        sim_steps = int(sim_hours * 60 / SimulationConfig.TIME_STEP)
        
        st.markdown("---")
        st.markdown("### 🔥 SİMÜLASYON PARAMETRELERİ")
        
        spread_intensity = st.select_slider(
            "Yayılım Hızı",
            options=["Çok Yavaş", "Yavaş", "Normal", "Hızlı", "Çok Hızlı"],
            value="Hızlı",
            help="Yangının ne kadar hızlı yayılacağını belirler"
        )
        
        # Spread multiplier mapping
        spread_multipliers = {
            "Çok Yavaş": 1.0,
            "Yavaş": 2.0,
            "Normal": 3.0,
            "Hızlı": 5.0,
            "Çok Hızlı": 8.0
        }
        
        random_seed = st.checkbox(
            "Rastgele Sonuçlar",
            value=False,
            help="Her simülasyonda farklı sonuç almak için işaretleyin"
        )
        
        st.markdown("---")
        st.markdown("### ÖZELLİKLER")
        st.success("""
        
        ✅ CORINE arazi verileri
        ✅ Open Weather hava durumu
        ✅ Adım adım müdahale planı
        ✅ Firebreak konumları (GPS)
        ✅ Retardant dökülecek noktalar
        ✅ Zaman bazlı tahminler
        ✅ İtfaiye araç rotaları
        """)
        
        st.markdown("---")
        st.info("""
        
        Bu sistem profesyonel müdahale 
        ekiplerine destek için tasarlanmıştır.
        
        Her adımı takip edin!
        """)
    
    # Session state başlatma
    if 'simulation_running' not in st.session_state:
        st.session_state.simulation_running = False
    if 'lat' not in st.session_state:
        st.session_state.lat = None
    if 'lon' not in st.session_state:
        st.session_state.lon = None
    if 'simulation_results' not in st.session_state:
        st.session_state.simulation_results = None
    if 'show_results' not in st.session_state:
        st.session_state.show_results = False
    
    st.markdown("### 🗺️ YANGIN KONUMUNU SEÇİN")
    
    # Sadece simülasyon çalışmıyorsa haritayı göster
    if not st.session_state.simulation_running:
        m = folium.Map(location=[39.0, 35.0], zoom_start=6)
        folium.plugins.Fullscreen().add_to(m)
        m.add_child(folium.LatLngPopup())
        
        map_data = st_folium(m, width=None, height=500, returned_objects=["last_clicked"], key="fire_map")
        
        if map_data and map_data.get("last_clicked"):
            st.session_state.lat = map_data["last_clicked"]["lat"]
            st.session_state.lon = map_data["last_clicked"]["lng"]
    
    # Konum seçildiyse göster
    if st.session_state.lat is not None and st.session_state.lon is not None:
        st.success(f"✅ Yangın konumu: **{st.session_state.lat:.4f}°K, {st.session_state.lon:.4f}°D**")
        
        # Simülasyon çalışmıyorsa buton göster
        if not st.session_state.simulation_running and not st.session_state.show_results:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🚨 OPERASYONEL SİMÜLASYONU BAŞLAT", 
                            use_container_width=True, type="primary"):
                    st.session_state.simulation_running = True
                    # rerun() KALDIRDIK - Sayfa otomatik yenilenecek
    else:
        # Konum seçilmemişse bilgi ver
        if not st.session_state.simulation_running:
            st.info("👆 Haritadan yangın konumunu seçmek için bir noktaya tıklayın")
    
    # Simülasyon çalışıyorsa veya sonuçlar varsa göster
    if st.session_state.simulation_running or st.session_state.show_results:
        lat = st.session_state.lat
        lon = st.session_state.lon
        
        # Üstte YENİ SİMÜLASYON butonu - her zaman görünür
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔄 YENİ SİMÜLASYON", use_container_width=True, type="secondary"):
                # Tüm state'i temizle
                st.session_state.simulation_running = False
                st.session_state.show_results = False
                st.session_state.simulation_results = None
                st.session_state.lat = None
                st.session_state.lon = None
                st.rerun()  # Sadece burada rerun - yeni simülasyon için
        
        st.markdown("---")
        
        try:
                
                st.markdown("---")
                st.markdown("## 📋 OPERASYONEL YANGIN ANALİZİ")
                
                # 1. KONUM BİLGİSİ
                with st.spinner("📍 Konum bilgileri alınıyor..."):
                    location = get_location_name(lat, lon)
                
                st.info(f"📍 **Konum:** {location['ilçe']} / {location['il']}")
                
                # 2. HAVA DURUMU
                with st.spinner("☁️ Gerçek zamanlı hava durumu çekiliyor..."):
                    weather = get_weather_data(lat, lon)
                
                st.markdown("### 🌤️ METEOROLOJ İ VERİLERİ (GERÇEK ZAMANLI)")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🌡️ Sıcaklık", f"{weather['sıcaklık']:.1f}°C")
                with col2:
                    st.metric("💧 Nem", f"{weather['nem']}%")
                with col3:
                    st.metric("💨 Rüzgar", f"{weather['rüzgar_hız']:.1f} m/s")
                with col4:
                    st.metric("🧭 Yön", f"{weather['rüzgar_yön']}°")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🔽 Basınç", f"{weather['basınç']} hPa")
                with col2:
                    st.metric("👁️ Görüş", f"{weather['görüş_mesafesi']:.1f} km")
                with col3:
                    st.metric("☁️ Bulut", f"{weather['bulut']}%")
                with col4:
                    st.metric("📝 Durum", weather['durum'])
                
                # 3. ARAZİ ANALİZİ
                with st.spinner("🗺️ Gerçek CORINE arazi verileri işleniyor..."):
                    local_corine, local_slope = extract_real_terrain(
                        corine_dataset, lat, lon, grid_size
                    )
                
                st.markdown("### 🌲 ARAZİ ANALİZİ (GERÇEK CORINE VERİSİ)")
                
                # CORINE sınıf dağılımı
                unique_classes, counts = np.unique(local_corine, return_counts=True)
                corine_df = pd.DataFrame({
                    'CORINE Kodu': unique_classes,
                    'Arazi Tipi': [CORINE_FUEL_MAP.get(int(c), {}).get('name', f'Sınıf {c}') 
                                  for c in unique_classes],
                    'Alan (%)': (counts / counts.sum() * 100).round(1),
                    'Yanabilirlik': [CORINE_FUEL_MAP.get(int(c), {}).get('flammability', 0) 
                                    for c in unique_classes]
                })
                corine_df = corine_df.sort_values('Alan (%)', ascending=False)
                
                st.dataframe(corine_df, use_container_width=True)
                
                # Dominant arazi tipi
                dominant_class = unique_classes[np.argmax(counts)]
                dominant_props = CORINE_FUEL_MAP.get(int(dominant_class), {})
                
                st.warning(f"""
                **🎯 Baskın Arazi Tipi:** {dominant_props.get('name', 'Bilinmiyor')}
                - Yakıt Yükü: {dominant_props.get('fuel_load', 0)} kg/m²
                - Yanabilirlik: {dominant_props.get('flammability', 0):.2%}
                - Tehlike Seviyesi: {'🔴 ÇOK YÜKSEK' if dominant_props.get('flammability', 0) > 0.8 else '🟠 YÜKSEK' if dominant_props.get('flammability', 0) > 0.5 else '🟡 ORTA'}
                """)
                
                # 4. SİMÜLASYON
                st.markdown("### 🔥 YANGIN SİMÜLASYONU")
                
                start_pos = (grid_size[0] // 2, grid_size[1] // 2)
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text(f"Simülasyon başlıyor... {sim_hours} saatlik tahmin")
                
                history, intensity_history, time_stamps = simulate_fire_operational(
                    start_pos, grid_size, weather, local_slope, local_corine,
                    steps=sim_steps, time_horizon_hours=sim_hours,
                    spread_multiplier=spread_multipliers[spread_intensity],
                    use_random_seed=not random_seed
                )
                
                progress_bar.progress(100)
                status_text.text(f"✅ Simülasyon tamamlandı: {len(history)} adım")
                
                # ========== YENİ: 2D IZGARA GÖRSELLEŞTİRMESİ ==========
                st.markdown("### 🔥 2D IZGARA SİMÜLASYONU")
                
                # Son yangın durumu
                final_fire_grid = history[-1]
                final_intensity = intensity_history[-1]
                
                # 2D Heatmap
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                
                # 1. Yangın Yayılımı
                im1 = axes[0].imshow(final_fire_grid, cmap='hot', interpolation='bilinear', 
                                    vmin=0, vmax=1, origin='lower')
                axes[0].set_title('🔥 Yangın Yayılımı', fontsize=14, fontweight='bold')
                axes[0].set_xlabel('X Grid')
                axes[0].set_ylabel('Y Grid')
                plt.colorbar(im1, ax=axes[0], label='Yanma Durumu (0=Yanmamış, 1=Yanmış)')
                axes[0].plot(start_pos[1], start_pos[0], 'g*', markersize=15, 
                           label='Başlangıç Noktası', markeredgecolor='white', markeredgewidth=1.5)
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                
                # 2. Yangın Yoğunluğu
                im2 = axes[1].imshow(final_intensity, cmap='YlOrRd', interpolation='bilinear', 
                                    origin='lower')
                axes[1].set_title('⚡ Yangın Yoğunluğu', fontsize=14, fontweight='bold')
                axes[1].set_xlabel('X Grid')
                axes[1].set_ylabel('Y Grid')
                plt.colorbar(im2, ax=axes[1], label='Yoğunluk (Rothermel Oranı)')
                axes[1].grid(True, alpha=0.3)
                
                # 3. CORINE Arazi Örtüsü
                im3 = axes[2].imshow(local_corine, cmap='terrain', interpolation='nearest', 
                                    origin='lower')
                axes[2].set_title('🗺️ CORINE Arazi Örtüsü', fontsize=14, fontweight='bold')
                axes[2].set_xlabel('X Grid')
                axes[2].set_ylabel('Y Grid')
                plt.colorbar(im3, ax=axes[2], label='CORINE Sınıf Kodu')
                
                # Yangın sınırını çiz
                from scipy.ndimage import binary_erosion
                fire_boundary = final_fire_grid - binary_erosion(final_fire_grid)
                y_bound, x_bound = np.where(fire_boundary > 0)
                axes[2].scatter(x_bound, y_bound, c='red', s=1, alpha=0.5, label='Yangın Sınırı')
                axes[2].legend()
                axes[2].grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # ========== YENİ: VORONOİ ZONLAMASI ==========
                st.markdown("### 🎯 VORONOİ RISK ZONLAMASI")
                
                # Yangın merkezlerini bul (en yüksek yoğunluklu noktalar)
                from scipy.spatial import Voronoi, voronoi_plot_2d
                
                # Yoğun yangın noktalarını seç (top 5)
                intensity_flat = final_intensity.flatten()
                top_indices = np.argsort(intensity_flat)[-5:]  # En yoğun 5 nokta
                fire_centers = []
                for idx in top_indices:
                    if intensity_flat[idx] > 0.1:  # Minimum yoğunluk
                        y = idx // grid_size[1]
                        x = idx % grid_size[1]
                        fire_centers.append([x, y])
                
                if len(fire_centers) >= 3:
                    # Voronoi diyagramı oluştur
                    fire_centers = np.array(fire_centers)
                    
                    # Grid kenarlarına dummy noktalar ekle (sınır sorunlarını çözmek için)
                    dummy_points = [
                        [0, 0], [grid_size[1]-1, 0], 
                        [0, grid_size[0]-1], [grid_size[1]-1, grid_size[0]-1],
                        [grid_size[1]//2, 0], [grid_size[1]//2, grid_size[0]-1],
                        [0, grid_size[0]//2], [grid_size[1]-1, grid_size[0]//2]
                    ]
                    all_points = np.vstack([fire_centers, dummy_points])
                    
                    vor = Voronoi(all_points)
                    
                    # Voronoi görselleştirmesi
                    fig_vor, ax_vor = plt.subplots(figsize=(12, 10))
                    
                    # Arka plan: Yangın yoğunluğu
                    im_bg = ax_vor.imshow(final_intensity, cmap='YlOrRd', alpha=0.6, 
                                         origin='lower', extent=[0, grid_size[1], 0, grid_size[0]])
                    
                    # Voronoi çizgilerini çiz
                    for simplex in vor.ridge_vertices:
                        simplex = np.asarray(simplex)
                        if np.all(simplex >= 0):  # Sonsuz çizgileri atla
                            ax_vor.plot(vor.vertices[simplex, 0], vor.vertices[simplex, 1], 
                                      'b-', linewidth=2, alpha=0.8)
                    
                    # Yangın merkezlerini işaretle
                    ax_vor.plot(fire_centers[:, 0], fire_centers[:, 1], 'r*', 
                              markersize=20, label='Yangın Merkezleri',
                              markeredgecolor='white', markeredgewidth=2)
                    
                    # Risk zonlarını numaralandır
                    for i, center in enumerate(fire_centers, 1):
                        ax_vor.text(center[0], center[1]+2, f'Zon {i}', 
                                  fontsize=12, fontweight='bold', color='white',
                                  ha='center', va='bottom',
                                  bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))
                    
                    ax_vor.set_xlim(0, grid_size[1])
                    ax_vor.set_ylim(0, grid_size[0])
                    ax_vor.set_xlabel('X Grid', fontsize=12)
                    ax_vor.set_ylabel('Y Grid', fontsize=12)
                    ax_vor.set_title('🎯 Voronoi Risk Zonları\n(Mavi çizgiler: Zon sınırları, Kırmızı yıldızlar: Yangın merkezleri)', 
                                   fontsize=14, fontweight='bold')
                    ax_vor.legend(loc='upper right', fontsize=11)
                    ax_vor.grid(True, alpha=0.3)
                    plt.colorbar(im_bg, ax=ax_vor, label='Yangın Yoğunluğu')
                    
                    st.pyplot(fig_vor)
                    plt.close()
                    
                    # Risk zonu açıklaması
                    st.info(f"""
                    **🎯 Voronoi Risk Zonları Açıklaması:**
                    
                    - Harita **{len(fire_centers)} risk zonuna** bölündü
                    - Her zon bir **yangın merkezine** göre tanımlandı
                    - **Mavi çizgiler**: Zonlar arası sınırlar
                    - **Kırmızı yıldızlar**: En yoğun yangın noktaları
                    - Her zon, kendi merkezine **en yakın** alanları içerir
                    
                    **Müdahale Stratejisi:**
                    - Her zona **ayrı bir ekip** atanmalı
                    - Zon sınırlarında **koordinasyon** kritik
                    - Merkezlere **öncelik** verilmeli
                    """)
                    
                else:
                    st.warning("⚠️ Voronoi diyagramı için yeterli yangın merkezi bulunamadı (minimum 3 gerekli)")
                
                # ========== YENİ: ZAMAN SERİSİ ANİMASYONU ==========
                st.markdown("### 📹 YANGIN YAYILIM ANİMASYONU")
                
                # Zaman adımı seçici
                time_step = st.slider(
                    "Zaman Adımı Seç (her adım 5 dakika)",
                    min_value=0,
                    max_value=len(history)-1,
                    value=0,
                    step=1,
                    key="time_slider"
                )
                
                # Seçilen adımı göster
                current_grid = history[time_step]
                elapsed_minutes = time_step * SimulationConfig.TIME_STEP
                
                fig_anim, ax_anim = plt.subplots(figsize=(10, 8))
                im_anim = ax_anim.imshow(current_grid, cmap='hot', interpolation='bilinear',
                                        vmin=0, vmax=1, origin='lower')
                ax_anim.set_title(f'🔥 Yangın Durumu - {elapsed_minutes} Dakika ({elapsed_minutes/60:.1f} saat)',
                                fontsize=14, fontweight='bold')
                ax_anim.set_xlabel('X Grid')
                ax_anim.set_ylabel('Y Grid')
                plt.colorbar(im_anim, ax=ax_anim, label='Yanma Durumu')
                ax_anim.plot(start_pos[1], start_pos[0], 'g*', markersize=15, 
                           label='Başlangıç', markeredgecolor='white', markeredgewidth=1.5)
                
                # Yanan alan hesapla
                burned_area_step = np.sum(current_grid) * (SimulationConfig.PIXEL_SIZE ** 2)
                ax_anim.text(0.02, 0.98, f'Yanan Alan: {burned_area_step:.2f} km²',
                           transform=ax_anim.transAxes, fontsize=12,
                           verticalalignment='top', bbox=dict(boxstyle='round', 
                           facecolor='wheat', alpha=0.8))
                
                ax_anim.legend()
                ax_anim.grid(True, alpha=0.3)
                
                st.pyplot(fig_anim)
                plt.close()
                
                # 5. MÜDAHALE PLANI OLUŞTUR
                st.markdown("### 🚨 MÜDAHALE PLANI")
                
                final_fire_grid = history[-1]
                final_intensity = intensity_history[-1]
                
                with st.spinner("📋 Müdahale planı oluşturuluyor..."):
                    planner = InterventionPlanner()
                    
                    # Firebreak konumları
                    firebreak_locs = planner.find_firebreak_locations(
                        final_fire_grid, local_corine, distance_km=0.5
                    )
                    
                    # Retardant dökülecek noktalar
                    retardant_zones = planner.find_retardant_drop_zones(
                        final_fire_grid, final_intensity, weather['rüzgar_yön']
                    )
                
                # ADIM ADIM TALİMATLAR
                st.markdown("## 📝 ADIM ADIM MÜDAHALE TALİMATLARI")
                
                st.markdown("### 🔴 ADIM 1: ACİL TAHLİYE")
                st.error("""
                **Tahliye edilecek bölgeler:**
                - Yangın merkezinden 2 km yarıçap içindeki tüm yerleşimler
                - Rüzgar yönündeki ({}°) 5 km'lik koridor
                
                **Tahliye rotaları:**
                - Ana yol: {} yönü
                - Yedek rota: {} yönü
                
                **Tahliye süresi:** Maksimum 2 saat içinde tamamlanmalı!
                """.format(
                    weather['rüzgar_yön'],
                    'Kuzey' if 315 <= weather['rüzgar_yön'] or weather['rüzgar_yön'] < 45 else 
                    'Güney' if 135 <= weather['rüzgar_yön'] < 225 else
                    'Batı' if 225 <= weather['rüzgar_yön'] < 315 else 'Doğu',
                    'Güney' if 315 <= weather['rüzgar_yön'] or weather['rüzgar_yön'] < 45 else 
                    'Kuzey' if 135 <= weather['rüzgar_yön'] < 225 else
                    'Doğu' if 225 <= weather['rüzgar_yön'] < 315 else 'Batı'
                ))
                
                st.markdown("### 🟠 ADIM 2: YANGIN ENGELLEME HATLARI (FIREBREAK)")
                st.warning(f"""
                **{len(firebreak_locs)} adet firebreak konumu belirlendi.**
                
                **Firebreak oluşturma öncelikleri:**
                """)
                
                for idx, fb in enumerate(firebreak_locs[:5], 1):
                    fb_lat, fb_lon = planner.convert_grid_to_gps(
                        fb['grid_y'], fb['grid_x'], lat, lon, grid_size
                    )
                    
                    corine_name = CORINE_FUEL_MAP.get(fb['corine_class'], {}).get('name', 'Bilinmiyor')
                    
                    st.info(f"""
                    **Firebreak #{idx}** (Öncelik: {'🔴 ÇOK YÜKSEK' if fb['priority'] > 0.8 else '🟠 YÜKSEK'})
                    - 📍 GPS: {fb_lat:.6f}°K, {fb_lon:.6f}°D
                    - 🌲 Arazi: {corine_name}
                    - 📏 Genişlik: {'30 metre' if fb['priority'] > 0.8 else '20 metre'}
                    - 🔨 Yöntem: {'Buldozer + Kimyasal' if fb['priority'] > 0.8 else 'Mekanik temizleme'}
                    - ⏱️ Tahmini süre: {'2-3 saat' if fb['priority'] > 0.8 else '1-2 saat'}
                    """)
                
                st.markdown("### 🟡 ADIM 3: HAVADAN RETARDANT DÖKÜMÜ")
                st.warning(f"""
                **{len(retardant_zones)} adet retardant dökülecek nokta belirlendi.**
                
                **Uçak/Helikopter koordinasyonu:**
                """)
                
                for idx, zone in enumerate(retardant_zones[:5], 1):
                    zone_lat, zone_lon = planner.convert_grid_to_gps(
                        zone['grid_y'], zone['grid_x'], lat, lon, grid_size
                    )
                    
                    st.info(f"""
                    **Dökülecek Nokta #{idx}** (Öncelik: {zone['priority']})
                    - 📍 GPS: {zone_lat:.6f}°K, {zone_lon:.6f}°D
                    - 🔥 Yangın Yoğunluğu: {zone['intensity']:.2f}
                    - ✈️ Önerilen Araç: {'Ağır Helikopter (10,000L)' if zone['priority'] == 'YÜKSEK' else 'Orta Helikopter (5,000L)'}
                    - 💧 Retardant Tipi: {'Uzun etkili (Class A)' if zone['priority'] == 'YÜKSEK' else 'Standart'}
                    - 🎯 Dökülecek Alan: 50m x 50m
                    """)
                
                st.markdown("### 🟢 ADIM 4: KARA EKİPLERİ KONUŞLANDIRMA")
                st.success("""
                **İtfaiye araçları:**
                - Ana ekip: Yangın merkezinin batı yakasına konuşlanacak
                - Destek ekibi: Kuzey flanklarını güvence altına alacak
                - Yedek ekip: Tahliye rotalarını koruyacak
                
                **Gerekli ekipman:**
                - 15 itfaiye aracı (su tankeri)
                - 8 dozer/greyder (firebreak için)
                - 5 ambulans (sağlık)
                - 3 komuta aracı
                
                **İletişim:**
                - Frekans: 156.800 MHz (VHF)
                - Yedek: 462.675 MHz (UHF)
                """)
                
                # 6. MÜDAHALE HARİTASI
                st.markdown("### 🗺️ MÜDAHALE HARİTASI")
                
                intervention_map = create_intervention_map(
                    lat, lon, final_fire_grid,
                    firebreak_locs, retardant_zones, grid_size
                )
                
                st_folium(intervention_map, width=None, height=600)
                
                st.info("""
                **🗺️ Harita Açıklaması:**
                - 🔴 Kırmızı noktalar: Yangın alanı
                - 🔵 Mavi işaretler: Firebreak konumları
                - 🟠 Turuncu işaretler: Retardant dökülecek noktalar
                
                **İşaretlere tıklayarak detaylı bilgi alabilirsiniz!**
                """)
                
                # 7. ZAMAN BAZLI TAHMİNLER
                st.markdown("### ⏰ ZAMAN BAZLI TAHMİNLER")
                
                # 1 saat, 3 saat, 6 saat sonraki durumu göster
                time_points = [0, int(len(history)*0.17), int(len(history)*0.5), len(history)-1]
                time_labels = ['ŞİMDİ', '1 SAAT SONRA', '3 SAAT SONRA', f'{sim_hours} SAAT SONRA']
                
                cols = st.columns(4)
                for idx, (time_idx, label) in enumerate(zip(time_points, time_labels)):
                    with cols[idx]:
                        burned_area = np.sum(history[time_idx]) * (SimulationConfig.PIXEL_SIZE ** 2)
                        
                        st.metric(
                            label=label,
                            value=f"{burned_area:.1f} km²",
                            delta=f"+{burned_area-np.sum(history[0])*(SimulationConfig.PIXEL_SIZE**2):.1f} km²" if time_idx > 0 else "Başlangıç"
                        )
                
                # 8. İSTATİSTİKLER
                st.markdown("### 📊 YANGIN İSTATİSTİKLERİ")
                
                final_burned = np.sum(final_fire_grid) * (SimulationConfig.PIXEL_SIZE ** 2)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🔥 Yanan Alan", f"{final_burned:.2f} km²")
                with col2:
                    st.metric("📏 Çevre Uzunluğu", 
                             f"{np.sum(planner.calculate_fire_perimeter(final_fire_grid))*0.1:.1f} km")
                with col3:
                    st.metric("⚡ Maks. Yoğunluk", f"{np.max(final_intensity):.2f}")
                with col4:
                    st.metric("⏱️ Simülasyon Süresi", f"{sim_hours} saat")
                
                # 9. UYARILAR VE TAVSİYELER
                st.markdown("### ⚠️ UYARILAR VE TAVSİYELER")
                
                # Risk seviyesi
                avg_flammability = np.mean([
                    CORINE_FUEL_MAP.get(int(c), {}).get('flammability', 0)
                    for c in np.unique(local_corine)
                ])
                
                if avg_flammability > 0.7 or weather['rüzgar_hız'] > 10 or weather['nem'] < 30:
                    st.error("""
                    🔴 **EKSTREM TEHLİKE!**
                    
                    - Yangın hızla yayılma potansiyeli çok yüksek
                    - Spot yangınlar (uzak tutuşmalar) beklenmeli
                    - Gece müdahalesi zorunlu
                    - Ek takviye ekipler talep edilmeli
                    - Sivil havacılık bölgeyi terk etmeli
                    """)
                elif avg_flammability > 0.5 or weather['rüzgar_hız'] > 5:
                    st.warning("""
                    🟠 **YÜKSEK TEHLİKE**
                    
                    - Yangın kontrol altına alınabilir
                    - Standart protokoller uygulanmalı
                    - Hava desteği efektif olacak
                    - 24-48 saat içinde kontrol mümkün
                    """)
                else:
                    st.info("""
                    🟡 **ORTA TEHLİKE**
                    
                    - Yangın yönetilebilir seviyede
                    - Hızlı müdahale ile kontrol mümkün
                    - Standart prosedürler yeterli
                    """)
                
                # 10. RAPOR İNDİRME
                st.markdown("### 📥 RAPOR İNDİRME")
                
                # Rapor oluştur
                report_text = f"""
OPERASYONEL YANGIN YÖNETİM RAPORU
Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M')}
=====================================

KONUM BİLGİLERİ:
- Koordinat: {lat:.6f}°K, {lon:.6f}°D
- Yer: {location['ilçe']}, {location['il']}

METEOROLOJ İ (GERÇEK ZAMANLI):
- Sıcaklık: {weather['sıcaklık']:.1f}°C
- Nem: {weather['nem']}%
- Rüzgar: {weather['rüzgar_hız']:.1f} m/s, {weather['rüzgar_yön']}°
- Basınç: {weather['basınç']} hPa

ARAZİ ANALİZİ (CORINE):
- Baskın arazi: {dominant_props.get('name', 'Bilinmiyor')}
- Yanabilirlik: {dominant_props.get('flammability', 0):.1%}

SİMÜLASYON SONUÇLARI:
- Yanan alan: {final_burned:.2f} km²
- Simülasyon süresi: {sim_hours} saat

MÜDAHALE PLANI:
- Firebreak sayısı: {len(firebreak_locs)}
- Retardant noktası: {len(retardant_zones)}

İLK 5 FİREBREAK KONUMU:
"""
                for idx, fb in enumerate(firebreak_locs[:5], 1):
                    fb_lat, fb_lon = planner.convert_grid_to_gps(
                        fb['grid_y'], fb['grid_x'], lat, lon, grid_size
                    )
                    report_text += f"{idx}. GPS: {fb_lat:.6f}°K, {fb_lon:.6f}°D\n"
                
                report_text += f"\nİLK 5 RETARDANT NOKTASI:\n"
                for idx, zone in enumerate(retardant_zones[:5], 1):
                    zone_lat, zone_lon = planner.convert_grid_to_gps(
                        zone['grid_y'], zone['grid_x'], lat, lon, grid_size
                    )
                    report_text += f"{idx}. GPS: {zone_lat:.6f}°K, {zone_lon:.6f}°D (Öncelik: {zone['priority']})\n"
                
                st.download_button(
                    label="📄 Müdahale Raporunu İndir (TXT)",
                    data=report_text,
                    file_name=f"yangin_mudahale_raporu_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
                
                # Başarı mesajı
                st.success("""
                ### ✅ OPERASYONEL ANALİZ TAMAMLANDI!
                
                Yukarıdaki adım adım talimatları takip ederek müdahale gerçekleştirin.
                
                **Önemli:** Bu raporu kriz masasıyla paylaşın!
                """)
                
                # Simülasyon tamamlandı - state'i güncelle
                st.session_state.simulation_running = False
                st.session_state.show_results = True
                
                st.success("✅ **SİMÜLASYON TAMAMLANDI!** Sonuçlar ekranda kalacak. Yeni simülasyon için yukarıdaki '🔄 YENİ SİMÜLASYON' butonuna tıklayın.")
                
        except Exception as e:
            st.error(f"❌ **Simülasyon Hatası:** {str(e)}")
            st.warning("⚠️ Bir hata oluştu. Lütfen farklı bir konum seçerek tekrar deneyin.")
            
            # State'i koruyalım - kullanıcı düzeltme yapabilsin
            st.session_state.simulation_running = False
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🔄 Farklı Konum ile Tekrar Dene", use_container_width=True):
                    st.session_state.lat = None
                    st.session_state.lon = None
                    st.session_state.show_results = False
                    st.rerun()

if __name__ == "__main__":
    main()
