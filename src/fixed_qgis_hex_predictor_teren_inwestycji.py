#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 00:37:03 2025

@author: adrian
"""

import pandas as pd
import numpy as np
from pathlib import Path
import processing
import os
from qgis.core import ( QgsProject, QgsWkbTypes,QgsMessageLog,QgsField,
                       QgsVectorLayer, QgsFields, QgsFeature) 
from PyQt5.QtCore import QVariant
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import pickle
import math
from collections import defaultdict
import os
import sys
# Wyłącz ostrzeżenia XCB
os.environ['QT_LOGGING_RULES'] = 'qt.qpa.xcb.warning=false'
os.environ['QT_X11_NO_MITSHM'] = '1'

# Dodatkowe zmienne środowiskowe dla Qt
os.environ['QT_AUTO_SCREEN_SCALE_FACTOR'] = '0'
os.environ['QT_SCREEN_SCALE_FACTORS'] = '1'

# Obsługa błędów Qt - dodaj po importach PyQt5
from PyQt5.QtCore import QVariant
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='PyQt5')

# Funkcja do bezpiecznego dodawania warstw
def safe_add_layer_to_project(layer, layer_name=None):
    """Bezpiecznie dodaje warstwę do projektu z obsługą błędów XCB"""
    try:
        if layer_name:
            layer.setName(layer_name)
        
        # Sprawdź czy warstwa już istnieje
        existing_layers = [l.name() for l in QgsProject.instance().mapLayers().values()]
        if layer.name() in existing_layers:
            print(f"Warstwa {layer.name()} już istnieje - usuwam starą")
            for l in QgsProject.instance().mapLayers().values():
                if l.name() == layer.name():
                    QgsProject.instance().removeMapLayer(l.id())
                    break
        
        # Dodaj warstwę
        QgsProject.instance().addMapLayer(layer, False)  # False = nie dodawaj do legend od razu
        
        # Następnie dodaj do legend tree (bezpieczniej)
        root = QgsProject.instance().layerTreeRoot()
        root.addLayer(layer)
        
        print(f"✅ Bezpiecznie dodano warstwę: {layer.name()}")
        return True
        
    except Exception as e:
        print(f"❌ Błąd dodawania warstwy {layer.name()}: {e}")
        return False

# Funkcja do bezpiecznego usuwania warstw memory
def safe_remove_memory_layers():
    """Bezpiecznie usuwa warstwy memory"""
    try:
        layers_to_remove = []
        for layer_id, layer in QgsProject.instance().mapLayers().items():
            if hasattr(layer, 'dataProvider') and layer.dataProvider().name() == 'memory':
                layers_to_remove.append(layer_id)
        
        for layer_id in layers_to_remove:
            QgsProject.instance().removeMapLayer(layer_id)
            
        print(f"Usunięto {len(layers_to_remove)} warstw memory")
        
    except Exception as e:
        print(f"Błąd usuwania warstw memory: {e}")

# Funkcja do bezpiecznego odświeżania canvas
def safe_refresh_canvas():
    """Bezpiecznie odświeża canvas"""
    try:
        from qgis.utils import iface
        if iface and iface.mapCanvas():
            iface.mapCanvas().refresh()
    except:
        pass  # Ignoruj błędy odświeżania
# Optymalizacja dla słabszego sprzętu
torch.set_num_threads(4)
project_path = QgsProject.instance().fileName()
project_directory = os.path.dirname(project_path)

# === Parametry ===
INPUT_FEATURES = ['Z', 'Intensity', 'ReturnNumber', 'NumberOfReturns', 'Red', 'Green', 'Blue']
INPUT_CSV = f"{project_directory}/Classification_2_przyciete.csv"
OUTPUT_CSV = f"{project_directory}/punkty_pbc_wyniki_predykcji_teren_inwestycji.csv"
SCRIPTS_PATH = os.path.dirname(os.path.abspath(__file__))
model_name = "best_hex_model.pth"
MODEL_PATH = os.path.join(SCRIPTS_PATH, model_name)
scaler_name = "scaler_hex.pkl"  # Zmieniono na .pkl
SCALER_PATH = os.path.join(SCRIPTS_PATH, scaler_name)

HEX_SIZE = 1.0  # Rozmiar heksagonu użyty podczas treningu

# === POPRAWIONY Model heksagonalny ===
class HexTerrainNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim // 4, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

def hex_grid_coordinates(x, y, hex_size):
    """Konwertuje współrzędne kartezjańskie na współrzędne siatki heksagonalnej"""
    q = (2/3 * x) / hex_size
    r = (-1/3 * x + math.sqrt(3)/3 * y) / hex_size
    
    # Zaokrąglenie do najbliższego heksagonu
    q_round = round(q)
    r_round = round(r)
    s_round = round(-q - r)
    
    # Korekcja zaokrągleń
    q_diff = abs(q_round - q)
    r_diff = abs(r_round - r)
    s_diff = abs(s_round - (-q - r))
    
    if q_diff > r_diff and q_diff > s_diff:
        q_round = -r_round - s_round
    elif r_diff > s_diff:
        r_round = -q_round - s_round
    else:
        s_round = -q_round - r_round
    
    return (q_round, r_round)

def create_hexagon_features(points_data):
    """Grupuje punkty w heksagony i tworzy cechy dla każdego heksagonu"""
    hex_groups = defaultdict(list)
    
    print("Grupowanie punktów w heksagony...")
    for idx, row in points_data.iterrows():
        hex_coord = hex_grid_coordinates(row['X'], row['Y'], HEX_SIZE)
        hex_groups[hex_coord].append(row)
    
    print(f"Utworzono {len(hex_groups)} heksagonów z {len(points_data)} punktów")
    
    hex_features = []
    hex_labels = []
    hex_coords = []  # Do mapowania z powrotem na punkty
    
    for hex_coord, points in hex_groups.items():
        if len(points) < 3:  # Pomijamy heksagony z małą liczbą punktów
            continue
            
        points_df = pd.DataFrame(points)
        
        # Podstawowe statystyki dla każdej cechy
        features = []
        
        for feature in INPUT_FEATURES:
            values = points_df[feature].values
            if len(values) > 0:
                features.extend([
                    np.mean(values),      # średnia
                    np.std(values) if len(values) > 1 else 0.0,       # odchylenie standardowe
                    np.min(values),       # minimum
                    np.max(values),       # maksimum
                    np.percentile(values, 25),  # kwartyl dolny
                    np.percentile(values, 75),  # kwartyl górny
                ])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Dodatkowe cechy przestrzenne
        features.extend([
            len(points),  # liczba punktów w heksagonie
            points_df['Z'].max() - points_df['Z'].min(),  # różnica wysokości
            np.std(points_df['Z']),  # szorstkość powierzchni
        ])
        
        # Cechy dotyczące zwrotów lasera
        if 'ReturnNumber' in points_df.columns and 'NumberOfReturns' in points_df.columns:
            first_returns = (points_df['ReturnNumber'] == 1).sum()
            features.append(first_returns / len(points))
            
            last_returns = (points_df['ReturnNumber'] == points_df['NumberOfReturns']).sum()
            features.append(last_returns / len(points))
            
            features.append(points_df['NumberOfReturns'].mean())
        
        # Cechy kolorystyczne
        if all(col in points_df.columns for col in ['Red', 'Green', 'Blue']):
            ndvi_like = (points_df['Green'] - points_df['Red']) / (points_df['Green'] + points_df['Red'] + 1e-8)
            features.extend([
                np.mean(ndvi_like),
                np.std(ndvi_like)
            ])
            
            brightness = (points_df['Red'] + points_df['Green'] + points_df['Blue']) / 3
            features.extend([
                np.mean(brightness),
                np.std(brightness)
            ])
        
        hex_features.append(features)
        hex_coords.append(hex_coord)
        
        # Etykieta - najczęstsza klasa w heksagonie (jeśli dostępna)
        if 'label' in points_df.columns:
            labels = points_df['label'].values
            unique_labels, counts = np.unique(labels, return_counts=True)
            dominant_label = int(unique_labels[np.argmax(counts)])
            hex_labels.append(dominant_label)
        else:
            hex_labels.append(0)  # domyślna wartość dla predykcji
    
    return np.array(hex_features), np.array(hex_labels), hex_coords, hex_groups

def map_hex_predictions_to_points(hex_predictions, hex_coords, hex_groups, original_points):
    """Mapuje predykcje z heksagonów z powrotem na punkty"""
    point_predictions = np.zeros(len(original_points))
    
    for i, (hex_coord, prediction) in enumerate(zip(hex_coords, hex_predictions)):
        if hex_coord in hex_groups:
            points_in_hex = hex_groups[hex_coord]
            for point in points_in_hex:
                # Znajdź indeks tego punktu w oryginalnych danych
                point_idx = point.name if hasattr(point, 'name') else point['original_index']
                if point_idx < len(point_predictions):
                    point_predictions[point_idx] = prediction
    
    return point_predictions

class QGISHexLidarPredictor:
    """Klasa do predykcji LIDAR z heksagonami - gotowa do użycia w QGIS"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.hex_size = HEX_SIZE
        self.input_dim = None
        
    def load_model(self, model_path, scaler_path):
        try:
            # Wczytaj checkpoint
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Odczytaj parametry modelu z checkpointu
            self.input_dim = checkpoint.get('input_dim', 52)  # domyślnie 52 dla heksagonów
            num_classes = checkpoint.get('num_classes', 5)
            self.hex_size = checkpoint.get('hex_size', HEX_SIZE)
            
            # Stwórz model o odpowiedniej architekturze
            self.model = HexTerrainNet(input_dim=self.input_dim, output_dim=num_classes)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            # Wczytaj scaler
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
                
            print(f"Model załadowany: input_dim={self.input_dim}, output_dim={num_classes}")
            print(f"Scaler załadowany z {scaler_path}")
            return True
            
        except Exception as e:
            print(f"Błąd ładowania modelu: {e}")
            return False
    
    def predict_points(self, points_df):
        """Predykcja dla punktów przez heksagony"""
        try:
            # Dodaj indeksy oryginalnych punktów
            points_df = points_df.copy()
            points_df['original_index'] = range(len(points_df))
            
            # Stwórz cechy heksagonalne
            X_hex, _, hex_coords, hex_groups = create_hexagon_features(points_df)
            
            if len(X_hex) == 0:
                print("Brak heksagonów do predykcji!")
                return np.zeros(len(points_df))
            
            print(f"Stworzono {len(X_hex)} heksagonów o wymiarach {X_hex.shape}")
            
            # Normalizacja
            X_scaled = self.scaler.transform(X_hex)
            
            # Predykcja
            predictions = []
            batch_size = 64
            
            with torch.no_grad():
                for i in range(0, len(X_scaled), batch_size):
                    batch = X_scaled[i:i+batch_size]
                    X_tensor = torch.tensor(batch, dtype=torch.float32)
                    logits = self.model(X_tensor)
                    batch_preds = torch.argmax(logits, dim=1).numpy()
                    predictions.extend(batch_preds)
            
            predictions = np.array(predictions)
            
            # Mapuj predykcje z heksagonów na punkty
            point_predictions = map_hex_predictions_to_points(
                predictions, hex_coords, hex_groups, points_df
            )
            
            return point_predictions
            
        except Exception as e:
            print(f"Błąd predykcji: {e}")
            return np.zeros(len(points_df))

# Pozostałe funkcje pomocnicze - bez zmian
def layer_to_df(layer):
    data = []
    for f in layer.getFeatures():
        data.append(f.attributes())
    return pd.DataFrame(data, columns=[field.name() for field in layer.fields()])

def wczytaj_csv_do_qgis(sciezka_csv, nazwa_kolumny_x='X', nazwa_kolumny_y='Y', 
                        separator=',', crs_kod='EPSG:2180', nazwa_warstwy=None, 
                        kolumny_int=['predicted_label']):
    if not os.path.exists(sciezka_csv):
        print(f"Błąd: Plik {sciezka_csv} nie istnieje!")
        return None
    
    if nazwa_warstwy is None:
        nazwa_warstwy = os.path.splitext(os.path.basename(sciezka_csv))[0]
    
    uri = f"file:///{sciezka_csv}?delimiter={separator}&xField={nazwa_kolumny_x}&yField={nazwa_kolumny_y}&crs={crs_kod}&detectTypes=no"
    warstwa = QgsVectorLayer(uri, nazwa_warstwy, "delimitedtext")
    
    if not warstwa.isValid():
        print(f"Błąd: Nie można wczytać warstwy z pliku {sciezka_csv}")
        return None
    
    safe_add_layer_to_project(warstwa)
    
    if kolumny_int:
        konwertuj_kolumny_na_int(warstwa, kolumny_int)
    
    print(f"Wczytano warstwę: {nazwa_warstwy} ({warstwa.featureCount()} obiektów)")
    return warstwa

def konwertuj_kolumny_na_int(warstwa, nazwy_kolumn):
    provider = warstwa.dataProvider()
    
    for nazwa_kolumny in nazwy_kolumn:
        field_index = warstwa.fields().lookupField(nazwa_kolumny)
        if field_index == -1:
            continue
            
        nowe_pole = QgsField(f"{nazwa_kolumny}_int", QVariant.Int)
        provider.addAttributes([nowe_pole])
        warstwa.updateFields()
        
        nowy_field_index = warstwa.fields().lookupField(f"{nazwa_kolumny}_int")
        
        warstwa.startEditing()
        for feature in warstwa.getFeatures():
            stara_wartosc = feature[nazwa_kolumny]
            try:
                nowa_wartosc = int(str(stara_wartosc)) if stara_wartosc is not None else 0
                warstwa.changeAttributeValue(feature.id(), nowy_field_index, nowa_wartosc)
            except (ValueError, TypeError):
                warstwa.changeAttributeValue(feature.id(), nowy_field_index, 0)
        
        warstwa.commitChanges()
        provider.deleteAttributes([field_index])
        warstwa.updateFields()
        
        warstwa.startEditing()
        nowy_field_index = warstwa.fields().lookupField(f"{nazwa_kolumny}_int")
        provider.renameAttributes({nowy_field_index: nazwa_kolumny})
        warstwa.commitChanges()
        warstwa.updateFields()
    
    return warstwa

def remove_memory_layers():
    for lyr in QgsProject.instance().mapLayers().values():
        if lyr.dataProvider().name() == 'memory':
            QgsProject.instance().removeMapLayer(lyr.id())

def apply_qml_style_to_layer(layer, qml_file_path=None, show_messages=True):
    if isinstance(layer, str):
        layer_name = layer
        layer = None
        for lyr in QgsProject.instance().mapLayers().values():
            if lyr.name() == layer_name:
                layer = lyr
                break
        
        if layer is None:
            if show_messages:
                print(f"Nie znaleziono warstwy: {layer_name}")
            return False
    
    if not os.path.exists(qml_file_path):
        if show_messages:
            print(f"Plik QML nie istnieje: {qml_file_path}")
        return False
    
    try:
        result = layer.loadNamedStyle(qml_file_path)
        if result[1]:
            layer.triggerRepaint()
            if show_messages:
                print(f"Styl zastosowany do warstwy: {layer.name()}")
            return True
        else:
            if show_messages:
                print(f"Nie udało się załadować stylu: {result[0]}")
            return False
    except Exception as e:
        if show_messages:
            print(f"Błąd ładowania stylu: {str(e)}")
        return False


def diagnose_layers(layer1, layer2):
    """Diagnozuje warstwy przed przycinaniem"""
    print("\n🔍 DIAGNOSTYKA WARSTW:")
    
    # Sprawdzenie CRS
    crs1 = layer1.crs().authid()
    crs2 = layer2.crs().authid()
    print(f"CRS warstwy punktowej: {crs1}")
    print(f"CRS warstwy maski: {crs2}")
    
    if crs1 != crs2:
        print("⚠️ UWAGA: Różne układy współrzędnych!")
        print("Może to być przyczyną braku przecięć")
    
    # Sprawdzenie zakresów
    extent1 = layer1.extent()
    extent2 = layer2.extent()
    
    print("\nZakres warstwy punktowej:")
    print(f"  X: {extent1.xMinimum():.2f} - {extent1.xMaximum():.2f}")
    print(f"  Y: {extent1.yMinimum():.2f} - {extent1.yMaximum():.2f}")
    
    print("\nZakres warstwy maski:")
    print(f"  X: {extent2.xMinimum():.2f} - {extent2.xMaximum():.2f}")
    print(f"  Y: {extent2.yMinimum():.2f} - {extent2.yMaximum():.2f}")
    
    # Sprawdzenie czy zakresy się przecinają
    intersection = extent1.intersect(extent2)
    if intersection.isEmpty():
        print("❌ ZAKRESY SIĘ NIE PRZECINAJĄ!")
        return False
    else:
        print("✅ Zakresy się przecinają:")
        print(f"  X: {intersection.xMinimum():.2f} - {intersection.xMaximum():.2f}")
        print(f"  Y: {intersection.yMinimum():.2f} - {intersection.yMaximum():.2f}")
    
    # Sprawdzenie typów geometrii
    print(f"\nTyp geometrii punktów: {layer1.geometryType()}")
    print(f"Typ geometrii maski: {layer2.geometryType()}")
    
    return True


def clip_layer_advanced(punkt_layer, mask_layer, output_name='Classification_2_przyciete'):
    """
    Zaawansowane przycinanie warstwy z różnymi metodami
    """
    print("\n🔧 ROZPOCZYNAM ZAAWANSOWANE PRZYCINANIE...")
    
    # Diagnostyka
    if not diagnose_layers(punkt_layer, mask_layer):
        print("❌ Problemy z warstwami - przywanie może się nie powieść")
        return None
    
    # Metoda 1: Extract by location (oryginalna)
    print("\n📍 METODA 1: Extract by location...")
    try:
        parametry = {
            'INPUT': punkt_layer,
            'PREDICATE': [0],  # intersect
            'INTERSECT': mask_layer,
            'OUTPUT': f'memory:{output_name}'
        }
        
        wynik = processing.run("native:extractbylocation", parametry)
        warstwa_wynik = wynik['OUTPUT']
        
        if warstwa_wynik.featureCount() > 0:
            print(f"✅ SUKCES! Metoda 1: {warstwa_wynik.featureCount()} obiektów")
            safe_add_layer_to_project(warstwa_wynik)
            return warstwa_wynik
        else:
            print("⚠️ Metoda 1: 0 obiektów")
    except Exception as e:
        print(f"❌ Błąd metody 1: {e}")
    
    # Metoda 2: Inne predykaty
    predykaty_do_testow = [
        ([1], "contain"),
        ([2], "disjoint"), 
        ([3], "equal"),
        ([4], "touch"),
        ([5], "overlap"),
        ([6], "are within"),
        ([7], "cross")
    ]
    
    for predykat, nazwa in predykaty_do_testow:
        print(f"\n📍 METODA 2.{predykat[0]}: Extract by location ({nazwa})...")
        try:
            parametry = {
                'INPUT': punkt_layer,
                'PREDICATE': predykat,
                'INTERSECT': mask_layer,
                'OUTPUT': f'memory:{output_name}_{nazwa.replace(" ", "_")}'
            }
            
            wynik = processing.run("native:extractbylocation", parametry)
            warstwa_wynik = wynik['OUTPUT']
            
            if warstwa_wynik.featureCount() > 0:
                print(f"✅ SUKCES! Metoda 2.{predykat[0]}: {warstwa_wynik.featureCount()} obiektów")
                safe_add_layer_to_project(warstwa_wynik)
                return warstwa_wynik
            else:
                print(f"⚠️ Metoda 2.{predykat[0]}: 0 obiektów")
        except Exception as e:
            print(f"❌ Błąd metody 2.{predykat[0]}: {e}")
    
    # Metoda 3: Clip zamiast extract
    print("\n✂️ METODA 3: Clip...")
    try:
        parametry = {
            'INPUT': punkt_layer,
            'OVERLAY': mask_layer,
            'OUTPUT': f'memory:{output_name}_clip'
        }
        
        wynik = processing.run("native:clip", parametry)
        warstwa_wynik = wynik['OUTPUT']
        
        if warstwa_wynik.featureCount() > 0:
            print(f"✅ SUKCES! Metoda 3: {warstwa_wynik.featureCount()} obiektów")
            safe_add_layer_to_project(warstwa_wynik)
            return warstwa_wynik
        else:
            print("⚠️ Metoda 3: 0 obiektów")
    except Exception as e:
        print(f"❌ Błąd metody 3: {e}")
    
    # Metoda 4: Reprojekcja i ponowna próba
    print("\n🔄 METODA 4: Z reprojekcją...")
    try:
        # Reprojekcja maski do CRS punktów
        target_crs = punkt_layer.crs()
        
        parametry_reproject = {
            'INPUT': mask_layer,
            'TARGET_CRS': target_crs,
            'OUTPUT': 'memory:mask_reprojected'
        }
        
        wynik_reproject = processing.run("native:reprojectlayer", parametry_reproject)
        mask_reprojected = wynik_reproject['OUTPUT']
        
        print(f"Reprojekcja wykonana: {mask_reprojected.crs().authid()}")
        
        # Teraz przycinanie
        parametry = {
            'INPUT': punkt_layer,
            'PREDICATE': [0],
            'INTERSECT': mask_reprojected,
            'OUTPUT': f'memory:{output_name}_reproj'
        }
        
        wynik = processing.run("native:extractbylocation", parametry)
        warstwa_wynik = wynik['OUTPUT']
        
        if warstwa_wynik.featureCount() > 0:
            print(f"✅ SUKCES! Metoda 4: {warstwa_wynik.featureCount()} obiektów")
            safe_add_layer_to_project(warstwa_wynik)
            return warstwa_wynik
        else:
            print("⚠️ Metoda 4: 0 obiektów")
            
    except Exception as e:
        print(f"❌ Błąd metody 4: {e}")
    
    # Metoda 5: Ręczne sprawdzenie przecięć (próbka)
    print("\n🔍 METODA 5: Ręczne sprawdzenie próbki...")
    try:
        # Pobierz pierwszych 10 punktów i sprawdź czy przecinają się z maską
        punkty_sample = []
        for i, punkt_feature in enumerate(punkt_layer.getFeatures()):
            if i >= 10:  # Tylko 10 pierwszych
                break
            punkty_sample.append(punkt_feature)
        
        # Pobierz geometrie maski
        mask_geometries = []
        for mask_feature in mask_layer.getFeatures():
            mask_geometries.append(mask_feature.geometry())
        
        print(f"Sprawdzam {len(punkty_sample)} punktów z {len(mask_geometries)} poligonami...")
        
        przeciecia = 0
        for punkt_feature in punkty_sample:
            punkt_geom = punkt_feature.geometry()
            for mask_geom in mask_geometries:
                if punkt_geom.intersects(mask_geom):
                    przeciecia += 1
                    break
        
        print(f"Znaleziono {przeciecia} przecięć w próbce")
        
        if przeciecia == 0:
            print("❌ Brak przecięć nawet w próbce - warstwy rzeczywiście się nie przecinają!")
        
    except Exception as e:
        print(f"❌ Błąd metody 5: {e}")
    
    print("\n❌ WSZYSTKIE METODY ZAWIODŁY - brak przecięć między warstwami!")
    return None

def simple_buffer_test(punkt_layer, mask_layer):
    """
    Test z buforem - czasem pomaga
    """
    print("\n🎯 TEST Z BUFOREM...")
    
    try:
        # Stwórz mały bufor wokół maski (1 metr)
        parametry_buffer = {
            'INPUT': mask_layer,
            'DISTANCE': 1,
            'OUTPUT': 'memory:mask_buffered'
        }
        
        wynik_buffer = processing.run("native:buffer", parametry_buffer)
        mask_buffered = wynik_buffer['OUTPUT']
        
        print("Utworzono bufor wokół maski")
        
        # Teraz przycinanie z buforem
        parametry = {
            'INPUT': punkt_layer,
            'PREDICATE': [0],
            'INTERSECT': mask_buffered,
            'OUTPUT': 'memory:Classification2_buffered'
        }
        
        wynik = processing.run("native:extractbylocation", parametry)
        warstwa_wynik = wynik['OUTPUT']
        
        if warstwa_wynik.featureCount() > 0:
            print(f"✅ SUKCES Z BUFOREM: {warstwa_wynik.featureCount()} obiektów")
            safe_add_layer_to_project(warstwa_wynik)
            return warstwa_wynik
        else:
            print("⚠️ Nawet z buforem: 0 obiektów")
            
    except Exception as e:
        print(f"❌ Błąd testu z buforem: {e}")
    
    return None


# PRZYCINANIE Classification_2
#if not os.path.exists(f"{project_directory}/Classification_2_przyciete.csv"):  
print("🎯 ROZPOCZYNAM PRZYCINANIE WARSTWY...")

nazwa_punktow = 'Classification_2'
nazwa_maski = 'granica_terenu'

warstwy_punktowe = QgsProject.instance().mapLayersByName(nazwa_punktow)
warstwy_maski = QgsProject.instance().mapLayersByName(nazwa_maski)

if not warstwy_punktowe:
    print(f"❌ Nie znaleziono warstwy: {nazwa_punktow}")
else:
    warstwa_punktowa = warstwy_punktowe[0]
    print(f"✅ Znaleziono warstwę punktową: {nazwa_punktow} ({warstwa_punktowa.featureCount()} obiektów)")

if not warstwy_maski:
    print(f"❌ Nie znaleziono warstwy: {nazwa_maski}")
else:
    warstwa_wektorowa = warstwy_maski[0]
    print(f"✅ Znaleziono warstwę maski: {nazwa_maski} ({warstwa_wektorowa.featureCount()} obiektów)")

if warstwy_punktowe and warstwy_maski:
    # Użyj zaawansowanej funkcji przycinania
    warstwa_przycieta = clip_layer_advanced(warstwa_punktowa, warstwa_wektorowa)
    
    # Jeśli nadal nie ma wyników, spróbuj z buforem
    if warstwa_przycieta is None or warstwa_przycieta.featureCount() == 0:
        print("\n🔄 Próbuję z buforem...")
        warstwa_przycieta = simple_buffer_test(warstwa_punktowa, warstwa_wektorowa)
    
    if warstwa_przycieta and warstwa_przycieta.featureCount() > 0:
        print(f"\n🎉 KOŃCOWY SUKCES: {warstwa_przycieta.featureCount()} obiektów!")
    else:
        print("\n❌ OSTATECZNA PORAŻKA: Brak przecięć między warstwami")
        print("Sprawdź czy:")
        print("  1. Warstwy mają ten sam układ współrzędnych")
        print("  2. Zakresy się nakładają") 
        print("  3. Geometrie są poprawne")
        print("  4. Warstwa maski zawiera rzeczywiście obszary, a nie tylko punkty/linie")
else:
    print("❌ Nie można kontynuować - brak jednej z warstw")

# === GŁÓWNY KOD WYKONAWCZY ===
print("Rozpoczynam predykcję z modelem heksagonalnym...")

# Sprawdź czy warstwa istnieje
warstwa_nazwa = "Classification_2_przyciete"
warstwy = QgsProject.instance().mapLayersByName(warstwa_nazwa)

if not warstwy:
    print(f"Nie znaleziono warstwy: {warstwa_nazwa}")
else:
    print(f"Znaleziono warstwę: {warstwa_nazwa}")
    warstwa = warstwy[0]
    layer_df = layer_to_df(warstwa)
    
    print("DIAGNOSTYKA DANYCH:")
    print(f"Kształt: {layer_df.shape}")
    print(f"Kolumny: {list(layer_df.columns)}")
    
    # Sprawdź czy są wymagane kolumny
    required_cols = ['X', 'Y'] + INPUT_FEATURES
    missing_cols = [col for col in required_cols if col not in layer_df.columns]
    
    if missing_cols:
        print(f"Brakuje kolumn: {missing_cols}")
        exit(1)
    
    # Zapisz do CSV (opcjonalnie)
    layer_df.to_csv(INPUT_CSV, index=False)
    
    # Stwórz predyktor i wczytaj model
    predictor = QGISHexLidarPredictor()
    
    if not predictor.load_model(MODEL_PATH, SCALER_PATH):
        print("Nie można wczytać modelu!")
        exit(1)
    
    # Wykonaj predykcję
    print("Rozpoczynam predykcję...")
    predictions = predictor.predict_points(layer_df)
    
    # Statystyki predykcji
    unique, counts = np.unique(predictions, return_counts=True)
    print("\nRozkład predykcji:")
    for class_id, count in zip(unique, counts):
        percentage = (count / len(predictions)) * 100
        print(f"   Klasa {class_id}: {count:,} punktów ({percentage:.1f}%)")
    
    # Konwersja klas (jeśli potrzeba)
    predictions_converted = predictions.copy()
    for i in range(len(predictions_converted)):
        if predictions_converted[i] in [2, 3]:
            predictions_converted[i] = 0
        elif predictions_converted[i] == 4:
            predictions_converted[i] = 1
    
    # Zapisz wyniki
    layer_df['predicted_label'] = predictions_converted
    layer_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Zapisano wyniki do: {OUTPUT_CSV}")
    
    # # Wczytaj jako warstwę do QGIS
    # wczytaj_csv_do_qgis(OUTPUT_CSV, nazwa_kolumny_x='X', nazwa_kolumny_y='Y', 
    #                     separator=',', crs_kod='EPSG:2180', 
    #                     nazwa_warstwy='punkty_pbc_wyniki_predykcji_teren_inwestycji',
    #                     kolumny_int=['predicted_label'])
    # Sprawdź czy warstwa z wynikami już istnieje
    existing_result_layers = QgsProject.instance().mapLayersByName('punkty_pbc_wyniki_predykcji_teren_inwestycji')
    if existing_result_layers:
        print("Warstwa wyników już istnieje - usuwam starą")
        for layer in existing_result_layers:
            QgsProject.instance().removeMapLayer(layer.id())
    
    # Wczytaj jako warstwę do QGIS
    result_layer = wczytaj_csv_do_qgis(OUTPUT_CSV, nazwa_kolumny_x='X', nazwa_kolumny_y='Y', 
                        separator=',', crs_kod='EPSG:2180', 
                        nazwa_warstwy='punkty_pbc_wyniki_predykcji_teren_inwestycji',
                        kolumny_int=['predicted_label'])
    
    if result_layer:
        # Aplikuj styl tylko jeśli warstwa została utworzona
        apply_qml_style_to_layer(
            layer='punkty_pbc_wyniki_predykcji_teren_inwestycji',
            qml_file_path='/home/adrian/Documents/JXPROJEKT/style/punkty_PBC_new.qml', 
            show_messages=True
        )
        print("Warstwa z wynikami została utworzona i ostylowana")
    else:
        print("Błąd tworzenia warstwy z wynikami")
        # Usuń warstwy tymczasowe
        safe_remove_memory_layers()
    
    # # Aplikuj styl
    # apply_qml_style_to_layer(
    #     layer='punkty_pbc_wyniki_predykcji_teren_inwestycji',
    #     qml_file_path='/home/adrian/Documents/JXPROJEKT/style/punkty_PBC_new.qml', 
    #     show_messages=True
    # )

print("Predykcja zakończona!")