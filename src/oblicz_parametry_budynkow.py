#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 07:15:39 2025

@author: adrian
"""
import processing
from qgis.core import (QgsProject, QgsVectorLayer, QgsFeature,QgsWkbTypes,
    QgsFields, QgsField)
from pathlib import Path
from PyQt5.QtCore import QVariant
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import StandardScaler
from qgis.utils import iface
import os
from PyQt5.QtWidgets import QMessageBox, QFileDialog


def apply_qml_style_to_layer(layer, qml_file_path=None, show_messages=True):
    """
    Aplikuje styl QML do warstwy wektorowej.
    
    Args:
        layer: Obiekt QgsVectorLayer lub nazwa warstwy (str)
        qml_file_path: Ścieżka do pliku QML (str). Jeśli None, otworzy dialog wyboru pliku
        show_messages: Czy pokazywać komunikaty o błędach/sukcesie (bool)
    
    Returns:
        bool: True jeśli stylizacja została zastosowana pomyślnie, False w przeciwnym razie
    """
    
    # Konwersja nazwy warstwy na obiekt warstwy jeśli potrzeba
    if isinstance(layer, str):
        layer_name = layer
        layer = None
        for lyr in QgsProject.instance().mapLayers().values():
            if lyr.name() == layer_name:
                layer = lyr
                break
        
        if layer is None:
            if show_messages:
                QMessageBox.warning(None, "Błąd", f"Nie znaleziono warstwy: {layer_name}")
            return False
    
    # Sprawdzenie czy warstwa jest wektorowa
    if not isinstance(layer, QgsVectorLayer):
        if show_messages:
            QMessageBox.warning(None, "Błąd", "Wybrana warstwa nie jest warstwą wektorową")
        return False
    
    # Wybór pliku QML jeśli nie został podany
    if qml_file_path is None:
        qml_file_path, _ = QFileDialog.getOpenFileName(
            None,
            "Wybierz plik stylu QML",
            "",
            "Pliki QML (*.qml);;Wszystkie pliki (*)"
        )
        
        if not qml_file_path:
            return False
    
    # Sprawdzenie czy plik istnieje
    if not os.path.exists(qml_file_path):
        if show_messages:
            QMessageBox.warning(None, "Błąd", f"Plik QML nie istnieje: {qml_file_path}")
        return False
    
    # Aplikacja stylu
    try:
        result = layer.loadNamedStyle(qml_file_path)
        
        if result[1]:  # result[1] zawiera informację o powodzeniu operacji
            # Odświeżenie warstwy
            layer.triggerRepaint()
            iface.layerTreeView().refreshLayerSymbology(layer.id())
            
            if show_messages:
                QMessageBox.information(None, "Sukces", 
                    f"Styl został pomyślnie zastosowany do warstwy: {layer.name()}")
            return True
        else:
            if show_messages:
                QMessageBox.warning(None, "Błąd", 
                    f"Nie udało się załadować stylu: {result[0]}")
            return False
            
    except Exception as e:
        if show_messages:
            QMessageBox.critical(None, "Błąd", f"Wystąpił błąd podczas ładowania stylu: {str(e)}")
        return False


def zapis_do_gpkg(layer_name):
    def fid_kolizja(warstwa):
        for field in warstwa.fields():
            if field.name().lower() == "fid" and field.typeName().lower() != "integer":
                return True
        return False

    def utworz_kopie_bez_fid(warstwa, nowa_nazwa):
        geometria = QgsWkbTypes.displayString(warstwa.wkbType())
        crs = warstwa.crs().authid()
        kopia = QgsVectorLayer(f"{geometria}?crs={crs}", nowa_nazwa, "memory")

        fields = QgsFields()
        for field in warstwa.fields():
            if field.name().lower() != "fid":
                fields.append(field)
        kopia.dataProvider().addAttributes(fields)
        kopia.updateFields()

        for feat in warstwa.getFeatures():
            nowy = QgsFeature(fields)
            attrs = [feat[field.name()] for field in fields]
            nowy.setAttributes(attrs)
            nowy.setGeometry(feat.geometry())
            kopia.dataProvider().addFeature(nowy)

        kopia.updateExtents()
        QgsProject.instance().addMapLayer(kopia)
        return kopia

    # Ścieżka do projektu
    project_path = QgsProject.instance().fileName()
    if not project_path:
        print("❌ Projekt niezapisany.")
        return
    
    project_directory = Path(project_path).parent
    output_folder = Path(project_directory)
    # Tworzenie katalogu jeśli nie istnieje
    if not output_folder.exists():
        output_folder.mkdir(parents=True, exist_ok=True)
        print(f"📂 Utworzono katalog: {output_folder}")
    else:
        print(f"📁 Katalog już istnieje: {output_folder}")
        
    output_path = f"{output_folder}/{layer_name}.gpkg"

    # Pobierz warstwę
    warstwy = QgsProject.instance().mapLayersByName(layer_name)
    if not warstwy:
        print(f"❌ Nie znaleziono warstwy: {layer_name}")
        return
    warstwa = warstwy[0]

    # Obsługa konfliktu z 'fid'
    if fid_kolizja(warstwa):
        print("⚠️ Wykryto kolizję z polem 'fid'. Tworzę kopię bez tego pola.")
        warstwa = utworz_kopie_bez_fid(warstwa, f"{layer_name}_safe")

    # Zapis przy użyciu processing
    processing.run("native:savefeatures", {
    'INPUT': warstwa,
    'OUTPUT': output_path})

    print(f"✅ Warstwa zapisana do: {output_path}")

    # Wczytaj z powrotem
    vlayer = QgsVectorLayer(f"{output_path}|layername={layer_name}", layer_name, "ogr")
    if vlayer.isValid():
        QgsProject.instance().addMapLayer(vlayer)
        print("✅ Warstwa wczytana ponownie do projektu.")
    else:
        print("❌ Nie udało się wczytać zapisanej warstwy.")


def remove_memory_layers():
    for lyr in QgsProject.instance().mapLayers().values():
        if lyr.dataProvider().name() == 'memory':
            QgsProject.instance().removeMapLayer(lyr.id())


def analyze_roof_slope_from_point_cloud(points_df, building_id):
    """
    Analizuje nachylenie dachu bezpośrednio z chmury punktów
    
    Args:
        points_df: DataFrame z kolumnami ['X', 'Y', 'Z']
        building_id: ID budynku
    
    Returns:
        dict: wyniki analizy nachylenia
    """
    if len(points_df) < 20:
        return {"slope": 0, "confidence": 0, "method": "insufficient_data"}
    
    points = points_df[['X', 'Y', 'Z']].values
    
    # === KROK 1: Czyszczenie danych ===
    cleaned_points = clean_point_cloud(points)
    
    if len(cleaned_points) < 15:
        return {"slope": 0, "confidence": 0, "method": "insufficient_clean_data"}
    
    # === KROK 2: Segmentacja płaszczyzn dachu ===
    roof_planes = segment_roof_planes(cleaned_points)
    
    if not roof_planes:
        return {"slope": 0, "confidence": 0, "method": "no_planes_found"}
    
    # === KROK 3: Oblicz nachylenie dla każdej płaszczyzny ===
    plane_slopes = []
    plane_confidences = []
    
    for plane_points in roof_planes:
        if len(plane_points) < 10:
            continue
            
        slope_info = calculate_plane_slope_robust(plane_points)
        if slope_info and slope_info['slope'] > 0:
            plane_slopes.append(slope_info['slope'])
            plane_confidences.append(slope_info['confidence'])
    
    if  len(plane_slopes) == 0:
        # Fallback: cały dach jako jedna płaszczyzna
        fallback_slope = calculate_overall_slope(cleaned_points)
        return { 
            "slope": fallback_slope,
            "confidence": 0.5,
            "method": "fallback_overall_accepted",
            "num_points": len(cleaned_points)}
    
    # === KROK 4: Wybierz najlepsze nachylenie ===
    # Średnia ważona nachyleń (waga = pewność)
    if len(plane_slopes) == 1:
        plane_slopes.sort(reverse=True)
        final_slope = plane_slopes[0]
        confidence = plane_confidences[0]
        # try:
        #     final_slope = (plane_slopes[0] + plane_slopes[1])/2
        #     confidence = plane_confidences[0]
        # except:
        #         final_slope = plane_slopes[0]
        #         confidence = plane_confidences[0]
    else:
        weights = np.array(plane_confidences)
        final_slope = np.average(plane_slopes, weights=weights)
        confidence = np.mean(plane_confidences)
    
    return {
        "slope": final_slope,
        "confidence": confidence,
        "method": f"multi_plane_{len(plane_slopes)}",
        "num_points": len(cleaned_points)
    }

def clean_point_cloud(points):
    """
    Czyści chmurę punktów z outlierów
    """
    # Usuń punkty odstające na podstawie Z
    z_values = points[:, 2]
    z_median = np.median(z_values)
    z_mad = np.median(np.abs(z_values - z_median))  # Median Absolute Deviation
    
    # Usuń punkty daleko od mediany (kominy, anteny, błędy)
    threshold = z_median + 3 * z_mad
    mask_z = z_values <= threshold
    
    # Usuń punkty bardzo niskie (może być grunt/błędy)
    z_10th_percentile = np.percentile(z_values, 10)
    mask_z_low = z_values >= z_10th_percentile
    
    # Połącz maski
    final_mask = mask_z & mask_z_low
    
    return points[final_mask]

def segment_roof_planes(points):
    """
    Segmentuje punkty dachu na płaszczyzny używając DBSCAN
    """
    if len(points) < 15:
        return [points]
    
    # === Metoda 1: Clustering przestrzenny z normalną ===
    
    # Oblicz lokalne normalne
    normals = calculate_local_normals(points)
    
    # Normalizuj współrzędne dla DBSCAN
    scaler = StandardScaler()
    points_scaled = scaler.fit_transform(points)
    
    # Połącz pozycje i normalne (normalne mają większą wagę)
    features = np.hstack([points_scaled, normals * 2])
    
    # DBSCAN clustering
    clustering = DBSCAN(eps=0.3, min_samples=8).fit(features)
    
    # Pogrupuj punkty według klastrów
    segments = []
    unique_labels = set(clustering.labels_)
    
    for label in unique_labels:
        if label != -1:  # Ignoruj szum
            mask = clustering.labels_ == label
            segment_points = points[mask]
            if len(segment_points) >= 10:
                segments.append(segment_points)
    print(f"🔍 segment roof: znaleziono {len(segments)} segmentów")
    for i, segment in enumerate(segments):
        print(f" Segment {i}: {len(segment)} punktów")

    # Jeśli nie znaleziono segmentów, zwróć cały dach
    if not segments:
        segments = [points]
    
    return segments

def calculate_local_normals(points, k=8):
    """
    Oblicza normalne lokalne używając PCA na sąsiadach
    """
    from sklearn.neighbors import NearestNeighbors
    
    # Znajdź k najbliższych sąsiadów
    nbrs = NearestNeighbors(n_neighbors=min(k+1, len(points))).fit(points)
    distances, indices = nbrs.kneighbors(points)
    
    normals = []
    for i, neighbors_idx in enumerate(indices):
        # Pobierz punkty sąsiadów
        neighbors = points[neighbors_idx]
        
        # Wycentruj punkty
        centered = neighbors - np.mean(neighbors, axis=0)
        
        # PCA - wektor własny z najmniejszą wartością własną to normalna
        try:
            _, _, V = np.linalg.svd(centered)
            normal = V[-1]  # ostatni wektor własny
            
            # Upewnij się, że normalna wskazuje "w górę"
            if normal[2] < 0:
                normal = -normal
                
            normals.append(normal)
        except:
            normals.append([0, 0, 1])  # fallback
    
    return np.array(normals)

def calculate_plane_slope_robust(points):
    """
    Oblicza nachylenie płaszczyzny używając RANSAC (odporne na outliers)
    """
    if len(points) < 10:
        return None
    
    X = points[:, :2]  # x, y
    y = points[:, 2]   # z
    print("🚀 uruchamiam RANSAC dla", len(points), "punktów")
    print("X sample:", X[:5])
    print("Z sample:", y[:5])

    try:
        # RANSAC dla odporności na outliers
        min_pts = min(10, len(points) // 2)
        ransac = RANSACRegressor(
            min_samples=min_pts,
            residual_threshold=0.1,
            max_trials=100,
            random_state=42
        )

        print("➡️ X shape:", X.shape)
        print("➡️ y shape:", y.shape)
        print("➡️ Min/Max Z:", np.min(y), np.max(y))
        print("➡️ Próbuję RANSAC...")

        ransac.fit(X, y)
        
        # Współczynniki płaszczyzny: z = ax + by + c
        a, b = ransac.estimator_.coef_
        c = ransac.estimator_.intercept_
        
        # Wektor normalny do płaszczyzny
        normal = np.array([a, b, -1])
        normal = normal / np.linalg.norm(normal)
        
        # Kąt nachylenia (kąt między normalną a [0,0,1])
        cos_angle = abs(normal[2])
        if cos_angle > 0.999:  # Praktycznie płaska
            slope_degrees = 0
        else:
            angle_rad = np.arccos(cos_angle)
            slope_degrees = np.degrees(angle_rad)
        
        # Pewność na podstawie RANSAC
        inlier_ratio = np.sum(ransac.inlier_mask_) / len(points)
        confidence = inlier_ratio
        if not hasattr(ransac, 'inlier_mask_') or ransac.inlier_mask_ is None:
            print("❌ RANSAC: brak inlier_mask_")
            return None


        return {
            "slope": slope_degrees,
            "confidence": confidence,
            "inlier_ratio": inlier_ratio,
            "coefficients": [a, b, c]
        }
        
    except Exception as e:
        import traceback
        print(f"❌ Wyjątek w RANSAC:{e}")
        traceback.print_exc()
        return calculate_plane_slope_simple(points)

    
def calculate_plane_slope_simple(points):
    """
    Prosta metoda dopasowania płaszczyzny (fallback)
    """
    print("🔄 Fallback: uruchamiam plane_slope_simple")

    if len(points) < 3:
        return None
    
    X = points[:, :2]
    y = points[:, 2]
    
    # Dodaj kolumnę jedynek
    X_with_intercept = np.column_stack([X, np.ones(len(X))])
    
    try:
        # Metoda najmniejszych kwadratów
        coeffs, residuals, rank, s = np.linalg.lstsq(X_with_intercept, y, rcond=None)
        a, b, c = coeffs
        
        # Oblicz nachylenie
        normal = np.array([a, b, -1])
        normal = normal / np.linalg.norm(normal)
        
        cos_angle = abs(normal[2])
        if cos_angle > 0.999:
            slope_degrees = 0
        else:
            angle_rad = np.arccos(cos_angle)
            slope_degrees = np.degrees(angle_rad)
        
        # Prosta ocena pewności na podstawie residuals
        if len(residuals) > 0:
            mse = residuals[0] / len(points)
            confidence = max(0, 1 - mse)  # Im mniejszy błąd, tym większa pewność
        else:
            confidence = 0.5
    
        
        return {
            "slope": slope_degrees,
            "confidence": confidence,
            "method": "simple_lstsq"
        }    
        
    except Exception as e:
        print(f"Błąd w prostej metodzie: {e}")
        return None
    
    
    
def calculate_overall_slope(points):
    """
    Oblicza ogólne nachylenie dachu (fallback method)
    """
    if len(points) < 10:
        return 0
    
    try:
        # Znajdź główną oś budynku używając PCA
        points_2d = points[:, :2]
        points_centered = points_2d - np.mean(points_2d, axis=0)
        
        # Wyznacz główną oś kierunkową za pomocą PCA
        _, _, V = np.linalg.svd(points_centered)
        main_axis = V[0]  # główna oś
        
        projections = np.dot(points_centered, main_axis)
        horizontal_dist = np.max(projections) - np.min(projections)
     
        if horizontal_dist == 0:
            print("⚠️ Brak rozrzutu w kierunku głównej osi — nachylenie = 0")
            return 0
    
        median_proj = np.median(projections)
        mask_half1 = projections <= median_proj
        mask_half2 = projections > median_proj
    
        # Oblicz średnie Z w każdej połowie
        z_half1 = np.mean(points[mask_half1, 2]) if np.any(mask_half1) else None
        z_half2 = np.mean(points[mask_half2, 2]) if np.any(mask_half2) else None
    
        # Różnica wysokości — fallback do średniej globalnej
        if z_half1 is not None and z_half2 is not None:
            height_diff = abs(z_half1 - z_half2)
        elif z_half1 is not None:
            height_diff = abs(z_half1 - np.mean(points[:, 2]))
        elif z_half2 is not None:
            height_diff = abs(z_half2 - np.mean(points[:, 2]))
        else:
            print("⚠️ Brak punktów w obu połówkach — nachylenie = 0")
            return 0
    
        slope_ratio = height_diff / horizontal_dist
        slope_degrees = np.degrees(np.arctan(slope_ratio))
        return slope_degrees
    
    except Exception as e:
        print("❌ Błąd w calculate_overall_slope:", e)
        return 0


def process_buildings_roof_slopes(points_layer):
    """
    Główna funkcja przetwarzania nachyleń dachów
    """    
    if not points_layer:
        print("❌ Brak aktywnej warstwy!")
        return
    
    # Pobierz warstwę poligonów budynków
    buildings_layer = QgsProject.instance().mapLayersByName("budynki_z_szer_elew_front")[0]
    print(f"✅ Używam warstwy budynków: {buildings_layer.name()}")
    
    # Pobierz punkty do DataFrame
    print("📊 Pobieram punkty...")
    points_data = []
    for feature in points_layer.getFeatures():
        geom = feature.geometry()
        if geom.type() == 0:  # Point
            # point = geom.asPoint()
            # attrs = feature.attributes()
            
            # Zakładam kolumny: można dostosować do rzeczywistych nazw
            row = {
                'X': feature.attribute('X'),
                'Y': feature.attribute('Y'),
                'Z': feature.attribute('Z'),
                'ID_BUDYNKU': feature.attribute('ID_BUDYNKU')
            }
            points_data.append(row)
    
    points_df = pd.DataFrame(points_data)
    print(f"✅ Pobrano {len(points_df)} punktów")
    print("Zakres wysokości Z:", points_df['Z'].min(), "–", points_df['Z'].max())
    print("Unikalne wysokości Z:", points_df['Z'].nunique())

    # Dodaj pole nachylenia do warstwy budynków
    if 'nachylenie_chmura' not in [field.name() for field in buildings_layer.fields()]:
        buildings_layer.startEditing()
        buildings_layer.dataProvider().addAttributes([
            QgsField('nachylenie_chmura', QVariant.Double),
            QgsField('pewnosc_nachylenia', QVariant.Double),
            QgsField('metoda_nachylenia', QVariant.String),
            QgsField('liczba_punktow', QVariant.Int)
        ])
        buildings_layer.updateFields()
    
    buildings_layer.startEditing()
    
    # Przetwarzaj każdy budynek
    processed = 0
    for building_feature in buildings_layer.getFeatures():
        building_id = building_feature.attribute('ID_BUDYNKU')  # Dostosuj nazwę kolumny
        
        if building_id is None:
            continue
        
        # Pobierz punkty dla tego budynku
        building_points = points_df[points_df['ID_BUDYNKU'].astype(str) == str(building_id)]
        print(f"➡️ Budynek {building_id} ma {len(building_points)} punktów")

        
        if len(building_points) < 10:
            # Zbyt mało punktów
            slope_info = {"slope": 0, "confidence": 0, "method": "insufficient_points", "num_points": len(building_points)}
        else:
            # Analizuj nachylenie
            slope_info = analyze_roof_slope_from_point_cloud(building_points, building_id)
        
        # Zaokrąglij nachylenie
        slope = slope_info.get('slope', 0)
        if slope is None:
            slope = 0
        if slope > 2.5:
            slope = round(slope / 5) * 5  # Zaokrąglij do 5°
        else:
            slope = round(slope, 1)
            continue

        # Zapisz wyniki
        buildings_layer.changeAttributeValue(
            building_feature.id(),
            buildings_layer.fields().indexOf('nachylenie_chmura'),
            slope
        )
        buildings_layer.changeAttributeValue(
            building_feature.id(),
            buildings_layer.fields().indexOf('pewnosc_nachylenia'),
            round(slope_info.get('confidence', 0), 2)
        )
        buildings_layer.changeAttributeValue(
            building_feature.id(),
            buildings_layer.fields().indexOf('metoda_nachylenia'),
            slope_info.get('method', 'unknown')
        )
        buildings_layer.changeAttributeValue(
            building_feature.id(),
            buildings_layer.fields().indexOf('liczba_punktow'),
            slope_info.get('num_points', 0)
        )
        
        processed += 1
        if processed % 10 == 0:
            print(f"⏳ Przetworzono {processed} budynków...")
    
    buildings_layer.commitChanges()
    print(f"✅ Analiza zakończona! Przetworzono {processed} budynków")
    print("🔢 ID w budynkach:", [f['ID_BUDYNKU'] for f in buildings_layer.getFeatures()])
    print("🔢 Unikalne ID w punktach:", points_df['ID_BUDYNKU'].unique().tolist()[:10])


def filter_outliers_iqr(data, multiplier=1.5):
    """
    Filtruje outliery używając metody IQR (Interquartile Range)
    """
    if len(data) < 4:
        return data
    
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    filtered_data = [x for x in data if lower_bound <= x <= upper_bound]
    
    if len(filtered_data) < max(3, len(data) * 0.3):
        return data
    
    return filtered_data

def calculate_roof_height(roof_points):
    """
    Oblicza reprezentatywną wysokość dachu z filtrowaniem outlierów
    """
    if not roof_points or len(roof_points) == 0:
        return 0
    
    filtered_points = filter_outliers_iqr(roof_points, multiplier=1.5)
    
    if len(filtered_points) > 20:
        sorted_points = sorted(filtered_points, reverse=True)
        percentile_70 = int(len(sorted_points) * 0.05)
        percentile_95 = int(len(sorted_points) * 0.30)
        selected_points = sorted_points[percentile_70:percentile_95]
        
        if len(selected_points) > 0:
            return np.mean(selected_points)
    
    return np.mean(filtered_points)

def calculate_ground_height(ground_points):
    """
    Oblicza reprezentatywną wysokość gruntu z filtrowaniem outlierów
    """
    if not ground_points or len(ground_points) == 0:
        return 0
    
    filtered_points = filter_outliers_iqr(ground_points, multiplier=1.5)
    return np.mean(filtered_points)

def add_height_to_buildings_layer():
    """
    Główna funkcja dodająca kolumnę wysokość do warstwy budynki_z_szer_elew_front
    """
    try:
        # === 1. Pobierz warstwę docelową ===
        target_layer = QgsProject.instance().mapLayersByName('budynki_z_szer_elew_front')[0]
        
        # Sprawdź czy kolumna 'wysokosc' już istnieje
        field_names = [field.name() for field in target_layer.fields()]
        
        if 'wysokosc' not in field_names:
            # Dodaj nowe pole do warstwy
            target_layer.dataProvider().addAttributes([QgsField('wysokosc', QVariant.Double)])
            target_layer.updateFields()
            print("Dodano kolumnę 'wysokosc' do warstwy")
        else:
            print("Kolumna 'wysokosc' już istnieje - będzie aktualizowana")
        
        # === 2. DEBUG: Sprawdź dostępne warstwy ===
        available_layers = [layer.name() for layer in QgsProject.instance().mapLayers().values()]
        print(f"Dostępne warstwy: {available_layers}")
        
        # === 3. Pobierz dane z warstwy gruntu ===
        try:
            ground_layer = QgsProject.instance().mapLayersByName('Classification_2_bufor_with_IDs')[0]
            print(f"✅ Znaleziono warstwę gruntu: {ground_layer.name()}")
            print(f"Liczba obiektów w warstwie gruntu: {ground_layer.featureCount()}")
        except IndexError:
            print("❌ BŁĄD: Nie znaleziono warstwy 'Classification_2_bufor_with_IDs'")
            return False
        
        # DEBUG: Sprawdź strukturę pól warstwy gruntu
        ground_field_names = [field.name() for field in ground_layer.fields()]
        print(f"Pola warstwy gruntu: {ground_field_names}")
        
        ground_data = []
        for feature in ground_layer.getFeatures():
            attrs = feature.attributes()
            fields = {field.name(): i for i, field in enumerate(ground_layer.fields())}
            
            # DEBUG: Sprawdź czy pola istnieją
            if 'Z' not in fields:
                print("❌ BŁĄD: Brak pola 'Z' w warstwie gruntu")
                return False
            if 'ID_DZIALKI' not in fields:
                print("❌ BŁĄD: Brak pola 'ID_DZIALKI' w warstwie gruntu")
                return False
            
            z_value = attrs[fields['Z']] if attrs[fields['Z']] is not None else 0
            id_dzialki = attrs[fields['ID_DZIALKI']] if attrs[fields['ID_DZIALKI']] is not None else ''
            
            ground_data.append({
                'Z': z_value,
                'ID_DZIALKI': id_dzialki
            })
        
        ground_df = pd.DataFrame(ground_data)
        print(f"✅ Pobrano {len(ground_df)} punktów gruntu")
        print(f"Zakres wysokości gruntu: {ground_df['Z'].min():.2f} - {ground_df['Z'].max():.2f}")
        print(f"Unikalne ID_DZIALKI w gruncie: {ground_df['ID_DZIALKI'].nunique()}")
        
        # === 4. Pobierz dane z warstwy dachów ===
        try:
            roof_layer = QgsProject.instance().mapLayersByName('Classification_6_with_IDs')[0]
            print(f"✅ Znaleziono warstwę dachów: {roof_layer.name()}")
            print(f"Liczba obiektów w warstwie dachów: {roof_layer.featureCount()}")
        except IndexError:
            print("❌ BŁĄD: Nie znaleziono warstwy 'Classification_6_with_IDs'")
            return False
        
        # DEBUG: Sprawdź strukturę pól warstwy dachów
        roof_field_names = [field.name() for field in roof_layer.fields()]
        print(f"Pola warstwy dachów: {roof_field_names}")
        
        roof_data = []
        for feature in roof_layer.getFeatures():
            attrs = feature.attributes()
            fields = {field.name(): i for i, field in enumerate(roof_layer.fields())}
            
            # DEBUG: Sprawdź czy pola istnieją
            if 'Z' not in fields:
                print("❌ BŁĄD: Brak pola 'Z' w warstwie dachów")
                return False
            if 'ID_DZIALKI' not in fields:
                print("❌ BŁĄD: Brak pola 'ID_DZIALKI' w warstwie dachów")
                return False
            if 'ID_BUDYNKU' not in fields:
                print("❌ BŁĄD: Brak pola 'ID_BUDYNKU' w warstwie dachów")
                return False
            
            z_value = attrs[fields['Z']] if attrs[fields['Z']] is not None else 0
            id_dzialki = attrs[fields['ID_DZIALKI']] if attrs[fields['ID_DZIALKI']] is not None else ''
            id_budynku = attrs[fields['ID_BUDYNKU']] if attrs[fields['ID_BUDYNKU']] is not None else ''
            
            roof_data.append({
                'Z': z_value,
                'ID_DZIALKI': id_dzialki,
                'ID_BUDYNKU': id_budynku
            })
        
        roof_df = pd.DataFrame(roof_data)
        print(f"✅ Pobrano {len(roof_df)} punktów dachów")
        print(f"Zakres wysokości dachów: {roof_df['Z'].min():.2f} - {roof_df['Z'].max():.2f}")
        print(f"Unikalne ID_BUDYNKU w dachach: {roof_df['ID_BUDYNKU'].nunique()}")
        print(f"Unikalne ID_DZIALKI w dachach: {roof_df['ID_DZIALKI'].nunique()}")
        
        # === 5. Oblicz wysokości ===
        # Średnie wysokości gruntu dla każdej działki
        print("🔄 Obliczam wysokości gruntu...")
        ground_heights = ground_df.groupby('ID_DZIALKI')['Z'].apply(list).reset_index()
        ground_heights['Z_ground_mean'] = ground_heights['Z'].apply(calculate_ground_height)
        ground_heights = ground_heights[['ID_DZIALKI', 'Z_ground_mean']]
        
        print(f"Obliczono wysokości gruntu dla {len(ground_heights)} działek")
        print(f"Przykładowe wysokości gruntu:")
        print(ground_heights.head())
        
        # Wysokości dachów dla każdego budynku
        print("🔄 Obliczam wysokości dachów...")
        roof_heights = roof_df.groupby('ID_BUDYNKU').agg({
            'Z': list,
            'ID_DZIALKI': 'first'
        }).reset_index()
        
        roof_heights['Z_roof_mean'] = roof_heights['Z'].apply(calculate_roof_height)
        
        print(f"Obliczono wysokości dachów dla {len(roof_heights)} budynków")
        print(f"Przykładowe wysokości dachów:")
        print(roof_heights[['ID_BUDYNKU', 'Z_roof_mean', 'ID_DZIALKI']].head())
        
        # Połącz dane i oblicz wysokości
        print("🔄 Łączę dane i obliczam wysokości budynków...")
        buildings_heights = pd.merge(roof_heights, ground_heights, on='ID_DZIALKI', how='left')
        
        # DEBUG: Sprawdź wyniki połączenia
        print(f"Po połączeniu: {len(buildings_heights)} budynków")
        print("Przykładowe dane po połączeniu:")
        print(buildings_heights[['ID_BUDYNKU', 'Z_roof_mean', 'Z_ground_mean']].head())
        
        # Sprawdź czy są brakujące dane gruntu
        missing_ground = buildings_heights['Z_ground_mean'].isna().sum()
        if missing_ground > 0:
            print(f"⚠️ UWAGA: {missing_ground} budynków nie ma danych o gruncie")
            # Zastąp brakujące dane medianą
            ground_median = ground_df['Z'].median()
            buildings_heights['Z_ground_mean'].fillna(ground_median, inplace=True)
            print(f"Użyto mediany gruntu: {ground_median:.2f}")
        
        buildings_heights['wysokosc'] = round(
            buildings_heights['Z_roof_mean'] - buildings_heights['Z_ground_mean'], 2
        )
        
        # DEBUG: Sprawdź obliczone wysokości
        print(f"Obliczone wysokości:")
        print(f"  Średnia: {buildings_heights['wysokosc'].mean():.2f} m")
        print(f"  Min: {buildings_heights['wysokosc'].min():.2f} m")
        print(f"  Max: {buildings_heights['wysokosc'].max():.2f} m")
        print(f"  Liczba zer: {(buildings_heights['wysokosc'] == 0).sum()}")
        
        # Stwórz słownik ID_BUDYNKU -> wysokosc
        height_dict = dict(zip(buildings_heights['ID_BUDYNKU'], buildings_heights['wysokosc']))
        print(f"Utworzono słownik wysokości dla {len(height_dict)} budynków")
        
        # === 6. Aktualizuj warstwę docelową ===
        target_layer.startEditing()
        
        # Pobierz indeks kolumny wysokosc
        height_field_idx = target_layer.fields().indexFromName('wysokosc')
        id_field_idx = target_layer.fields().indexFromName('ID_BUDYNKU')
        
        if height_field_idx == -1:
            print("❌ BŁĄD: Nie znaleziono pola 'wysokosc' w warstwie docelowej")
            return False
        if id_field_idx == -1:
            print("❌ BŁĄD: Nie znaleziono pola 'ID_BUDYNKU' w warstwie docelowej")
            return False
        
        updated_count = 0
        not_found_count = 0
        
        for feature in target_layer.getFeatures():
            building_id = feature.attributes()[id_field_idx]
            
            if building_id in height_dict:
                height_value = height_dict[building_id]
                target_layer.changeAttributeValue(feature.id(), height_field_idx, height_value)
                updated_count += 1
            else:
                not_found_count += 1
                print(f"⚠️ Nie znaleziono danych dla budynku ID: {building_id}")
        
        # Zapisz zmiany
        target_layer.commitChanges()
        
        print(f"✅ Sukces! Zaktualizowano wysokość dla {updated_count} budynków")
        print(f"⚠️ Nie znaleziono danych dla {not_found_count} budynków")
        if len(height_dict) > 0:
            print(f"📊 Średnia wysokość: {np.mean(list(height_dict.values())):.2f} m")
            print(f"📊 Zakres wysokości: {min(height_dict.values()):.2f} - {max(height_dict.values()):.2f} m")
        
        # Odśwież warstwę
        target_layer.triggerRepaint()
        iface.layerTreeView().refreshLayerSymbology(target_layer.id())
        
        return True
        
    except Exception as e:
        import traceback
        print(f"❌ Błąd podczas dodawania wysokości: {str(e)}")
        print("Pełny traceback:")
        traceback.print_exc()
        return False


def przytnij_punkty_do_poligonow(nazwa_punktow, nazwa_maski , output_name):
    # Pobierz warstwy z projektu
    warstwa_punktowa = QgsProject.instance().mapLayersByName(nazwa_punktow)[0]
    warstwa_poligonow = QgsProject.instance().mapLayersByName(nazwa_maski)[0]

    # Parametry i uruchomienie narzędzia
    parametry = {
        'INPUT': warstwa_punktowa,
        'PREDICATE': [0],  # 0 = przecina / zawiera się w (czyli punkt wewnątrz poligonu)
        'INTERSECT': warstwa_poligonow,
        'OUTPUT': f"memory:{output_name}",
    }
    
    wynik = processing.run("native:extractbylocation", parametry)
    warstwa_przycieta = wynik['OUTPUT']
    # Dodaj wynikową warstwę do projektu
    QgsProject.instance().addMapLayer(warstwa_przycieta)
    
# PRZYGOTOWANIE PUNKTOW    
#1 BUFOR 5M WOKÓL KAZDEGO BUDYNKU
# Pobierz warstwę budynków po nazwie
warstwa_budynkow = QgsProject.instance().mapLayersByName("budynki_z_szer_elew_front")[0]
# Parametry bufora
parametry = {
    'INPUT': warstwa_budynkow,
    'DISTANCE': 5,
    'SEGMENTS': 5,
    'END_CAP_STYLE': 0,        # 0 = zaokrąglone końce
    'JOIN_STYLE': 0,           # 0 = okrągłe połączenia
    'MITER_LIMIT': 2,
    'DISSOLVE': False,
    'SEPARATE_DISJOINT': False,
    'OUTPUT': 'memory:bufor_5m_budynki'
}
# Uruchom algorytm otoczki
wynik = processing.run("native:buffer", parametry)
# Dodaj wynikową warstwę do projektu
warstwa_bufora = wynik['OUTPUT']
QgsProject.instance().addMapLayer(warstwa_bufora)

#2 RÓŻNICA SYMETRYCZNA
# Pobierz warstwy z projektu
warstwa_bufora = QgsProject.instance().mapLayersByName("bufor_5m_budynki")[0]
warstwa_budynkow = QgsProject.instance().mapLayersByName("budynki_z_szer_elew_front")[0]
# Parametry algorytmu
parametry = {
    'INPUT': warstwa_bufora,
    'OVERLAY': warstwa_budynkow,
    'OUTPUT': 'memory:roznica_symetryczna_bufor_vs_budynki'
}
# Uruchomienie algorytmu
wynik = processing.run("native:symmetricaldifference", parametry)
# Dodanie wynikowej warstwy do projektu
QgsProject.instance().addMapLayer(wynik['OUTPUT'])

#3 PRZYCINA PUNKTY GRUNTU DO BUFOROW WOKÓL BUDYNKÓW
przytnij_punkty_do_poligonow('Classification_2', 
                             'roznica_symetryczna_bufor_vs_budynki' , 
                             "Classification_2_bufor")   
    
#4 DOŁĄCZA ID_DZIALKI DO PUNKTOW GRUNTU
wynik = processing.run("native:joinattributesbylocation", {
    'INPUT': QgsProject.instance().mapLayersByName("Classification_2_bufor")[0],
    'JOIN': QgsProject.instance().mapLayersByName("dzialki_ze_wskaznikami")[0],
    'PREDICATE': [0],  # 1 = zawiera (punkt zawarty w poligonie)
    'JOIN_FIELDS': ['ID_DZIALKI',],
    'METHOD': 0,       # przypisz pierwszy pasujący
    'DISCARD_NONMATCHING': False,
    'OUTPUT': 'memory:Classification_2_bufor_with_IDs'
})
QgsProject.instance().addMapLayer(wynik['OUTPUT'])
print("✅ Gotowe: utworzono warstwę Classification_2_bufor_with_IDs z przypisanymi polami.")
zapis_do_gpkg("Classification_2_bufor_with_IDs")
#5 PRZYCINA PUNKTY BUDYNKOW
przytnij_punkty_do_poligonow('Classification_6', 
                             'budynki_z_szer_elew_front' , 
                             "Classification_6_przyciete")   

# DODAJE ATRYBUTY BUDYNKOW DO WARSTWY PUNKTOWEJ
wynik = processing.run("native:joinattributesbylocation", {
    'INPUT': QgsProject.instance().mapLayersByName("Classification_6_przyciete")[0],
    'JOIN': QgsProject.instance().mapLayersByName("budynki_z_szer_elew_front")[0],
    'PREDICATE': [0],  # 1 = zawiera (punkt zawarty w poligonie)
    'JOIN_FIELDS': ['ID_BUDYNKU', 'ID_DZIALKI'],
    'METHOD': 0,       # przypisz pierwszy pasujący
    'DISCARD_NONMATCHING': False,
    'OUTPUT': 'memory:Classification_6_with_IDs'
})

QgsProject.instance().addMapLayer(wynik['OUTPUT'])
print("✅ Gotowe: utworzono warstwę Classification_6_with_IDs z przypisanymi polami.")

# OBLICZA WYSOKOŚCI I DOPISUJE DO WARSWTWY BUDYNKOW
add_height_to_buildings_layer()

# OBLICZANIE NACHYLENIA POŁACI DACHOWYCH
print("🏠 Rozpoczynam analizę nachyleń dachów z chmury punktów...")
process_buildings_roof_slopes(
    points_layer=QgsProject.instance().mapLayersByName('Classification_6_with_IDs')[0]
    )

#ZAPISUJE DO PLIKU WARSTWE PUNKTOW Z ID
layer = QgsProject.instance().mapLayersByName('Classification_6_with_IDs')[0]
if layer:
    print(layer.name())
else:
    print("Brak aktywnej warstwy")
            
layer_name = layer.name()
zapis_do_gpkg(layer_name)
remove_memory_layers()

# STYLIZACJA
apply_qml_style_to_layer("Classification_6_with_IDs", 
                         r"/home/adrian/Documents/JXPROJEKT/style/buildings_points_classificationByZ.qml")

