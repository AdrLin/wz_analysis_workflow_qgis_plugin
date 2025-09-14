#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skrypt PyQGIS - Pipeline przetwarzania chmury punktów
Wykonuje filtrowanie, konwersję do wektora i podział według klasyfikacji
"""

from qgis.core import (
    QgsApplication, QgsProcessingFeedback, QgsProject,
    QgsVectorLayer, QgsCoordinateReferenceSystem, QgsProcessingContext
)
from qgis import processing
import os

def get_active_layer():
    """Pobiera aktywną warstwę z QGIS"""
    layer = iface.activeLayer()
    if layer is None:
        raise Exception("Brak aktywnej warstwy! Wybierz warstwę point cloud w panelu warstw.")
    
    print(f"Aktywna warstwa: {layer.name()}")
    print(f"Typ warstwy: {layer.type()}")
    print(f"Źródło: {layer.source()}")
    
    return layer

def setup_processing():
    """Konfiguracja środowiska przetwarzania"""
    context = QgsProcessingContext()
    feedback = QgsProcessingFeedback()
    return context, feedback

def step1_filter_points(input_layer, context, feedback):
    """
    Krok 1: Filtrowanie punktów z Classification != 0
    """
    print("\n=== KROK 1: Filtrowanie punktów ===")
    
    # Parametry filtrowania
    filter_params = {
        'INPUT': input_layer,
        'FILTER_EXPRESSION': 'Classification != 0',
        'FILTER_EXTENT': None,
        'OUTPUT': 'TEMPORARY_OUTPUT'
    }
    
    print("Parametry filtrowania:")
    for key, value in filter_params.items():
        print(f"  {key}: {value}")
    
    # Wykonanie filtrowania
    result = processing.run("pdal:filter", filter_params, context=context, feedback=feedback)
    
    filtered_layer = result['OUTPUT']
    print(f"✓ Filtrowanie zakończone: {filtered_layer}")
    
    return filtered_layer

def step2_export_to_vector(filtered_layer, output_path, context, feedback):
    """
    Krok 2: Konwersja do wektora z wybranymi atrybutami
    """
    print("\n=== KROK 2: Konwersja do wektora ===")
    
    # Atrybuty do zachowania
    attributes = [
        'X', 'Y', 'Z', 'Classification', 'Intensity', 
        'ReturnNumber', 'NumberOfReturns', 'Red', 'Green', 'Blue'
    ]
    
    # Parametry eksportu do wektora
    export_params = {
        'INPUT': filtered_layer,
        'ATTRIBUTE': attributes,
        'FILTER_EXPRESSION': '',
        'FILTER_EXTENT': None,
        'OUTPUT': output_path
    }
    
    print("Parametry eksportu do wektora:")
    for key, value in export_params.items():
        print(f"  {key}: {value}")
    
    # Wykonanie eksportu
    result = processing.run("pdal:exportvector", export_params, context=context, feedback=feedback)
    
    vector_layer = result['OUTPUT']
    print(f"✓ Eksport do wektora zakończony: {vector_layer}")
    
    return vector_layer

def step3_reproject_and_load(vector_path, project_crs, context, feedback):
    """
    Krok 3: Eksport w układzie współrzędnych projektu i wczytanie jako pcv_CRS
    """
    print("\n=== KROK 3: Reprojekcja i wczytanie ===")
    
    # Ścieżka do reprojectowanego pliku
    project_dir = os.path.dirname(vector_path)
    reprojected_path = os.path.join(project_dir, "pcv_CRS.gpkg")
    
    # Parametry reprojekcji
    reproject_params = {
        'INPUT': vector_path,
        'TARGET_CRS': project_crs,
        'OUTPUT': reprojected_path
    }
    
    print("Parametry reprojekcji:")
    for key, value in reproject_params.items():
        print(f"  {key}: {value}")
    
    # Wykonanie reprojekcji
    result = processing.run("native:reprojectlayer", reproject_params, context=context, feedback=feedback)
    
    # Wczytanie warstwy do projektu
    pcv_layer = QgsVectorLayer(reprojected_path, "pcv_CRS", "ogr")
    if pcv_layer.isValid():
        QgsProject.instance().addMapLayer(pcv_layer)
        print(f"✓ Warstwa pcv_CRS wczytana do projektu")
    else:
        raise Exception(f"Błąd wczytywania warstwy: {reprojected_path}")
    
    return pcv_layer, reprojected_path

def step4_split_by_classification(pcv_layer, output_dir, context, feedback):
    """
    Krok 4: Podział warstwy według atrybutu Classification
    """
    print("\n=== KROK 4: Podział według klasyfikacji ===")
    
    # Parametry podziału
    split_params = {
        'INPUT': pcv_layer,
        'FIELD': 'Classification',
        'FILE_TYPE': 0,  # GeoPackage
        'OUTPUT': output_dir,
        'PREFIX_FIELD': True
    }
    
    print("Parametry podziału:")
    for key, value in split_params.items():
        print(f"  {key}: {value}")
    
    # Wykonanie podziału
    result = processing.run("native:splitvectorlayer", split_params, context=context, feedback=feedback)
    
    output_layers = result['OUTPUT_LAYERS']
    print(f"✓ Podział zakończony. Utworzono {len(output_layers)} warstw")
    
    return output_layers

def step5_load_specific_classifications(output_dir):
    """
    Krok 5: Wczytanie Classification_2 i Classification_6 do projektu
    """
    print("\n=== KROK 5: Wczytywanie Classification_2 i Classification_6 ===")
    
    classifications_to_load = [2, 6]
    loaded_layers = []
    
    for classification in classifications_to_load:
        layer_name = f"Classification_{classification}"
        layer_path = os.path.join(output_dir, f"{layer_name}.gpkg")
        
        if os.path.exists(layer_path):
            layer = QgsVectorLayer(layer_path, layer_name, "ogr")
            if layer.isValid():
                QgsProject.instance().addMapLayer(layer)
                loaded_layers.append(layer)
                print(f"✓ Wczytano: {layer_name}")
            else:
                print(f"✗ Błąd wczytywania: {layer_name}")
        else:
            print(f"✗ Nie znaleziono pliku: {layer_path}")
    
    return loaded_layers

def main():
    """
    Główna funkcja wykonująca cały pipeline
    """
    try:
        print("=== ROZPOCZĘCIE PIPELINE PRZETWARZANIA CHMURY PUNKTÓW ===")
        
        # Konfiguracja
        context, feedback = setup_processing()
        
        # Pobierz aktywną warstwę
        active_layer = get_active_layer()
        
        # Pobierz CRS projektu
        project = QgsProject.instance()
        project_crs = project.crs()
        print(f"CRS projektu: {project_crs.authid()}")
        
        # Określ ścieżki wyjściowe
        if hasattr(active_layer, 'source') and active_layer.source():
            source_path = active_layer.source()
            if '|' in source_path:
                source_path = source_path.split('|')[0]
            project_dir = os.path.dirname(source_path)
        else:
            project_dir = os.path.expanduser("~/Documents")
        
        vector_output_path = os.path.join(project_dir, "points_cloud_vector.gpkg")
        
        print(f"Katalog projektu: {project_dir}")
        print(f"Ścieżka wektora: {vector_output_path}")
        
        # KROK 1: Filtrowanie punktów
        filtered_layer = step1_filter_points(active_layer, context, feedback)
        
        # KROK 2: Konwersja do wektora
        vector_layer = step2_export_to_vector(filtered_layer, vector_output_path, context, feedback)
        
        # KROK 3: Reprojekcja i wczytanie jako pcv_CRS
        pcv_layer, pcv_path = step3_reproject_and_load(vector_output_path, project_crs, context, feedback)
        
        # KROK 4: Podział według klasyfikacji
        output_layers = step4_split_by_classification(pcv_layer, project_dir, context, feedback)
        
        # KROK 5: Wczytanie Classification_2 i Classification_6
        loaded_layers = step5_load_specific_classifications(project_dir)
        
        print("\n=== PIPELINE ZAKOŃCZONY POMYŚLNIE ===")
        print(f"Utworzono {len(output_layers)} warstw z podziału")
        print(f"Wczytano {len(loaded_layers)} warstw do projektu")
        
        # Odśwież widok
        iface.mapCanvas().refresh()
        
        return True
        
    except Exception as e:
        print(f"\n❌ BŁĄD W PIPELINE: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

# Uruchomienie skryptu
main()
