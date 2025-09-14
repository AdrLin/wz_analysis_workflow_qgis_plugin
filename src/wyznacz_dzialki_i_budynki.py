#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  9 12:35:11 2025

@author: adrian
"""

import processing
from qgis.utils import iface
from qgis.core import (
    QgsProject, QgsVectorLayer, QgsFeature, 
    QgsField,QgsSpatialIndex,QgsMessageLog, Qgis,
    QgsFields, QgsFeatureRequest, QgsWkbTypes,
    QgsVectorFileWriter,QgsCoordinateTransformContext
)
from pathlib import Path
from qgis.PyQt.QtWidgets import (QDialog, QVBoxLayout,
            QListWidget, QPushButton, QListWidgetItem)

from PyQt5.QtCore import QVariant
import os
from PyQt5.QtWidgets import QMessageBox, QFileDialog
from qgis.PyQt.QtCore import Qt

MINIMALNA_POWIERZCHNIA = 2.0  # m²

def create_extent_polygon(layer_name=None, output_path=None):
    """
    Tworzy wielokąt zasięgu warstwy i zapisuje go jako nową warstwę.
    
    Args:
        layer_name (str): Nazwa warstwy źródłowej
        output_path (str): Ścieżka do zapisania warstwy zasięgu
    """
    
    # Pobierz warstwę
    if layer_name is None:
        layer = iface.activeLayer()
        if layer is None:
            print("Brak aktywnej warstwy!")
            return None
    else:
        layers = QgsProject.instance().mapLayersByName(layer_name)
        if not layers:
            print(f"Nie znaleziono warstwy: {layer_name}")
            return None
        layer = layers[0]
    
    # Utwórz wielokąt zasięgu
    if output_path is None:
        output_path = 'memory:'
    
    result = processing.run("native:polygonfromlayerextent", {
        'INPUT': layer,
        'ROUND_TO': 0,
        'OUTPUT': output_path
    })
    
    extent_layer = result['OUTPUT']
    
    # Dodaj warstwę do projektu jeśli to warstwa w pamięci
    if output_path == 'memory:':
        extent_layer.setName(f"Zasięg_{layer.name()}")
        QgsProject.instance().addMapLayer(extent_layer)
        print(f"Utworzono warstwę zasięgu: Zasięg_{layer.name()}")
    else:
        print(f"Zapisano warstwę zasięgu: {output_path}")
    
    return extent_layer


def utworz_bufor_200m():
    """
    Funkcja tworząca bufor 200m wokół warstwy 'granica_obszaru_analizowanego'
    """
    
    # Nazwa warstwy wejściowej
    nazwa_warstwy = 'granica_obszaru_analizowanego'
    
    # Pobranie warstwy z projektu QGIS
    warstwa_wejsciowa = QgsProject.instance().mapLayersByName(nazwa_warstwy)
    
    # Sprawdzenie czy warstwa istnieje
    if not warstwa_wejsciowa:
        QgsMessageLog.logMessage(
            f"Nie znaleziono warstwy: {nazwa_warstwy}",
            "Buffer Script", 
            Qgis.Critical
        )
        print(f"BŁĄD: Nie znaleziono warstwy '{nazwa_warstwy}'")
        return None
    
    # Pobranie pierwszej warstwy z listy
    warstwa = warstwa_wejsciowa[0]
    
    # Sprawdzenie czy to warstwa wektorowa
    if not isinstance(warstwa, QgsVectorLayer):
        QgsMessageLog.logMessage(
            f"Warstwa {nazwa_warstwy} nie jest warstwą wektorową",
            "Buffer Script", 
            Qgis.Critical
        )
        print(f"BŁĄD: Warstwa '{nazwa_warstwy}' nie jest warstwą wektorową")
        return None
    
    print(f"Znaleziono warstwę: {nazwa_warstwy}")
    print(f"Liczba obiektów w warstwie: {warstwa.featureCount()}")
    
    # Parametry dla algorytmu buffer
    parametry = {
        'INPUT': warstwa,           # Warstwa wejściowa
        'DISTANCE': 200,            # Odległość bufora w metrach
        'SEGMENTS': 5,              # Liczba segmentów do aproksymacji krzywej
        'END_CAP_STYLE': 0,         # Styl zakończenia (0 = okrągły)
        'JOIN_STYLE': 0,            # Styl połączenia (0 = okrągły)
        'MITER_LIMIT': 2,           # Limit kąta ostrych połączeń
        'DISSOLVE': False,          # Czy rozpuścić nakładające się bufory
        'OUTPUT': 'TEMPORARY_OUTPUT'  # Warstwa tymczasowa
    }
    
    try:
        # Uruchomienie algorytmu buffer
        print("Tworzenie bufora 200m...")
        
        wynik = processing.run("native:buffer", parametry)
        
        # Pobranie warstwy wynikowej
        warstwa_bufor = wynik['OUTPUT']
        
        # Dodanie warstwy do projektu
        if warstwa_bufor:
            # Ustawienie nazwy warstwy
            warstwa_bufor.setName(f"{nazwa_warstwy}_bufor_200m")
            
            # Dodanie do projektu QGIS
            QgsProject.instance().addMapLayer(warstwa_bufor)
            
            print(f"Sukces! Utworzono bufor dla warstwy '{nazwa_warstwy}'")
            print(f"Nazwa nowej warstwy: {warstwa_bufor.name()}")
            print(f"Liczba obiektów w buforze: {warstwa_bufor.featureCount()}")
            
            # Logowanie sukcesu
            QgsMessageLog.logMessage(
                f"Utworzono bufor 200m dla warstwy {nazwa_warstwy}",
                "Buffer Script", 
                Qgis.Success
            )
            
            return warstwa_bufor
            
        else:
            print("BŁĄD: Nie udało się utworzyć bufora")
            return None
            
    except Exception as e:
        error_msg = f"Błąd podczas tworzenia bufora: {str(e)}"
        print(error_msg)
        QgsMessageLog.logMessage(error_msg, "Buffer Script", Qgis.Critical)
        return None
    
    
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
            
            
# BUFOR WOKÓŁ OBSZARU ANALIZY W CELU AGREGACJI DZIAŁEK WYSTEPUJACYCH NA GRANICY OBSZARU            
utworz_bufor_200m()     

# WYZNACZENIE DZIAŁEK W BUFORZE       
# Warstwy wejściowe
dzialki_layer = QgsProject.instance().mapLayersByName("dzialki  ewgib 2025 ostrowski szare poligony")[0]
granica_layer = QgsProject.instance().mapLayersByName("granica_obszaru_analizowanego_bufor_200m")[0]
# Załaduj geometrię z granicy (zakładamy, że to jeden poligon)
granica_geom = None
for feat in granica_layer.getFeatures():
    granica_geom = feat.geometry()
    break  # zakładamy jeden poligon

if granica_geom is None:
    raise Exception("Nie znaleziono geometrii w warstwie 'granica obszaru analizowanego'")

# Utwórz nową warstwę wynikową w pamięci
crs = dzialki_layer.crs().authid()
output_layer = QgsVectorLayer(f"Polygon?crs={crs}", "dzialki_w_obszarze_200m", "memory")
provider = output_layer.dataProvider()
provider.addAttributes(dzialki_layer.fields())
output_layer.updateFields()

# Przefiltruj działki
selected_features = []

for feat in dzialki_layer.getFeatures():
    if feat.geometry().intersects(granica_geom):
        new_feat = QgsFeature(output_layer.fields())
        new_feat.setGeometry(feat.geometry())
        for i, field in enumerate(dzialki_layer.fields()):
            new_feat.setAttribute(i, feat[field.name()])
        selected_features.append(new_feat)

provider.addFeatures(selected_features)
output_layer.updateExtents()

# Dodaj do projektu
QgsProject.instance().addMapLayer(output_layer)

print(f"Znaleziono {len(selected_features)} działek w obrębie granicy.")


# WYZNACZANIE BUDYNKOW WOKOL OBSZARU ANALIZY
#PRZYCIECIE WARSTWY BUDYNKOW W CELU PRZYSPIESZENIA AKCJI
# WYZNACZ ZASIEG 
extent_layer = create_extent_polygon('dzialki_w_obszarze_200m')
# Pobierz warstwy z projektu
warstwa_przycinana = QgsProject.instance().mapLayersByName('budynkiEWGiB_merge_bdot10k')[0]
warstwa_maski = QgsProject.instance().mapLayersByName('Zasięg_dzialki_w_obszarze_200m')[0]

# Parametry i uruchomienie narzędzia
parametry = {
    'INPUT': warstwa_przycinana,
    'PREDICATE': [0],  # 0 = przecina / zawiera się w (czyli punkt wewnątrz poligonu)
    'INTERSECT': warstwa_maski,
    'OUTPUT': 'memory:budynki_w_zasiegu'
}

wynik = processing.run("native:extractbylocation", parametry)
warstwa_przycieta = wynik['OUTPUT']

# Dodaj wynikową warstwę do projektu
QgsProject.instance().addMapLayer(warstwa_przycieta)


# WYBIERZ BUDYNKI LEZACE NA DZIAŁKACH W OBSZARZE
# Warstwy wejściowe
budynki_layer = QgsProject.instance().mapLayersByName('budynki_w_zasiegu')[0]
dzialki_layer = QgsProject.instance().mapLayersByName('dzialki_w_obszarze_200m')[0]

# EPSG z warstwy budynków
epsg = budynki_layer.crs().authid()

# Utwórz warstwę wynikową (budynki w granicy)
budynki_out = QgsVectorLayer(f"Polygon?crs={epsg}", "budynki_w_obszarze_200m", "memory")
dp_out = budynki_out.dataProvider()
dp_out.addAttributes(budynki_layer.fields())
budynki_out.updateFields()

# Próg przecięcia – min. 10% powierzchni budynku
PROG_PROCENT = 0.1

# Przetwarzanie budynków
for budynek in budynki_layer.getFeatures():
    geom_b = budynek.geometry()
    pow_b = geom_b.area()

    for dzialka in dzialki_layer.getFeatures():
        geom_d = dzialka.geometry()
        if geom_b.intersects(geom_d):
            czesc_wspolna = geom_b.intersection(geom_d)
            if czesc_wspolna.area() / pow_b >= PROG_PROCENT:
                nowy_feat = QgsFeature(budynki_out.fields())
                nowy_feat.setGeometry(geom_b)
                for field in budynki_layer.fields():
                    nowy_feat.setAttribute(field.name(), budynek[field.name()])
                dp_out.addFeature(nowy_feat)
                break  # Nie sprawdzaj kolejnych działek – wystarczy jedna spełniająca warunek

# Dodaj warstwę wynikową do projektu
QgsProject.instance().addMapLayer(budynki_out)



# AGREGACJA DZIAŁEK
# --- 1. Wczytanie warstw ---
dzialki_layer = QgsProject.instance().mapLayersByName('dzialki_w_obszarze_200m')[0]
budynki_layer = QgsProject.instance().mapLayersByName('budynki_w_obszarze_200m')[0]
epsg = dzialki_layer.crs().authid()
PROG_PROCENT = 0.1  # Próg powierzchni wspólnej

# --- 2. Budujemy listę grup działek połączonych przez budynki ---
grupy_dzialek = []

for budynek in budynki_layer.getFeatures():
    geom_b = budynek.geometry()
    pow_b = geom_b.area()

    przeciete = []
    for dzialka in dzialki_layer.getFeatures():
        if geom_b.intersects(dzialka.geometry()):
            wspolna = geom_b.intersection(dzialka.geometry())
            if wspolna.area() / pow_b >= PROG_PROCENT:
                przeciete.append(str(dzialka['ID_DZIALKI']))
    
    if przeciete:
        przypisano = False
        for grupa in grupy_dzialek:
            if set(przeciete) & grupa:
                grupa.update(przeciete)
                przypisano = True
                break
        if not przypisano:
            grupy_dzialek.append(set(przeciete))

# --- 3. Scalanie grup, które mają wspólne działki ---
def merge_groups(lista):
    wynik = []
    while lista:
        g1, *reszta = lista
        g1 = set(g1)
        zmiana = True
        while zmiana:
            zmiana = False
            nowa = []
            for g2 in reszta:
                if g1 & g2:
                    g1 |= g2
                    zmiana = True
                else:
                    nowa.append(g2)
            reszta = nowa
        wynik.append(g1)
        lista = reszta
    return wynik

grupy_dzialek = merge_groups(grupy_dzialek)

# --- 4. Słownik działek po ID ---
dzialki_dict = {str(f["ID_DZIALKI"]): f for f in dzialki_layer.getFeatures()}

# --- 5. Warstwa wynikowa ---
fields = QgsFields()
fields.append(QgsField("ID_DZIALKI", QVariant.String))
fields.append(QgsField("NUMER_DZIALKI", QVariant.String))
fields.append(QgsField("NUMER_OBREBU", QVariant.String))
fields.append(QgsField("POLE_EWIDENCYJNE", QVariant.Double))

agg_layer = QgsVectorLayer(f"MultiPolygon?crs={epsg}", "dzialki_zagregowane", "memory")
agg_dp = agg_layer.dataProvider()
agg_dp.addAttributes(fields)
agg_layer.updateFields()

uzyte_dzialki = set()

for grupa in grupy_dzialek:
    geometrie = []
    numery = []
    obreby = set()
    pole_suma = 0
    
    for dz_id in grupa:
        dz = dzialki_dict[dz_id]
        geom = dz.geometry()
        geometrie.append(geom)
        numery.append(str(dz['NUMER_DZIALKI']))
        obreby.add(str(dz['NUMER_OBREBU']))
        pole_suma += geom.area()
        uzyte_dzialki.add(dz.id())
    
    zlozona_geom = geometrie[0]
    for g in geometrie[1:]:
        zlozona_geom = zlozona_geom.combine(g)
    
    feat = QgsFeature(agg_layer.fields())
    feat.setGeometry(zlozona_geom)
    feat["ID_DZIALKI"] = ";".join(sorted(grupa))
    feat["NUMER_DZIALKI"] = ";".join(numery)
    feat["NUMER_OBREBU"] = list(obreby)[0] if len(obreby) == 1 else ";".join(sorted(obreby))
    feat["POLE_EWIDENCYJNE"] = pole_suma
    agg_dp.addFeature(feat)

# --- 6. Działki nieużyte (bez agregacji) ---
for dz in dzialki_layer.getFeatures():
    if dz.id() not in uzyte_dzialki:
        feat = QgsFeature(agg_layer.fields())
        feat.setGeometry(dz.geometry())
        feat["ID_DZIALKI"] = str(dz["ID_DZIALKI"])
        feat["NUMER_DZIALKI"] = str(dz["NUMER_DZIALKI"])
        feat["NUMER_OBREBU"] = str(dz["NUMER_OBREBU"])
        feat["POLE_EWIDENCYJNE"] = dz.geometry().area()
        agg_dp.addFeature(feat)

agg_layer.updateExtents()
QgsProject.instance().addMapLayer(agg_layer)

# JOIN ID_DZIALKI DO BUDYNKOW
# Warstwy
budynki_layer = QgsProject.instance().mapLayersByName("budynki_w_obszarze_200m")[0]
dzialki_layer = QgsProject.instance().mapLayersByName("dzialki_zagregowane")[0]

# Dodaj pole ID_DZIALKI jeśli nie istnieje
if 'ID_DZIALKI' not in [f.name() for f in budynki_layer.fields()]:
    budynki_layer.dataProvider().addAttributes([QgsField('ID_DZIALKI', QVariant.String)])
    budynki_layer.updateFields()

# Indeks przestrzenny działek
dzialki_index = QgsSpatialIndex(dzialki_layer.getFeatures())

# Mapowanie: id_budynku -> id_dzialki
updates = {}

for budynek in budynki_layer.getFeatures():
    geom_b = budynek.geometry()
    max_area = 0
    id_dzialki_final = None

    # Znajdź potencjalne działki
    dzialki_ids = dzialki_index.intersects(geom_b.boundingBox())
    
    for dzialka_id in dzialki_ids:
        dzialka = dzialki_layer.getFeature(dzialka_id)
        geom_d = dzialka.geometry()
        if geom_d and geom_b.intersects(geom_d):
            czesc_wspolna = geom_b.intersection(geom_d)
            wspolna_pow = czesc_wspolna.area()
            if wspolna_pow > max_area:
                max_area = wspolna_pow
                id_dzialki_final = dzialka['ID_DZIALKI']

    updates[budynek.id()] = id_dzialki_final

# Przypisz ID_DZIALKI do budynków
budynki_layer.startEditing()
for fid, id_dzialki in updates.items():
    budynki_layer.changeAttributeValue(fid, budynki_layer.fields().indexFromName('ID_DZIALKI'), id_dzialki)
budynki_layer.commitChanges()

print("Przypisano ID_DZIALKI do budynków.")

# WYZNACZANIE DZIALEK WE WLASCIWYM OBSZARZE
# Warstwy wejściowe
dzialki_layer = QgsProject.instance().mapLayersByName("dzialki_zagregowane")[0]
granica_layer = QgsProject.instance().mapLayersByName("granica_obszaru_analizowanego")[0]

# Załaduj geometrię z granicy (zakładamy, że to jeden poligon)
granica_geom = None
for feat in granica_layer.getFeatures():
    granica_geom = feat.geometry()
    break  # zakładamy jeden poligon

if granica_geom is None:
    raise Exception("Nie znaleziono geometrii w warstwie 'granica obszaru analizowanego'")

# Utwórz nową warstwę wynikową w pamięci
crs = dzialki_layer.crs().authid()
output_layer = QgsVectorLayer(f"Polygon?crs={crs}", "dzialki_w_obszarze", "memory")
provider = output_layer.dataProvider()
provider.addAttributes(dzialki_layer.fields())
output_layer.updateFields()

# Przefiltruj działki
selected_features = []

for feat in dzialki_layer.getFeatures():
    if feat.geometry().intersects(granica_geom):
        new_feat = QgsFeature(output_layer.fields())
        new_feat.setGeometry(feat.geometry())
        for i, field in enumerate(dzialki_layer.fields()):
            new_feat.setAttribute(i, feat[field.name()])
        selected_features.append(new_feat)

provider.addFeatures(selected_features)
output_layer.updateExtents()

# Dodaj do projektu
QgsProject.instance().addMapLayer(output_layer)

print(f"Znaleziono {len(selected_features)} działek w obrębie granicy.")


project_path = QgsProject.instance().fileName() 
if project_path:
    project_directory = Path(project_path).parent
    print(f"Katalog projektu: {project_directory}")
else:
    print("Projekt nie został jeszcze zapisany.")
    # Możesz ustawić domyślną ścieżkę lub zakończyć działanie
    project_directory = Path.cwd()  # Użyj bieżącego katalogu jako fallback

# Nazwa i ścieżka do pliku geopackage (poprawiona składnia)
temp_layer =output_layer
layer_name = temp_layer.name()
output_path = str(project_directory / f"{layer_name}.gpkg")

# Sprawdź czy warstwa tymczasowa istnieje
# temp_layer = QgsProject.instance().mapLayersByName(layer_name)

if temp_layer:
    # temp_layer = temp_layer[0]
    # Zapisz warstwę tymczasową do pliku geopackage
    options = QgsVectorFileWriter.SaveVectorOptions()
    options.driverName = 'GPKG'
    options.fileEncoding = 'UTF-8'
    options.layerName = layer_name
    
    # Poprawiona składnia zapisu
    result = QgsVectorFileWriter.writeAsVectorFormatV3(
        temp_layer, 
        output_path, 
        QgsCoordinateTransformContext(), 
        options
    )
    
    # Sprawdź wynik zapisu
    if result[0] == QgsVectorFileWriter.NoError:
        print(f"Warstwa została pomyślnie zapisana do: {output_path}")
        
        # Usuń starą warstwę tymczasową z projektu
        QgsProject.instance().removeMapLayer(temp_layer)
        
        # Wczytaj zapisaną warstwę do projektu
        saved_layer = QgsVectorLayer(f"{output_path}|layername={layer_name}", layer_name, "ogr")
        if saved_layer.isValid():
            QgsProject.instance().addMapLayer(saved_layer)
            print("Warstwa została pomyślnie wczytana do projektu.")
        else:
            print("Błąd podczas wczytywania warstwy.")
    else:
        print(f"Błąd podczas zapisywania warstwy: {result[1]}")
else:
    print("Nie znaleziono warstwy tymczasowej.")
    
# WYZNACZANIE BUDYNKOW WE WŁASCIWYM OBSZARZE
#PRZYCIECIE WARSTWY BUDYNKOW W CELU PRZYSPIESZENIA AKCJI
# WYZNACZ ZASIEG 
extent_layer = create_extent_polygon('dzialki_w_obszarze')
# Pobierz warstwy z projektu
warstwa_przycinana = QgsProject.instance().mapLayersByName('budynki_w_obszarze_200m')[0]
warstwa_maski = QgsProject.instance().mapLayersByName('Zasięg_dzialki_w_obszarze')[0]

# Parametry i uruchomienie narzędzia
parametry = {
    'INPUT': warstwa_przycinana,
    'PREDICATE': [0],  # 0 = przecina / zawiera się w (czyli punkt wewnątrz poligonu)
    'INTERSECT': warstwa_maski,
    'OUTPUT': 'memory:budynki_w_zasiegu'
}

wynik = processing.run("native:extractbylocation", parametry)
warstwa_przycieta = wynik['OUTPUT']

# Dodaj wynikową warstwę do projektu
QgsProject.instance().addMapLayer(warstwa_przycieta)


# WYBIERZ BUDYNKI LEZACE NA DZIAŁKACH W OBSZARZE
# Warstwy wejściowe
budynki_layer = QgsProject.instance().mapLayersByName('budynki_w_zasiegu')[0]
dzialki_layer = QgsProject.instance().mapLayersByName('dzialki_w_obszarze')[0]

# EPSG z warstwy budynków
epsg = budynki_layer.crs().authid()

# Utwórz warstwę wynikową (budynki w granicy)
budynki_out = QgsVectorLayer(f"Polygon?crs={epsg}", "budynki_w_obszarze", "memory")
dp_out = budynki_out.dataProvider()
dp_out.addAttributes(budynki_layer.fields())
budynki_out.updateFields()

# Próg przecięcia – min. 10% powierzchni budynku
PROG_PROCENT = 0.1

# Przetwarzanie budynków
for budynek in budynki_layer.getFeatures():
    geom_b = budynek.geometry()
    pow_b = geom_b.area()

    for dzialka in dzialki_layer.getFeatures():
        geom_d = dzialka.geometry()
        if geom_b.intersects(geom_d):
            czesc_wspolna = geom_b.intersection(geom_d)
            if czesc_wspolna.area() / pow_b >= PROG_PROCENT:
                nowy_feat = QgsFeature(budynki_out.fields())
                nowy_feat.setGeometry(geom_b)
                for field in budynki_layer.fields():
                    nowy_feat.setAttribute(field.name(), budynek[field.name()])
                dp_out.addFeature(nowy_feat)
                break  # Nie sprawdzaj kolejnych działek – wystarczy jedna spełniająca warunek

# Dodaj warstwę wynikową do projektu
QgsProject.instance().addMapLayer(budynki_out)


project_path = QgsProject.instance().fileName() 
if project_path:
    project_directory = Path(project_path).parent
    print(f"Katalog projektu: {project_directory}")
else:
    print("Projekt nie został jeszcze zapisany.")
    # Możesz ustawić domyślną ścieżkę lub zakończyć działanie
    project_directory = Path.cwd()  # Użyj bieżącego katalogu jako fallback

# Nazwa i ścieżka do pliku geopackage (poprawiona składnia)
temp_layer =budynki_out
layer_name = temp_layer.name()
output_path = str(project_directory / f"{layer_name}.gpkg")

# Sprawdź czy warstwa tymczasowa istnieje
# temp_layer = QgsProject.instance().mapLayersByName(layer_name)

if temp_layer:
    # temp_layer = temp_layer[0]
    # Zapisz warstwę tymczasową do pliku geopackage
    options = QgsVectorFileWriter.SaveVectorOptions()
    options.driverName = 'GPKG'
    options.fileEncoding = 'UTF-8'
    options.layerName = layer_name
    
    # Poprawiona składnia zapisu
    result = QgsVectorFileWriter.writeAsVectorFormatV3(
        temp_layer, 
        output_path, 
        QgsCoordinateTransformContext(), 
        options
    )
    
    # Sprawdź wynik zapisu
    if result[0] == QgsVectorFileWriter.NoError:
        print(f"Warstwa została pomyślnie zapisana do: {output_path}")
        
        # Usuń starą warstwę tymczasową z projektu
        QgsProject.instance().removeMapLayer(temp_layer)
        
        # Wczytaj zapisaną warstwę do projektu
        saved_layer = QgsVectorLayer(f"{output_path}|layername={layer_name}", layer_name, "ogr")
        if saved_layer.isValid():
            QgsProject.instance().addMapLayer(saved_layer)
            print("Warstwa została pomyślnie wczytana do projektu.")
        else:
            print("Błąd podczas wczytywania warstwy.")
    else:
        print(f"Błąd podczas zapisywania warstwy: {result[1]}")
else:
    print("Nie znaleziono warstwy tymczasowej.")
    
remove_memory_layers()

# FILTROWANIE
# === Konfiguracja ===
dzialki_layer = QgsProject.instance().mapLayersByName("dzialki_w_obszarze")[0]  # lub iface.activeLayer()
budynki_layer = QgsProject.instance().mapLayersByName("budynki_w_obszarze")[0]
pole_rodzaj = "rodzaj_zabudowy"
PROG_PROCENT = 0.1  # Minimalny udział powierzchni budynku na działce

# === Krok 1: Zbierz unikalne funkcje zabudowy z budynków ===
rodzaje_zabud = set()
for feat in budynki_layer.getFeatures():
    wartosc = feat[pole_rodzaj]
    if wartosc:
        rodzaje_zabud.update(wartosc.split(";"))

# === Krok 2: GUI do wyboru funkcji ===
class ZabudowaDialog(QDialog):
    def __init__(self, opcje):
        super().__init__()
        self.setWindowTitle("Wybierz funkcję zabudowy")
        self.resize(300, 400)
        layout = QVBoxLayout()
        self.listWidget = QListWidget()
        self.listWidget.setSelectionMode(QListWidget.MultiSelection)

        for opcja in sorted(opcje):
            item = QListWidgetItem(opcja)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.listWidget.addItem(item)

        self.ok_button = QPushButton("Filtruj działki")
        self.ok_button.clicked.connect(self.accept)

        layout.addWidget(self.listWidget)
        layout.addWidget(self.ok_button)
        self.setLayout(layout)

    def get_selected_options(self):
        return [
            self.listWidget.item(i).text()
            for i in range(self.listWidget.count())
            if self.listWidget.item(i).checkState() == Qt.Checked
        ]

dialog = ZabudowaDialog(rodzaje_zabud)
if not dialog.exec_():
    raise Exception("Anulowano wybór")

wybrane_funkcje = dialog.get_selected_options()
if not wybrane_funkcje:
    raise Exception("Nie wybrano żadnych funkcji zabudowy")

# === Krok 3: Znajdź działki z budynkami o wybranych funkcjach ===
dzialki_id_set = set()

for budynek in budynki_layer.getFeatures():
    rodzaje = str(budynek[pole_rodzaj]).split(";")
    if not any(r in wybrane_funkcje for r in rodzaje):
        continue

    geom_b = budynek.geometry()
    if not geom_b or not geom_b.isGeosValid():
        continue

    for dzialka in dzialki_layer.getFeatures():
        geom_d = dzialka.geometry()
        if not geom_d or not geom_d.isGeosValid():
            continue
        if geom_b.intersects(geom_d):
            wspolna = geom_b.intersection(geom_d)
            if wspolna.area() / geom_b.area() >= PROG_PROCENT:
                dzialki_id_set.add(dzialka.id())

# === Krok 4: Stwórz nową warstwę z wybranymi działkami ===
output_layer = QgsVectorLayer(f"MultiPolygon?crs={dzialki_layer.crs().authid()}", "dzialki_zgodne_z_funkcja", "memory")
output_layer.dataProvider().addAttributes(dzialki_layer.fields())
output_layer.updateFields()

for dzialka in dzialki_layer.getFeatures(QgsFeatureRequest().setFilterFids(list(dzialki_id_set))):
    new_feat = QgsFeature(output_layer.fields())
    new_feat.setGeometry(dzialka.geometry())
    for field in dzialki_layer.fields():
        new_feat.setAttribute(field.name(), dzialka[field.name()])
    output_layer.dataProvider().addFeature(new_feat)

output_layer.updateExtents()
QgsProject.instance().addMapLayer(output_layer)

print(f"Utworzono warstwę z {len(dzialki_id_set)} działkami.")

# Stwórz nową pustą warstwę wektorową o tej samej strukturze
filtered_layer = QgsVectorLayer('Polygon?crs=' + budynki_layer.crs().authid(), 'budynki_zgodne_z_funkcja', 'memory')
filtered_layer_data = filtered_layer.dataProvider()
filtered_layer_data.addAttributes(budynki_layer.fields())
filtered_layer.updateFields()

# Przefiltruj i dodaj pasujące obiekty do nowej warstwy
for feature in budynki_layer.getFeatures():
    if feature['rodzaj_zabudowy'] in wybrane_funkcje:
        filtered_layer_data.addFeature(feature)

# Dodaj przefiltrowaną warstwę do projektu
QgsProject.instance().addMapLayer(filtered_layer)


# DOŁACZANIE ID_DZIALKI DO BUDYNKÓW
# Warstwy
budynki_layer = QgsProject.instance().mapLayersByName("budynki_zgodne_z_funkcja")[0]
dzialki_layer = QgsProject.instance().mapLayersByName("dzialki_zgodne_z_funkcja")[0]

# Dodaj pole ID_DZIALKI jeśli nie istnieje
if 'ID_DZIALKI' not in [f.name() for f in budynki_layer.fields()]:
    budynki_layer.dataProvider().addAttributes([QgsField('ID_DZIALKI', QVariant.String)])
    budynki_layer.updateFields()

# Indeks przestrzenny działek
dzialki_index = QgsSpatialIndex(dzialki_layer.getFeatures())

# Mapowanie: id_budynku -> id_dzialki
updates = {}

for budynek in budynki_layer.getFeatures():
    geom_b = budynek.geometry()
    max_area = 0
    id_dzialki_final = None

    # Znajdź potencjalne działki
    dzialki_ids = dzialki_index.intersects(geom_b.boundingBox())
    
    for dzialka_id in dzialki_ids:
        dzialka = dzialki_layer.getFeature(dzialka_id)
        geom_d = dzialka.geometry()
        if geom_d and geom_b.intersects(geom_d):
            czesc_wspolna = geom_b.intersection(geom_d)
            wspolna_pow = czesc_wspolna.area()
            if wspolna_pow > max_area:
                max_area = wspolna_pow
                id_dzialki_final = dzialka['ID_DZIALKI']

    updates[budynek.id()] = id_dzialki_final

# Przypisz ID_DZIALKI do budynków
budynki_layer.startEditing()
for fid, id_dzialki in updates.items():
    budynki_layer.changeAttributeValue(fid, budynki_layer.fields().indexFromName('ID_DZIALKI'), id_dzialki)
budynki_layer.commitChanges()

print("Przypisano ID_DZIALKI do budynków.")

zapis_do_gpkg("dzialki_zgodne_z_funkcja")
zapis_do_gpkg("budynki_zgodne_z_funkcja")
remove_memory_layers()

apply_qml_style_to_layer(layer= "dzialki_zgodne_z_funkcja",
                         qml_file_path="/home/adrian/Documents/JXPROJEKT/style/dzialki w obszarze analizy.qml", 
                         show_messages=True)

apply_qml_style_to_layer(layer= "budynki_zgodne_z_funkcja",
                         qml_file_path="/home/adrian/Documents/JXPROJEKT/style/budynki_do_analizy.qml", 
                         show_messages=True)


def analiza_przestrzenna():
    """
    Skrypt do automatycznej analizy przestrzennej warstw:
    - dzialki_zgodne_z_funkcja
    - budynki_zgodne_z_funkcja  
    - granica_terenu
    """
    
    # Pobierz warstwy z projektu
    project = QgsProject.instance()
    
    # Znajdź warstwy po nazwach
    dzialki_layer = None
    budynki_layer = None
    granica_layer = None
    
    for layer in project.mapLayers().values():
        if layer.name() == 'dzialki_zgodne_z_funkcja':
            dzialki_layer = layer
        elif layer.name() == 'budynki_zgodne_z_funkcja':
            budynki_layer = layer
        elif layer.name() == 'granica_terenu':
            granica_layer = layer
    
    # Sprawdź czy wszystkie warstwy zostały znalezione
    if not dzialki_layer:
        print("BŁĄD: Nie znaleziono warstwy 'dzialki_zgodne_z_funkcja'")
        return
    if not budynki_layer:
        print("BŁĄD: Nie znaleziono warstwy 'budynki_zgodne_z_funkcja'")
        return  
    if not granica_layer:
        print("BŁĄD: Nie znaleziono warstwy 'granica_terenu'")
        return
    
    print("Znaleziono wszystkie wymagane warstwy. Rozpoczynam analizę...")
    
    # Próg pokrycia - 10%
    PROG_PROCENT = 0.1
    
    # Pobierz geometrię granicy terenu (zakładamy jeden poligon)
    granica_geom = None
    for feature in granica_layer.getFeatures():
        granica_geom = feature.geometry()
        break
    
    if not granica_geom:
        print("BŁĄD: Brak geometrii w warstwie granica_terenu")
        return
        
    print(f"Geometria granicy terenu: powierzchnia = {round(granica_geom.area(), 2)}")
    
    # ===== PRZETWARZANIE DZIAŁEK =====
    
    dzialki_do_usuniecia = []
    dzialki_do_przyciecia = []
    
    print("Analizuję działki...")
    
    for dzialka in dzialki_layer.getFeatures():
        dzialka_geom = dzialka.geometry()
        dzialka_area = dzialka_geom.area()
        
        if granica_geom.intersects(dzialka_geom):
            # Sprawdź czy działka jest całkowicie zawarta w granicy terenu
            if granica_geom.contains(dzialka_geom):
                # Działka całkowicie w granicy - usuń
                dzialki_do_usuniecia.append(dzialka.id())
                print(f"Działka ID {dzialka.id()}: powierzchnia={round(dzialka_area,2)} -> DO USUNIĘCIA (całkowicie w granicy)")
            else:
                # Działka częściowo w granicy - przytnij
                intersection = granica_geom.intersection(dzialka_geom)
                wspolna_area = intersection.area()
                procent_pokrycia = wspolna_area / dzialka_area
                
                print(f"Działka ID {dzialka.id()}: powierzchnia={round(dzialka_area,2)}, "
                      f"wspólna={round(wspolna_area,2)}, pokrycie={round(procent_pokrycia*100,1)}%")
                
                # Przytnij działkę (odejmij granicę terenu)
                dzialki_do_przyciecia.append(dzialka)
                print("  -> DO PRZYCIĘCIA (częściowo w granicy)")
    
    print("\nPODSUMOWANIE DZIAŁEK:")
    print(f"Działek do usunięcia: {len(dzialki_do_usuniecia)}")
    print(f"Działek do przycięcia: {len(dzialki_do_przyciecia)}")
    
    # Włącz edycję warstwy działek
    dzialki_layer.startEditing()
    
    # Usuń działki całkowicie pokryte
    if dzialki_do_usuniecia:
        dzialki_layer.dataProvider().deleteFeatures(dzialki_do_usuniecia)
        print(f"Usunięto {len(dzialki_do_usuniecia)} działek")
    
    # Przytnij działki częściowo pokrywające się
    for dzialka in dzialki_do_przyciecia:
        dzialka_geom = dzialka.geometry()
        
        # Wykonaj operację difference (odjęcie)
        new_geom = dzialka_geom.difference(granica_geom)
        new_area = new_geom.area()
        
        if (
            not new_geom.isEmpty()
            and new_geom.type() == QgsWkbTypes.PolygonGeometry
            and new_area >= MINIMALNA_POWIERZCHNIA
        ):
            # Zaktualizuj geometrię działki
            dzialki_layer.dataProvider().changeGeometryValues({dzialka.id(): new_geom})
            
            # Przelicz pole POLE_EWIDENCYJNE
            rounded_area = round(new_area, 2)
            field_index = dzialki_layer.fields().indexFromName('POLE_EWIDENCYJNE')
            if field_index != -1:
                dzialki_layer.dataProvider().changeAttributeValues({dzialka.id(): {field_index: rounded_area}})
                print(f"Przycięto działkę ID: {dzialka.id()}, nowe pole: {rounded_area}")
                print(f"Geometria działki ID {dzialka.id()} po przycięciu: {new_geom.asWkt()}")

            else:
                print("UWAGA: Pole 'POLE_EWIDENCYJNE' nie zostało znalezione w warstwie działek")
        else:
            dzialki_layer.dataProvider().deleteFeatures([dzialka.id()])
            print(f"❌ Usunięto działkę ID: {dzialka.id()} (geometria nieprawidłowa lub powierzchnia < {MINIMALNA_POWIERZCHNIA} m²)")
            
    # Zatwierdź zmiany w warstwie działek
    dzialki_layer.commitChanges()
    dzialki_layer.updateExtents()
    
    # ===== PRZETWARZANIE BUDYNKÓW =====
    
    print("\nAnalizuję budynki...")
    
    budynki_do_usuniecia = []
    
    for budynek in budynki_layer.getFeatures():
        budynek_geom = budynek.geometry()
        budynek_area = budynek_geom.area()
        
        if granica_geom.intersects(budynek_geom):
            # Oblicz powierzchnię wspólną
            intersection = granica_geom.intersection(budynek_geom)
            wspolna_area = intersection.area()
            procent_pokrycia = wspolna_area / budynek_area
            
            print(f"Budynek ID {budynek.id()}: powierzchnia={round(budynek_area,2)}, "
                  f"wspólna={round(wspolna_area,2)}, pokrycie={round(procent_pokrycia*100,1)}%")
            
            # Jeśli pokrycie > próg - usuń budynek
            if procent_pokrycia > PROG_PROCENT:
                budynki_do_usuniecia.append(budynek.id())
                print(f"  -> DO USUNIĘCIA (pokrycie > {PROG_PROCENT*100}%)")
            else:
                print(f"  -> BEZ ZMIAN (pokrycie <= {PROG_PROCENT*100}%)")
    
    print("\nPODSUMOWANIE BUDYNKÓW:")
    print(f"Budynków do usunięcia: {len(budynki_do_usuniecia)}")
    
    # Włącz edycję warstwy budynków
    budynki_layer.startEditing()
    
    # Usuń budynki przekraczające próg pokrycia
    if budynki_do_usuniecia:
        budynki_layer.dataProvider().deleteFeatures(budynki_do_usuniecia)
        print(f"Usunięto {len(budynki_do_usuniecia)} budynków")
    
    # Zatwierdź zmiany w warstwie budynków
    budynki_layer.commitChanges()
    budynki_layer.updateExtents()
    
    # Odśwież warstwy na mapie
    dzialki_layer.triggerRepaint()
    budynki_layer.triggerRepaint()
    
    print("\n=== ANALIZA ZAKOŃCZONA ===")
    print(f"- Usunięto działek: {len(dzialki_do_usuniecia)}")
    print(f"- Przycięto działek: {len(dzialki_do_przyciecia)}")
    print(f"- Usunięto budynków: {len(budynki_do_usuniecia)}")

# ===== DODATKOWE FUNKCJE POMOCNICZE =====

def sprawdz_warstwy():
    """Funkcja pomocnicza do sprawdzenia dostępnych warstw"""
    project = QgsProject.instance()
    print("Dostępne warstwy w projekcie:")
    for layer in project.mapLayers().values():
        print(f"- {layer.name()} (typ: {layer.type()})")


# ===== URUCHOMIENIE SKRYPTU =====
    # Uruchom główną analizę
analiza_przestrzenna()