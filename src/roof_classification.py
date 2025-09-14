#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 09:24:57 2025

@author: adrian
"""

import csv
import os
from qgis.core import (QgsProject, QgsVectorLayer, QgsFeature, 
                       Qgis,QgsMessageLog,
    QgsFields, QgsField, QgsVectorFileWriter, 
    QgsCoordinateTransformContext)
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QComboBox, QPushButton, QLabel, QHBoxLayout, QLineEdit
import processing
from pathlib import Path
from PyQt5.QtCore import QVariant
import time
from qgis.utils import iface
from PyQt5.QtWidgets import QMessageBox, QFileDialog

# Ścieżka do pliku CSV
project_path = QgsProject.instance().fileName()
project_directory = os.path.dirname(project_path)
csv_file = os.path.join(project_directory, "klasyfikacja_dachow.csv")
nazwa_pola = "ID_BUDYNKU"
kategorie = ["płaski", "jednospadowy", "dwuspadowy", "czterospadowy", "wielospadowy", "Pomiń", "Inny", "Zakończ"]
SCRIPTS_PATH = os.path.dirname(os.path.abspath(__file__))


    
# Skrypt do dodania kolumny "nachylenie" do warstwy QGIS
# i wypełnienia jej wartościami z kolumny "nachylenie_chmura" (NULL -> 0)

def add_nachylenie_column():
    """
    Dodaje kolumnę 'nachylenie' do warstwy 'budynki_z_szer_elew_front'
    i wypełnia ją wartościami z kolumny 'nachylenie_chmura' (NULL zamienia na 0)
    """
    
    # Nazwa warstwy do przetworzenia
    layer_name = "budynki_z_szer_elew_front"
    
    # Pobranie warstwy z aktualnego projektu
    layer = QgsProject.instance().mapLayersByName(layer_name)
    
    if not layer:
        print(f"Błąd: Nie znaleziono warstwy o nazwie '{layer_name}'")
        return False
    
    layer = layer[0]  # Pierwszy element z listy warstw
    
    # Sprawdzenie czy warstwa ma kolumnę źródłową
    source_field = "nachylenie_chmura"
    target_field = "nachylenie"
    
    fields = layer.fields()
    source_field_exists = fields.indexFromName(source_field) != -1
    target_field_exists = fields.indexFromName(target_field) != -1
    
    if not source_field_exists:
        print(f"Błąd: Nie znaleziono kolumny źródłowej '{source_field}'")
        return False
    
    # Rozpoczęcie edycji warstwy
    layer.startEditing()
    
    try:
        # Dodanie nowej kolumny jeśli nie istnieje
        if not target_field_exists:
            new_field = QgsField(target_field, QVariant.Double)
            layer.dataProvider().addAttributes([new_field])
            layer.updateFields()
            print(f"Dodano nową kolumnę '{target_field}'")
        else:
            print(f"Kolumna '{target_field}' już istnieje - aktualizowanie wartości")
        
        # Pobranie indeksów kolumn
        source_idx = layer.fields().indexFromName(source_field)
        target_idx = layer.fields().indexFromName(target_field)
        
        # Aktualizacja wartości dla wszystkich obiektów
        updated_count = 0
        null_count = 0
        
        for feature in layer.getFeatures():
            source_value = feature[source_idx]
            
            # Sprawdzenie czy wartość jest NULL i zamiana na 0
            if source_value is None or source_value == '':
                new_value = 0
                null_count += 1
            else:
                try:
                    new_value = int(source_value)
                except (ValueError, TypeError):
                    new_value = 0
                    null_count += 1
            
            # Aktualizacja wartości w nowej kolumnie
            layer.changeAttributeValue(feature.id(), target_idx, new_value)
            updated_count += 1
        
        # Zatwierdzenie zmian
        layer.commitChanges()
        
        print("Operacja zakończona pomyślnie!")
        print(f"Zaktualizowano {updated_count} rekordów")
        print(f"Zamieniono {null_count} wartości NULL na 0")
        
        return True
        
    except Exception as e:
        # Cofnięcie zmian w przypadku błędu
        layer.rollBack()
        print(f"Błąd podczas wykonywania operacji: {str(e)}")
        return False


def zapis_do_gpkg(temp_layer):
    project_path = QgsProject.instance().fileName() 
    if project_path:
        project_directory = Path(project_path).parent
        print(f"Katalog projektu: {project_directory}")
    else:
        print("Projekt nie został jeszcze zapisany.")
        project_directory = Path.cwd()

    layer_name = temp_layer.name()
    output_path = str(project_directory / f"{layer_name}.gpkg")

    if temp_layer:
        options = QgsVectorFileWriter.SaveVectorOptions()
        options.driverName = 'GPKG'
        options.fileEncoding = 'UTF-8'
        options.layerName = layer_name
        
        result = QgsVectorFileWriter.writeAsVectorFormatV3(
            temp_layer, 
            output_path, 
            QgsCoordinateTransformContext(), 
            options
        )
        
        if result[0] == QgsVectorFileWriter.NoError:
            print(f"Warstwa została pomyślnie zapisana do: {output_path}")
            QgsProject.instance().removeMapLayer(temp_layer)
            
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


# Tworzymy plik CSV, jeśli nie istnieje
if not os.path.exists(csv_file):
    with open(csv_file, mode="w", newline="", encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["ID_BUDYNKU", "Kategoria"])

        
def popraw_nachylenie_dla_plaskich(warstwa):
    idx_kat = warstwa.fields().indexFromName("Kategoria")
    idx_nach = warstwa.fields().indexFromName("nachylenie")
    licznik = 0

    if idx_kat == -1 or idx_nach == -1:
        print("❌ Nie znaleziono wymaganych pól w warstwie.")
        return

    warstwa.startEditing()
    for f in warstwa.getFeatures():
        if f["Kategoria"] == "płaski":
            try:
                nach = float(f["nachylenie"])
                if nach > 5:
                    f[idx_nach] = 5
                    warstwa.updateFeature(f)
                    licznik += 1
            except Exception as e:
                print(f"⚠️ Błąd przy analizie feature ID {f.id()}: {e}")
    warstwa.commitChanges()
    print(f"🔧 Zmieniono nachylenie dla {licznik} dachów płaskich > 5°")


# def przewidz_kategorie(feature):
#     """Prosta analiza geometrii poligonu do przewidywania typu dachu"""
#     geom = feature.geometry()
#     powierzchnia = geom.area()
#     bbox = geom.boundingBox()
#     szerokosc = bbox.width()
#     wysokosc = bbox.height()
#     stosunek = szerokosc / wysokosc if wysokosc != 0 else 0
#     warstwa = QgsProject.instance().mapLayersByName('budynki_z_szer_elew_front')[0]
#     idx_nach = warstwa.fields().indexFromName("nachylenie_chmura")
#     nachylenie_chmura = feature[idx_nach]

    
#     if nachylenie_chmura < 10:
#         return "płaski"
#     elif stosunek >= 2.0:
#         return "dwuspadowy"
#     else:
#         return "czterospadowy"

def przewidz_kategorie(feature):
    """Prosta analiza geometrii poligonu do przewidywania typu dachu"""
    warstwa = QgsProject.instance().mapLayersByName('budynki_z_predykcjami')[0]
    idx_pred = warstwa.fields().indexFromName("PREDYKCJA")
    idx_prob = warstwa.fields().indexFromName("PEWNOSC")

    prediction = feature[idx_pred]
    pewnosc = feature[idx_prob]
    
    return [prediction,pewnosc]



class KategoriaDialog(QDialog):
    def __init__(self, id_poligonu, sugerowana_kategoria, pewnosc_predykcji,kategorie, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Wybór kategorii dla {id_poligonu}")
        layout = QVBoxLayout()
        self.label = QLabel(f"Dach {id_poligonu} - \nsugerowana: {sugerowana_kategoria}; prawdopodobienstwo: {pewnosc_predykcji}")
        layout.addWidget(self.label)
        self.combo = QComboBox()
        self.combo.addItems(kategorie)
        self.combo.setCurrentText(sugerowana_kategoria)
        layout.addWidget(self.combo)
        self.input_other_category = QLineEdit()
        self.input_other_category.setPlaceholderText("Wpisz kategorię (jeśli 'Inny')")
        self.input_other_category.setEnabled(False)
        layout.addWidget(self.input_other_category)
        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("OK")
        self.cancel_button = QPushButton("Anuluj")
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)
        self.setLayout(layout)
        self.combo.currentTextChanged.connect(self.on_category_change)
        self.move(100, 100)
        
    def on_category_change(self):
        if self.combo.currentText() == "Inny":
            self.input_other_category.setEnabled(True)
        else:
            self.input_other_category.setEnabled(False)
    
    def get_selected_category(self):
        if self.combo.currentText() == "Inny":
            return self.input_other_category.text()
        return self.combo.currentText()


def remove_memory_layers():
    for lyr in QgsProject.instance().mapLayers().values():
        if lyr.dataProvider().name() == 'memory':
            QgsProject.instance().removeMapLayer(lyr.id())
     
            
def get_prediction_data_dir():
    """Zwraca ścieżkę do katalogu prediction_data w Downloads"""
    # Folder Downloads użytkownika
    downloads_dir = Path.home() / "Downloads"
    prediction_dir = downloads_dir / "prediction_data"
    
    # Utwórz folder jeśli nie istnieje
    prediction_dir.mkdir(exist_ok=True)
    
    return str(prediction_dir)


def wczytaj_csv_do_qgis(sciezka_csv, nazwa_kolumny_x=None, nazwa_kolumny_y=None, 
                        separator=None, crs_kod='EPSG:2180', nazwa_warstwy=None, 
                        kodowanie='UTF-8', auto_detect=True):
    """
    Wczytuje plik CSV jako warstwę do QGIS z automatyczną detekcją separatora
    
    Args:
        sciezka_csv (str): Ścieżka do pliku CSV
        nazwa_kolumny_x (str): Nazwa kolumny z współrzędną X (jeśli None, wczyta jako tabelę atrybutów)
        nazwa_kolumny_y (str): Nazwa kolumny z współrzędną Y (jeśli None, wczyta jako tabelę atrybutów)  
        separator (str): Separator używany w pliku CSV (jeśli None, auto-detekcja)
        crs_kod (str): Kod układu współrzędnych
        nazwa_warstwy (str): Nazwa warstwy (jeśli None, użyje nazwy pliku)
        kodowanie (str): Kodowanie pliku
        auto_detect (bool): Czy automatycznie wykryć separator
    
    Returns:
        QgsVectorLayer: Wczytana warstwa lub None jeśli błąd
    """
    import csv
    from urllib.parse import quote
    
    # Sprawdź czy plik istnieje
    if not os.path.exists(sciezka_csv):
        print(f"Błąd: Plik {sciezka_csv} nie istnieje!")
        return None
    
    # Automatyczna detekcja separatora
    if separator is None and auto_detect:
        try:
            with open(sciezka_csv, 'r', encoding=kodowanie) as file:
                # Przeczytaj pierwszą linię
                first_line = file.readline().strip()
                # Sprawdź różne separatory
                separators = [' ', ',', ';', '\t', '|']
                separator_counts = {}
                
                for sep in separators:
                    count = first_line.count(sep)
                    if count > 0:
                        separator_counts[sep] = count
                
                if separator_counts:
                    # Wybierz separator z największą liczbą wystąpień
                    separator = max(separator_counts.items(), key=lambda x: x[1])[0]
                    print(f"Wykryty separator: '{separator}' (wystąpienia: {separator_counts[separator]})")
                else:
                    separator = ' '  # Domyślny separator spacja
                    print("Nie wykryto separatora, używam spacji")
        except Exception as e:
            print(f"Błąd przy detekcji separatora: {e}")
            separator = ' '  # Fallback na spację
    
    # Jeśli nadal nie ma separatora, użyj spacji
    if separator is None:
        separator = ' '
    
    # Jeśli nie podano nazwy warstwy, użyj nazwy pliku
    if nazwa_warstwy is None:
        nazwa_warstwy = os.path.splitext(os.path.basename(sciezka_csv))[0]
    
    # Zakoduj ścieżkę dla URI
    sciezka_zakodowana = quote(sciezka_csv.replace('\\', '/'))
    
    # Mapowanie separatorów dla QGIS URI
    separator_mapping = {
        ' ': '%20',     # spacja
        ',': ',',       # przecinek
        ';': ';',       # średnik  
        '\t': '\\t',    # tabulator
        '|': '|'        # pipe
    }
    
    separator_uri = separator_mapping.get(separator, separator)
    
    # Jeśli są współrzędne, stwórz URI dla warstwy punktowej
    if nazwa_kolumny_x and nazwa_kolumny_y:
        uri = f"file:///{sciezka_zakodowana}?delimiter={separator_uri}&xField={nazwa_kolumny_x}&yField={nazwa_kolumny_y}&crs={crs_kod}&encoding={kodowanie}"
    else:
        # Jeśli brak współrzędnych, wczytaj jako tabelę atrybutów
        uri = f"file:///{sciezka_zakodowana}?delimiter={separator_uri}&encoding={kodowanie}"
    
    print(f"URI: {uri}")
    
    # Stwórz warstwę
    warstwa = QgsVectorLayer(uri, nazwa_warstwy, "delimitedtext")
    
    # Sprawdź czy warstwa została poprawnie wczytana
    if not warstwa.isValid():
        print(f"Błąd: Nie można wczytać warstwy z pliku {sciezka_csv}")
        print(f"Separator: '{separator}' (URI: '{separator_uri}')")
        print(f"Kodowanie: {kodowanie}")
        
        # Spróbuj z różnymi kodowaniami
        kodowania_do_sprawdzenia = ['UTF-8', 'Windows-1250', 'ISO-8859-2', 'CP1252']
        for kod in kodowania_do_sprawdzenia:
            if kod != kodowanie:
                print(f"Próbuję z kodowaniem: {kod}")
                if nazwa_kolumny_x and nazwa_kolumny_y:
                    uri_test = f"file:///{sciezka_zakodowana}?delimiter={separator_uri}&xField={nazwa_kolumny_x}&yField={nazwa_kolumny_y}&crs={crs_kod}&encoding={kod}"
                else:
                    uri_test = f"file:///{sciezka_zakodowana}?delimiter={separator_uri}&encoding={kod}"
                
                warstwa_test = QgsVectorLayer(uri_test, nazwa_warstwy, "delimitedtext")
                if warstwa_test.isValid():
                    warstwa = warstwa_test
                    kodowanie = kod
                    print(f"✓ Udało się z kodowaniem: {kod}")
                    break
        
        if not warstwa.isValid():
            print("Sprawdź czy nazwy kolumn i separator są poprawne")
            # Wyświetl próbkę pliku dla diagnostyki
            try:
                with open(sciezka_csv, 'r', encoding='utf-8', errors='ignore') as f:
                    print("Pierwsze 3 linie pliku:")
                    for i, line in enumerate(f):
                        if i >= 3:
                            break
                        print(f"  {i+1}: {repr(line.strip())}")
            except:
                pass
            return None
    
    # Dodaj warstwę do projektu
    QgsProject.instance().addMapLayer(warstwa)
    
    # Wyświetl informacje o warstwie
    print(f"✓ Wczytano warstwę: {nazwa_warstwy}")
    print(f"  - Liczba obiektów: {warstwa.featureCount()}")
    print(f"  - CRS: {warstwa.crs().authid()}")
    print(f"  - Separator: '{separator}'")
    print(f"  - Kodowanie: {kodowanie}")
    print(f"  - Kolumny: {[field.name() for field in warstwa.fields()]}")
    
    # Loguj do QGIS
    QgsMessageLog.logMessage(
        f"Wczytano warstwę CSV: {nazwa_warstwy} ({warstwa.featureCount()} obiektów)",
        "CSV Loader", 
        Qgis.Info
    )
    
    return warstwa

def wait_for_file():
    """
    Czeka na pojawienie się pliku building_predictions.csv w katalogu PREDICTION_DIR
    """
    # Utwórz pełną ścieżkę do pliku
    prediction_dir = PREDICTION_DIR

    file_path = Path(prediction_dir) / "building_predictions.csv"
    
    print(f"Oczekiwanie na plik: {file_path}")
    
    # Pętla oczekiwania na plik
    while True:
        if file_path.exists() and file_path.is_file():
            print(f"Plik znaleziony: {file_path}")
            return str(file_path)
        else:
            print(f"Plik nie istnieje. Sprawdzam ponownie za 5 sekund...")
            time.sleep(5)
            
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


PREDICTION_DIR = get_prediction_data_dir()

    
    # Czekaj na plik
file_path = wait_for_file()
    
# TUTAJ ROZPOCZYNA SIĘ WŁAŚCIWY KOD PROGRAMU
print("=" * 50)
print("PLIK ZNALEZIONY! Wykonywanie dalszego kodu...")
print("=" * 50)
   
try:    
    prediction_layer = wczytaj_csv_do_qgis(
        f"{PREDICTION_DIR}/building_predictions.csv",
        nazwa_kolumny_x=None,  # Brak współrzędnych - wczytaj jako tabelę
        nazwa_kolumny_y=None,
        nazwa_warstwy="Predykcje_budynkow"
    )

except Exception as e:
        print(f"Błąd podczas przetwarzania pliku: {e}")

params = {
    'INPUT': QgsProject.instance().mapLayersByName('budynki_z_szer_elew_front')[0],     # ZMIANA: warstwa z geometrią jako INPUT
    'FIELD': 'ID_BUDYNKU',
    'INPUT_2': QgsProject.instance().mapLayersByName('Predykcje_budynkow')[0],          # ZMIANA: tabela CSV jako INPUT_2  
    'FIELD_2': 'ID_BUDYNKU',
    'FIELDS_TO_COPY': ['PREDYKCJA', 'PEWNOSC'],                                        # ZMIANA: bez zagnieżdżonych nawiasów
    'METHOD': 1,
    'DISCARD_NONMATCHING': False,
    'PREFIX': '',
    'OUTPUT': 'memory:budynki_with_predictions'                                        # ZMIANA: inna nazwa
}

joined_layer = processing.run("native:joinattributestable", params)['OUTPUT']
joined_layer.setName('budynki_z_predykcjami')
QgsProject.instance().addMapLayer(joined_layer)
style_name = "budynki_do_analizy.qml"
style_path = os.path.join(SCRIPTS_PATH, style_name)
apply_qml_style_to_layer("budynki_z_predykcjami", style_path)

# GŁÓWNA CZĘŚĆ KODU
# dodaje kolumne 'nachylenie' i przenosi do niej warstosci obliczone przez algorytm
add_nachylenie_column()

warstwa_poligonowa = QgsProject.instance().mapLayersByName('budynki_z_predykcjami')[0]

# Sprawdź czy warstwa została wybrana
if not warstwa_poligonowa:
    print("❌ Nie wybrano aktywnej warstwy!")
    exit()

# Wczytaj wcześniej zaklasyfikowane budynki
zaklasyfikowane_budynki = set()
if os.path.exists(csv_file):
    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            zaklasyfikowane_budynki.add(row['ID_BUDYNKU'])
    print(f"📂 Wczytano {len(zaklasyfikowane_budynki)} wcześniej zaklasyfikowanych dachów.")

# POPRAWA: Najpierw sprawdź wszystkie obiekty i znajdź te niezaklasyfikowane
wszystkie_features = list(warstwa_poligonowa.getFeatures())
niezaklasyfikowane_features = []

for feature in wszystkie_features:
    id_poligonu = str(feature[nazwa_pola])
    if id_poligonu not in zaklasyfikowane_budynki:
        niezaklasyfikowane_features.append(feature)

print(f"🔍 Znaleziono {len(niezaklasyfikowane_features)} niezaklasyfikowanych dachów do przetworzenia.")
print(f"📊 Pomijam {len(wszystkie_features) - len(niezaklasyfikowane_features)} już zaklasyfikowanych dachów.")

# Jeśli wszystkie są zaklasyfikowane, zakończ
if not niezaklasyfikowane_features:
    print("✅ Wszystkie dachy są już zaklasyfikowane!")
else:
    # Przetwarzaj tylko niezaklasyfikowane
    for feature in niezaklasyfikowane_features:
        id_poligonu = str(feature[nazwa_pola])
        
        # Wyświetl dach na mapie
        bbox = feature.geometry().boundingBox()
        margines = max(bbox.width() * 0.4, bbox.height() * 0.4)  
        nowy_bbox = bbox.buffered(margines)  
        iface.mapCanvas().setExtent(nowy_bbox)
        warstwa_poligonowa.removeSelection()
        warstwa_poligonowa.select(feature.id())
        iface.mapCanvas().refresh()
        
        # Wyświetl dialog
        sugerowana_kategoria = przewidz_kategorie(feature)[0]
        pewnosc_predykcji = przewidz_kategorie(feature)[1]
        dialog = KategoriaDialog(id_poligonu, sugerowana_kategoria,pewnosc_predykcji, kategorie, iface.mainWindow())
        dialog.show()
        result = dialog.exec_()
        
        if result == QDialog.Accepted:
            kategoria = dialog.get_selected_category()
        elif result == QDialog.Rejected:
            print("🛑 Przerwano wybór kategorii!")
            break
            
        if kategoria == "Zakończ":
            print("🛑 Zakończenie procesu przez użytkownika.")
            break
            
        if kategoria == "Pomiń":
            print(f"🚀 Pominięto dach {id_poligonu}")
            continue
            
        # Zapisz do CSV
        with open(csv_file, mode="a", newline="", encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([id_poligonu, kategoria])
            
        print(f"✅ Zapisano: {id_poligonu} (kategoria: {kategoria})")
        
        # Dodaj do zestawu zaklasyfikowanych (żeby uniknąć duplikatów w tej sesji)
        zaklasyfikowane_budynki.add(id_poligonu)

print("🎉 Proces klasyfikacji zakończony!")

# === RESZTA KODU POZOSTAJE BEZ ZMIAN ===
# Wczytaj dane z CSV
with open(csv_file, newline='', encoding='utf-8') as f: 
    reader = csv.DictReader(f) 
    fieldnames = reader.fieldnames
    
    fields = QgsFields() 
    for name in fieldnames: 
        if name != "id": 
            fields.append(QgsField(name, QVariant.String))
            
    csv_memory_layer = QgsVectorLayer("None", "klasyfikacja_dachow", "memory") 
    csv_memory_layer.dataProvider().addAttributes(fields) 
    csv_memory_layer.updateFields()     
     
    features = [] 
    for row in reader: 
        feat = QgsFeature() 
        feat.initAttributes(len(fields)) 
        for i, name in enumerate(fieldnames): 
            if name != "id": 
                feat[i] = row[name] 
        features.append(feat)  
            
    csv_memory_layer.dataProvider().addFeatures(features) 
    csv_memory_layer.updateExtents()         
                
    QgsProject.instance().addMapLayer(csv_memory_layer) 

# DODAJE KLASYFIKACJE DACHÓW DO WARSTWY BUDYNKÓW 
try:
    params = { 
        'INPUT': QgsProject.instance().mapLayersByName('budynki_z_predykcjami')[0], 
        'FIELD': 'ID_BUDYNKU', 
        'INPUT_2': QgsProject.instance().mapLayersByName('klasyfikacja_dachow')[0], 
        'FIELD_2': 'ID_BUDYNKU', 
        'FIELDS_TO_COPY': ['Kategoria'], 
        'METHOD': 1, 
        'DISCARD_NONMATCHING': False, 
        'PREFIX': '', 
        'OUTPUT': 'memory:budynki_parametry' 
    }

    joined_layer = processing.run("native:joinattributestable", params)['OUTPUT'] 
    QgsProject.instance().addMapLayer(joined_layer) 
    zapis_do_gpkg(joined_layer)
    layer = QgsProject.instance().mapLayersByName("budynki_parametry")[0]
    popraw_nachylenie_dla_plaskich(layer)
except Exception as e:
    print(f"Błąd podczas łączenia z warstwą budynków: {e}")

# DODAJE KLASYFIKACJE DACHÓW DO WARSTWY PUNKTÓW 
try:
    params = { 
        'INPUT': QgsProject.instance().mapLayersByName('Classification_6_with_IDs')[0], 
        'FIELD': 'ID_BUDYNKU', 
        'INPUT_2': QgsProject.instance().mapLayersByName('klasyfikacja_dachow')[0], 
        'FIELD_2': 'ID_BUDYNKU', 
        'FIELDS_TO_COPY': ['Kategoria'], 
        'METHOD': 1, 
        'DISCARD_NONMATCHING': False, 
        'PREFIX': '', 
        'OUTPUT': 'memory:Classification_6_with_roofsKategories' 
    }

    joined_layer = processing.run("native:joinattributestable", params)['OUTPUT']
    QgsProject.instance().addMapLayer(joined_layer) 
    zapis_do_gpkg(joined_layer)
except Exception as e:
    print(f"Nie znaleziono warstwy 'Classification_6_with_IDs' lub błąd: {e}")
    
remove_memory_layers()
