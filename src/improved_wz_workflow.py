import os
import json
import time
from datetime import datetime
from PyQt5.QtWidgets import (QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QListWidget, QMessageBox, 
                             QInputDialog, QTextEdit, QFrame, QScrollArea,
                             QGroupBox, QSizePolicy, QSpacerItem)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from qgis.core import QgsProject
from qgis.core import (QgsWkbTypes,QgsFields,QgsFeature,
   QgsField, QgsVectorLayer, QgsProcessingContext,QgsProcessingFeedback
)
from qgis.PyQt.QtCore import QVariant
from pathlib import Path
from qgis import processing
import pandas as pd

# import sys
# if sys.version_info[0] >= 3:
#     import codecs
#     sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
# Zabezpieczenie dla przypadku gdy iface nie jest dostępne
# -*- coding: utf-8 -*-

# Import bezpieczny dla iface
try:
    from qgis.utils import iface
    IFACE_AVAILABLE = True
except ImportError:
    print("UWAGA: iface nie jest dostępne")
    iface = None
    IFACE_AVAILABLE = False

# Zabezpieczenie funkcji używających iface
def safe_iface_call(method_name, *args, **kwargs):
    """Bezpieczne wywołanie metod iface"""
    if not IFACE_AVAILABLE or not iface:
        print(f"UWAGA: Nie można wywołać iface.{method_name} - iface niedostępne")
        return None
    
    try:
        method = getattr(iface, method_name)
        return method(*args, **kwargs)
    except Exception as e:
        print(f"BŁĄD wywołania iface.{method_name}: {e}")
        return None
    
    
# Stała ścieżka do katalogu ze skryptami
SCRIPTS_PATH = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_FILE = os.path.join(SCRIPTS_PATH, 'workflow_checkpoint.json')
project_path = QgsProject.instance().fileName()
project_directory = Path(project_path).parent

# Style CSS dla interfejsu
MODERN_STYLE = """
QDockWidget {
    background-color: #f8f9fa;
    border: 1px solid #dee2e6;
    font-family: 'Segoe UI', Arial, sans-serif;
}

QDockWidget::title {
    background-color: #007bff;
    color: white;
    padding: 8px;
    font-weight: bold;
    font-size: 14px;
    text-align: center;
}

QPushButton {
    background-color: #007bff;
    color: white;
    border: none;
    padding: 10px 15px;
    border-radius: 5px;
    font-size: 12px;
    font-weight: 500;
    min-height: 35px;
}

QPushButton:hover {
    background-color: #0056b3;
}

QPushButton:pressed {
    background-color: #004085;
}

QPushButton:disabled {
    background-color: #6c757d;
    color: #adb5bd;
}

QPushButton.secondary {
    background-color: #6c757d;
}

QPushButton.secondary:hover {
    background-color: #545b62;
}

QPushButton.danger {
    background-color: #dc3545;
}

QPushButton.danger:hover {
    background-color: #c82333;
}

QPushButton.success {
    background-color: #28a745;
}

QPushButton.success:hover {
    background-color: #1e7e34;
}

QTextEdit {
    background-color: white;
    border: 1px solid #ced4da;
    border-radius: 4px;
    padding: 8px;
    font-size: 11px;
    font-family: 'Consolas', 'Courier New', monospace;
}

QLabel {
    color: #495057;
    font-size: 12px;
}

QLabel.title {
    font-size: 16px;
    font-weight: bold;
    color: #212529;
    padding: 5px 0;
}

QLabel.subtitle {
    font-size: 14px;
    font-weight: 600;
    color: #495057;
    padding: 3px 0;
}

QLabel.info {
    color: #17a2b8;
    font-weight: 500;
}

QLabel.warning {
    color: #ffc107;
    font-weight: 500;
}

QLabel.success {
    color: #28a745;
    font-weight: 500;
}

QLabel.error {
    color: #dc3545;
    font-weight: 500;
}

QGroupBox {
    font-weight: bold;
    border: 2px solid #ced4da;
    border-radius: 5px;
    margin-top: 10px;
    padding-top: 10px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px 0 5px;
    color: #495057;
}

QListWidget {
    border: 1px solid #ced4da;
    border-radius: 4px;
    background-color: white;
    font-size: 11px;
}

QFrame.separator {
    background-color: #dee2e6;
    max-height: 1px;
    margin: 5px 0;
}

QScrollArea {
    border: none;
    background-color: transparent;
}
"""







def utworz_folder(sciezka_folderu):
    try:
        os.makedirs(sciezka_folderu, exist_ok=True)
        print(f"Folder utworzony lub już istnieje: {sciezka_folderu}")
    except Exception as e:
        print(f"Błąd podczas tworzenia folderu: {e}")


def zmien_nazwy_plikow(folder, mapa_nazw):
    """
    folder: ścieżka do folderu z plikami
    mapa_nazw: słownik {'stara_nazwa': 'nowa_nazwa'}
    """
    for stara_nazwa, nowa_nazwa in mapa_nazw.items():
        # Zakładamy, że pliki to np. SHP, więc zmieniamy wszystkie powiązane rozszerzenia
        rozszerzenia = ['.csv']
        znaleziono = False

        for ext in rozszerzenia:
            stary_plik = os.path.join(folder, stara_nazwa + ext)
            nowy_plik = os.path.join(folder, nowa_nazwa + ext)

            if os.path.exists(stary_plik):
                os.rename(stary_plik, nowy_plik)
                znaleziono = True

        if znaleziono:
            print(f"Zmieniono nazwę: {stara_nazwa} → {nowa_nazwa}")
        else:
            print(f"Nie znaleziono plików dla: {stara_nazwa}")

mapa_nazw = {
    'zabudowa mieszkaniowa_dachy' : 'budynki_parametry_0_dachy',
    'Zabudowa produkcyjna _ usługowa _ gospodarcza_dachy' : 'budynki_parametry_1_dachy',
    'budynek transportu _ łączności_dachy' : 'budynki_parametry_2_dachy',
    'budynek niemieszkalny_dachy' : 'budynki_parametry_3_dachy',
    'zabudowa handlowo-usługowa_dachy' : 'budynki_parametry_4_dachy',
    'budynek biurowy_dachy' : 'budynki_parametry_5_dachy',
    'zabudowa przemysłowa_dachy' : 'budynki_parametry_6_dachy',
    'Zbiornik _ silos _ budynek magazynowy_dachy' : 'budynki_parametry_7_dachy',
    
}

def zapis_do_gpkg(layer_name, remove_old=False):
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


def remove_memory_layers():
    for lyr in QgsProject.instance().mapLayers().values():
        if lyr.dataProvider().name() == 'memory':
            QgsProject.instance().removeMapLayer(lyr.id())


class LineMeasurementController:
    def __init__(self):
        self.measurement_layer = None
        
    def create_line_layer(self):
        """Tworzy warstwę do rysowania linii zabudowy z automatycznym obliczaniem długości"""
        try:
            # Tworzenie nowej warstwy wektorowej typu LineString
            self.measurement_layer = QgsVectorLayer('LineString?crs=EPSG:2177', 'linie_zabudowy', 'memory')
            
            if not self.measurement_layer.isValid():
                print("❌ Błąd: Nie udało się utworzyć warstwy!")
                return None
            
            # Pobieranie dostawcy danych warstwy
            provider = self.measurement_layer.dataProvider()
            
            # Dodawanie pola distance
            fields = [
                QgsField('distance', QVariant.Double, 'double', 10, 2)
            ]
            provider.addAttributes(fields)
            self.measurement_layer.updateFields()
            
            # Dodawanie warstwy do projektu
            QgsProject.instance().addMapLayer(self.measurement_layer)
            
            # Przełączenie warstwy w tryb edycji
            self.measurement_layer.startEditing()
            
            print(f"✅ Utworzono warstwę '{self.measurement_layer.name()}' w trybie edycji")
            return self.measurement_layer
            
        except Exception as e:
            print(f"❌ Błąd podczas tworzenia warstwy: {e}")
            return None
    
    def setup_auto_length_calculation(self):
        """Konfiguruje automatyczne obliczanie długości linii"""
        if not self.measurement_layer or not self.measurement_layer.isValid():
            return
            
        try:
            # Znajdowanie indeksu pola distance
            distance_field_index = self.measurement_layer.fields().indexFromName('distance')
            
            if distance_field_index == -1:
                print("❌ Błąd: Nie znaleziono pola 'distance'")
                return
            
            # Sprawdź czy sygnały już są podłączone i odłącz je
            try:
                self.measurement_layer.featureAdded.disconnect()
                self.measurement_layer.geometryChanged.disconnect()
            except:
                pass  # Ignoruj jeśli nie były podłączone
            
            # Podłącz sygnały z zabezpieczeniami
            self.measurement_layer.featureAdded.connect(self.safe_on_feature_added)
            self.measurement_layer.geometryChanged.connect(self.safe_on_geometry_changed)
            
            print("✅ Skonfigurowano automatyczne obliczanie długości")
            
        except Exception as e:
            print(f"❌ Błąd podczas konfiguracji sygnałów: {e}")
    
    def safe_on_feature_added(self, feature_id):
        """Bezpieczna wersja obsługi dodania nowego obiektu"""
        try:
                # Sprawdź wszystkie warunki
            if not self.measurement_layer:
                print("UWAGA: measurement_layer nie istnieje")
                return
                
            if not self.measurement_layer.isValid():
                print("UWAGA: measurement_layer jest nieprawidłowa")
                return
                
            if not self.measurement_layer.isEditable():
                print("UWAGA: measurement_layer nie jest w trybie edycji")
                return
                
            feature = self.measurement_layer.getFeature(feature_id)
            if not feature.hasGeometry():
                return
            
            # Znajdowanie indeksu pola distance
            distance_field_index = self.measurement_layer.fields().indexFromName('distance')
            
            if distance_field_index == -1:
                return
            
            # Obliczanie długości geometrii
            length = feature.geometry().length()
            
            # Aktualizacja pola distance
            self.measurement_layer.changeAttributeValue(feature_id, distance_field_index, round(length, 2))
            
            print(f"📏 Dodano linię o długości: {round(length, 2)} m")
            
        except Exception as e:
            print(f"BŁĄD w safe_on_feature_added: {e}")
            import traceback
            traceback.print_exc()
    
    def safe_on_geometry_changed(self, feature_id, geometry):
        """Bezpieczna wersja obsługi zmiany geometrii"""
        try:
            if not self.measurement_layer or not self.measurement_layer.isValid():
                return
            if not self.measurement_layer.isEditable() or not geometry:
                return
            
            # Znajdowanie indeksu pola distance
            distance_field_index = self.measurement_layer.fields().indexFromName('distance')
            if distance_field_index == -1:
                return
            
            # Obliczanie nowej długości
            length = geometry.length()
            # Aktualizacja pola distance
            self.measurement_layer.changeAttributeValue(feature_id, distance_field_index, round(length, 2))
            print(f"🔄 Zaktualizowano długość: {round(length, 2)} m")
            
        except Exception as e:
            print(f"❌ Błąd w safe_on_geometry_changed: {e}")
    
    def start_measurement_process(self):
        """Rozpoczyna proces mierzenia linii zabudowy"""
        try:
            # Utworzenie warstwy do rysowania
            if not self.create_line_layer():
                return False
                
            # Konfiguracja automatycznego obliczania
            self.setup_auto_length_calculation()
            
            # Ustawienie aktywnej warstwy
            iface.setActiveLayer(self.measurement_layer)
            
            print("🚀 Rozpoczęto proces rysowania linii zabudowy")
            return True
            
        except Exception as e:
            print(f"❌ Błąd podczas uruchamiania procesu: {e}")
            return False
    
    def finish_measurement(self):
        """Zakończenie procesu mierzenia"""
        try:
            print("🔄 Kończenie pomiarów...")
            
            # Odłącz sygnały
            print("🔄 Odłączanie sygnałów...")
            if self.measurement_layer and self.measurement_layer.isValid():
                try:
                    self.measurement_layer.featureAdded.disconnect()
                    self.measurement_layer.geometryChanged.disconnect()
                    print("✅ Sygnały odłączone")
                except Exception as signal_error:
                    print(f"⚠️ Problem z odłączaniem sygnałów: {signal_error}")
                
                # Zakończ edycję warstwy
                print("🔄 Zakończenie edycji warstwy...")
                if self.measurement_layer.isEditable():
                    self.measurement_layer.commitChanges()
                    print("✅ Zmiany zapisane w warstwie")
            
            print("✅ Zakończono pomiary linii zabudowy")
            
            try:
                iface.messageBar().pushSuccess("Zakończono", "Pomiary linii zabudowy zostały zapisane!")
            except Exception as msg_error:
                print(f"⚠️ Problem z messageBar: {msg_error}")
            
        except Exception as e:
            print(f"❌ Błąd podczas zakończenia: {e}")
            import traceback
            traceback.print_exc()


# Globalna instancja kontrolera
line_controller = None


def zestawienie_dachow(nazwa_warstwy):
    """Funkcja zestawienia dachów - placeholder do implementacji później"""
    print(f"Wykonuję zestawienie dachów dla warstwy: {nazwa_warstwy}")
    """Funkcja do zestawienia dachów dla danej warstwy"""
    try:
        layers = QgsProject.instance().mapLayersByName(nazwa_warstwy)
        if not layers:
            print(f"Nie znaleziono warstwy: {nazwa_warstwy}")
            return False
        
        layer = layers[0]
        layer_name = layer.name()
        
        # Konwersja do pandas DataFrame
        data = []
        for feature in layer.getFeatures():
            attrs = feature.attributes()
            fields = [field.name() for field in layer.fields()]
            data.append(dict(zip(fields, attrs)))
        
        budynki = pd.DataFrame(data)
        
        # Sprawdzenie czy mamy wymagane kolumny
        if 'Kategoria' not in budynki.columns or 'nachylenie' not in budynki.columns:
            print(f"Warstwa {nazwa_warstwy} nie zawiera wymaganych kolumn: 'Kategoria' lub 'nachylenie'")
            return False
        
        # Sprawdzenie struktury danych
        print("Kolumny w warstwie wektorowej:")
        print(budynki.columns.tolist())
        print("\nPierwsze 5 wierszy:")
        print(budynki[['Kategoria', 'nachylenie']].head())
        
        # Utworzenie dataframe 'dachy' z analizą według kategorii
        dachy = budynki.groupby('Kategoria').agg({
            'nachylenie': ['count', 'min', 'max']
        }).reset_index()
        
        # Spłaszczenie kolumn wielopoziomowych
        dachy.columns = ['Kategoria', 'liczba_wystapien', 'min_nachylenie', 'max_nachylenie']
        dachy[['min_nachylenie', 'max_nachylenie']] = dachy[['min_nachylenie', 'max_nachylenie']].astype(int)
        dachy = dachy.sort_values('liczba_wystapien', ascending=False)
        
        # Wyświetlenie wyników
        print("\nDataframe 'dachy':")
        print(dachy)
        
        # Dodatkowe statystyki dla lepszego zrozumienia danych
        print("\nDodatkowe informacje:")
        print(f"Łączna liczba budynków: {len(budynki)}")
        print(f"Liczba unikalnych kategorii: {budynki['Kategoria'].nunique()}")
        print(f"Kategorie dachów: {budynki['Kategoria'].unique()}")
        
        # Ścieżka do folderu projektu
        project_folder = QgsProject.instance().homePath()
        output_path = f"{project_folder}/budynki_parametry_dachy/{layer_name}_dachy.csv"
        
        # Zapisanie dataframe jako CSV
        dachy.to_csv(output_path, index=False, encoding='utf-8')
        print(f"\nDataframe został zapisany jako '{output_path}'")
        
        # Opcjonalne: wyświetlenie podstawowych statystyk dla nachylenia
        print("\nPodstawowe statystyki nachylenia dachów:")
        print(budynki['nachylenie'].describe())
        
        return True
    except Exception as e:
            print(f"BŁĄD w zestawienie_dachow: {e}")
            import traceback
            traceback.print_exc()
            return False
        

def podziel_budynki(buildings_layer_name, output_dir, context, feedback):
    """
    Podział warstwy budynki_parametry według atrybutu rodzaj_zabudowy
    Zwraca listę nazw utworzonych warstw
    """
    print("Wykonuję podział budynków według funkcji")
    
    # Sprawdź czy warstwa istnieje
    buildings_layers = QgsProject.instance().mapLayersByName(buildings_layer_name)
    if not buildings_layers:
        raise Exception(f"Nie znaleziono warstwy: {buildings_layer_name}")
    
    buildings_layer = buildings_layers[0]
    
    # Konwertuj Path na string jeśli to konieczne
    if isinstance(output_dir, Path):
        output_dir_str = str(output_dir)
    else:
        output_dir_str = output_dir
    
    # Upewnij się, że katalog istnieje
    Path(output_dir_str).mkdir(parents=True, exist_ok=True)
    
    print(f"Katalog wyjściowy: {output_dir_str}")
    
    # Parametry podziału
    split_params = {
        'INPUT': buildings_layer,
        'FIELD': 'rodzaj_zabudowy',
        'FILE_TYPE': 0,  # GeoPackage
        'OUTPUT': output_dir_str,  # Katalog jako string
        'PREFIX_FIELD':False
    }
    
    print("Parametry podziału:")
    for key, value in split_params.items():
        print(f"  {key}: {value}")
    
    try:
        # Wykonanie podziału
        result = processing.run("native:splitvectorlayer", split_params, context=context, feedback=feedback)
        
        # Pobierz listę utworzonych warstw
        output_layers = result['OUTPUT_LAYERS']
        print(f"✓ Podział zakończony. Utworzono {len(output_layers)} warstw:")
        
        # Wyświetl nazwy utworzonych warstw
        for i, layer_path in enumerate(output_layers):
            print(f"  {i+1}. {layer_path}")
        
        return output_layers
        
    except Exception as e:
        print(f"Błąd podczas podziału warstwy: {str(e)}")
        raise
   
   
    
def setup_processing():
    """Konfiguracja środowiska przetwarzania"""
    context = QgsProcessingContext()
    feedback = QgsProcessingFeedback()
    return context, feedback


class StatusIndicator(QLabel):
    """Wskaźnik statusu z kolorowym tłem"""
    def __init__(self, text="", status="info"):
        super().__init__(text)
        self.set_status(status)
    
    def set_status(self, status):
        """Ustaw status: info, warning, success, error"""
        if status == "info":
            self.setProperty("class", "info")
        elif status == "warning":
            self.setProperty("class", "warning")
        elif status == "success":
            self.setProperty("class", "success")
        elif status == "error":
            self.setProperty("class", "error")
        self.style().unpolish(self)
        self.style().polish(self)


class ModernButton(QPushButton):
    """Nowoczesny przycisk z różnymi stylami"""
    def __init__(self, text="", button_type="primary"):
        super().__init__(text)
        self.set_type(button_type)
    
    def set_type(self, button_type):
        """Ustaw typ przycisku: primary, secondary, success, danger"""
        if button_type == "secondary":
            self.setProperty("class", "secondary")
        elif button_type == "success":
            self.setProperty("class", "success")
        elif button_type == "danger":
            self.setProperty("class", "danger")
        self.style().unpolish(self)
        self.style().polish(self)



class WZWorkflowDockWidget(QDockWidget):
    def __init__(self):
        super().__init__("Analiza WZ - Workflow")
        self.current_step = 0
        self.workflow_steps = []
        self.liczba_budynkow = 0
        self.rozne_funkcje = False
        
        # Zastosuj nowoczesny styl
        self.setStyleSheet(MODERN_STYLE)
        
        self.init_ui()
        self.detect_current_step()
        self.start_workflow()
        
    def init_ui(self):
        """Inicjalizacja interfejsu użytkownika"""
        # Główny widget z scroll area
        main_widget = QWidget()
        scroll_area = QScrollArea()
        scroll_area.setWidget(main_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # Sekcja statusu
        status_group = QGroupBox("Status Workflow")
        status_layout = QVBoxLayout()
        
        self.current_step_label = QLabel("Krok: 0")
        self.current_step_label.setProperty("class", "subtitle")
        
        self.status_indicator = StatusIndicator("Oczekiwanie na rozpoczęcie", "info")
        
        status_layout.addWidget(self.current_step_label)
        status_layout.addWidget(self.status_indicator)
        status_group.setLayout(status_layout)
        main_layout.addWidget(status_group)
        
        # Separator
        separator1 = QFrame()
        separator1.setProperty("class", "separator")
        separator1.setFrameShape(QFrame.HLine)
        main_layout.addWidget(separator1)
        
        # Sekcja komunikatów
        messages_group = QGroupBox("Komunikaty systemowe")
        messages_layout = QVBoxLayout()
        
        self.message_area = QTextEdit()
        self.message_area.setMaximumHeight(120)
        self.message_area.setMinimumHeight(80)
        self.message_area.setReadOnly(True)
        
        messages_layout.addWidget(self.message_area)
        messages_group.setLayout(messages_layout)
        main_layout.addWidget(messages_group)
        
        # Separator
        separator2 = QFrame()
        separator2.setProperty("class", "separator")
        separator2.setFrameShape(QFrame.HLine)
        main_layout.addWidget(separator2)
        
        # Sekcja akcji
        actions_group = QGroupBox("Akcje")
        actions_layout = QVBoxLayout()
        actions_layout.setSpacing(10)
        
        # Kontener na główne przyciski akcji
        self.button_container = QWidget()
        self.button_layout = QVBoxLayout()
        self.button_layout.setSpacing(8)
        self.button_container.setLayout(self.button_layout)
        
        actions_layout.addWidget(self.button_container)
        
        # Spacer elastyczny
        spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        actions_layout.addItem(spacer)
        
        actions_group.setLayout(actions_layout)
        main_layout.addWidget(actions_group)
        
        # Sekcja kontroli workflow
        control_group = QGroupBox("Kontrola Workflow")
        control_layout = QVBoxLayout()
        control_layout.setSpacing(8)
        
        # Przyciski kontrolne w poziomie
        control_buttons_layout = QHBoxLayout()
        
        back_btn = ModernButton("◄ Poprzedni krok", "secondary")
        back_btn.clicked.connect(self.go_back_to_checkpoint)
        
        reset_btn = ModernButton("🔄 Reset", "danger")
        reset_btn.clicked.connect(self.reset_workflow)
        
        control_buttons_layout.addWidget(back_btn)
        control_buttons_layout.addWidget(reset_btn)
        
        control_layout.addLayout(control_buttons_layout)
        control_group.setLayout(control_layout)
        main_layout.addWidget(control_group)
        
        main_widget.setLayout(main_layout)
        self.setWidget(scroll_area)
        
        # Ustaw minimalną szerokość
        self.setMinimumWidth(350)
    
    def add_message(self, message, message_type="info"):
        """Dodaj wiadomość do obszaru komunikatów z kolorowym oznaczeniem"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Ikony dla różnych typów wiadomości
        icons = {
            "info": "ℹ️",
            "success": "✅", 
            "warning": "⚠️",
            "error": "❌"
        }
        
        icon = icons.get(message_type, "ℹ️")
        formatted_message = f"[{timestamp}] {icon} {message}"
        
        self.message_area.append(formatted_message)
        
        # Automatycznie przewiń do końca
        scrollbar = self.message_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
        # Zaktualizuj status indicator
        if message_type == "error":
            self.status_indicator.setText("Wystąpił błąd")
            self.status_indicator.set_status("error")
        elif message_type == "success":
            self.status_indicator.setText("Operacja zakończona pomyślnie")
            self.status_indicator.set_status("success")
        elif message_type == "warning":
            self.status_indicator.setText("Uwaga - wymagana interwencja")
            self.status_indicator.set_status("warning")
    
    def clear_buttons(self):
        """Wyczyść wszystkie przyciski akcji"""
        for i in reversed(range(self.button_layout.count())): 
            child = self.button_layout.itemAt(i).widget()
            if child:
                child.setParent(None)
    
    def update_step_display(self):
        """Zaktualizuj wyświetlanie aktualnego kroku"""
        self.current_step_label.setText(f"Krok: {self.current_step}")
        
    
    
    def detect_current_step(self):
        """Wykryj aktualny krok na podstawie istniejących warstw"""
        existing_layers = [layer.name() for layer in QgsProject.instance().mapLayers().values()]
        
        # Mapowanie warstw do kroków workflow
        layer_step_mapping = [
            ("granica_terenu", 1),
            ("granica_obszaru_analizowanego", 2), 
            ("wymiary", 3),
            ("dzialki_w_obszarze", 5),
            ("budynki_z_szer_elew_front", 6),
            ("Classification_2", 7),
            ("punkty_pbc_wyniki_predykcji", 8),
            ("działki_ze_wskaznikami", 10),
            ("budynki_parametry", 11),
        ]
        
        # Znajdź najwyższy krok dla którego istnieją warstwy
        detected_step = 0
        for layer_name, step in layer_step_mapping:
            if layer_name in existing_layers:
                detected_step = max(detected_step, step)
        
        self.current_step = detected_step
        self.update_step_display()
        
        # Wczytaj dodatkowe dane z checkpoint jeśli istnieje
        self.load_checkpoint_data()
        
        if self.current_step > 0:
            self.add_message(f"Wykryto kontynuację workflow od kroku {self.current_step}", "info")
        
    def show_main_menu(self):
        """Pokaż główne menu wyboru"""
        self.clear_buttons()
        self.add_message("Co dziś robimy?", "info")
        self.status_indicator.setText("Wybór typu analizy")
        self.status_indicator.set_status("info")
        
        # Główne opcje
        analiza_btn = ModernButton("📊 Analiza do WZ", "primary")
        analiza_btn.clicked.connect(self.wybierz_analize)
        self.button_layout.addWidget(analiza_btn)
        
        mapa_btn = ModernButton("🗺️ Mapa do WZ", "secondary")
        mapa_btn.clicked.connect(self.wybierz_mape)
        mapa_btn.setEnabled(True)
        mapa_btn.setToolTip("Ta funkcja będzie dostępna w przyszłej wersji")
        self.button_layout.addWidget(mapa_btn)
        
    def wybierz_analize(self):
        """Menu wyboru typu analizy - z opcją kontynuacji"""
        self.clear_buttons()
        self.add_message("Wybierz typ analizy:", "info")
        self.status_indicator.setText("Wybór typu analizy")
        
        # Sprawdź czy istnieje checkpoint z poprzednią pracą
        if self.current_step > 0:
            # Pokaż opcję kontynuacji
            continue_btn = ModernButton(f"🔄 Kontynuuj od kroku {self.current_step}", "success")
            continue_btn.clicked.connect(self.potwierdz_kontynuacje)
            self.button_layout.addWidget(continue_btn)
            
            self.add_message(f"Wykryto poprzednią pracę - możesz kontynuować od kroku {self.current_step}", "info")
        
        # Zawsze pokaż opcję rozpoczęcia od nowa
        standard_btn = ModernButton("🏠 Nowa standardowa analiza", "primary")
        standard_btn.clicked.connect(self.nowa_standardowa_analiza)
        self.button_layout.addWidget(standard_btn)
        
        funkcje_btn = ModernButton("🔧 Użyj dostępnych funkcji", "secondary")
        funkcje_btn.clicked.connect(self.pokaz_funkcje)
        self.button_layout.addWidget(funkcje_btn)
    
    def wybierz_mape(self):
        self.execute_script("wierzcholki_z_zapisem.py")
        self.execute_script("olz_i_wymiarowanie_2_0.py")
        self.add_message
        
    def potwierdz_kontynuacje(self):
        """Potwierdź kontynuację workflow"""
        self.clear_buttons()
        self.add_message(f"Kontynuacja workflow od kroku {self.current_step}", "info")
        
        # Pokaż informację o aktualnym kroku
        step_descriptions = {
            1: "Odnalezienie terenu inwestycji",
            2: "Tworzenie bufora obszaru analizowanego", 
            3: "Rysowanie wymiarów",
            4: "Zapisywanie wymiarów",
            5: "Wyznaczanie działek i budynków",
            6: "Pomiar elewacji frontowych",
            7: "Przetwarzanie chmury punktów",
            8: "Klasyfikacja PBC",
            9: "Weryfikacja punktów",
            10: "Obliczanie wskaźników",
            11: "Parametry budynków",
            12: "Wyznaczanie linii zabudowy",
            13: "Dane działki inwestora",
            14: "Generowanie wyników końcowych"
        }
        
        current_description = step_descriptions.get(self.current_step, "Nieznany krok")
        self.add_message(f"Aktualny krok: {current_description}", "info")
        
        # Przyciski potwierdzenia
        tak_btn = ModernButton("✅ Tak, kontynuuj", "success") 
        tak_btn.clicked.connect(self.resume_from_detected_step)
        self.button_layout.addWidget(tak_btn)
        
        nie_btn = ModernButton("❌ Nie, wróć do menu", "secondary")
        nie_btn.clicked.connect(self.show_main_menu)
        self.button_layout.addWidget(nie_btn)
    
    def nowa_standardowa_analiza(self):
        """Rozpocznij nową standardową analizę od początku"""
        # Reset workflow
        self.current_step = 1
        self.liczba_budynkow = 0
        self.rozne_funkcje = False
        
        self.save_checkpoint('nowa_standardowa_analiza')
        
        if not self.sprawdz_warstwy_i_projekt():
            return
        
        self.add_message("Rozpoczynam nową analizę standardową", "success")
        self.add_message("Użyj wtyczki GISsupport do odnalezienia terenu planowanej inwestycji")
        self.show_continue_button("granica_terenu_zapis_wynikowULDK.py", "granica_terenu")
    
    def resume_from_detected_step(self):
        """Wznów workflow z wykrytego kroku"""
        self.add_message(f"Wznawianie workflow z kroku {self.current_step}", "info")
        self.clear_buttons()
        
        # Pokaż przycisk testowy
        # test_btn = ModernButton("Testuj funkcjonalność", "primary")
        # test_btn.clicked.connect(self.test_functionality)
        # self.button_layout.addWidget(test_btn)
        self.continue_workflow()
        
        
    def test_functionality(self):
        """Testowa funkcjonalność"""
        self.add_message("Test funkcjonalności - działa!", "success")
    
    def pokaz_funkcje(self):
        """Pokaż listę dostępnych funkcji z możliwością wywołania"""
        self.clear_buttons()
        self.add_message("Lista dostępnych funkcji:")
        
        # Słownik mapujący opisy na funkcje
        self.funkcje_map = {
            "Zapisz warstwę tymczasową": self.save_memory_layer,
            "Dodaj pola: WIZ, WNIZ, WPZ, WPBC do warstwy": self.add_fields_script,
            "Generuj analizę opisową": self.generator_analiz_opisowych,
            # "Generuj zestawienie dachów": self.zestawienie_dachow,
            # "Podziel budynki według funkcji": self.podziel_budynki,
            # "Sprawdź pokrycie terenu": self.check_land_coverage
        }
        
        funkcje_list = QListWidget()
        
        # Dodaj opisy do listy
        for opis in self.funkcje_map.keys():
            funkcje_list.addItem(opis)
        
        # Obsługa kliknięcia w element listy
        funkcje_list.itemClicked.connect(self.on_funkcja_clicked)
        
        self.button_layout.addWidget(funkcje_list)

    def on_funkcja_clicked(self, item):
        """Obsłuż kliknięcie w funkcję z listy"""
        opis_funkcji = item.text()
        if opis_funkcji in self.funkcje_map:
            funkcja = self.funkcje_map[opis_funkcji]
            try:
                if callable(funkcja):
                    funkcja()
            except Exception as e:
                print(f"Błąd podczas wywoływania funkcji {opis_funkcji}: {e}")
                
    def test_function_1(self):
        """Testowa funkcja 1"""
        self.add_message("Wykonano test funkcji 1", "success")
        
    def test_function_2(self):
        """Testowa funkcja 2"""
        self.add_message("Wykonano test funkcji 2", "success")
    
    def start_workflow(self):
        """Rozpocznij workflow - zawsze od menu głównego"""
        self.show_main_menu()
    
    def sprawdz_warstwy_i_projekt(self):
        """Sprawdź czy wymagane warstwy są wczytane i projekt zapisany"""
        project = QgsProject.instance()
        
        # Sprawdź czy projekt jest zapisany
        if not project.fileName():
            QMessageBox.warning(None, "Błąd", "Projekt musi być najpierw zapisany!")
            return False
        
        self.add_message("Projekt i warstwy sprawdzone - OK", "success")
        return True
    
    def show_continue_button(self, script_name, expected_layer=None, message=None):
        """Pokaż przycisk 'Dalej' do wykonania następnego skryptu"""
        self.clear_buttons()
        
        if message:
            self.add_message(message)
        
        dalej_btn = QPushButton("Dalej")
        dalej_btn.clicked.connect(lambda: self.execute_next_step(script_name, expected_layer))
        self.button_layout.addWidget(dalej_btn)
    
    def execute_next_step(self, script_name, expected_layer=None):
        """Wykonaj kolejny krok workflow"""
        self.add_message(f"Rozpoczynam wykonywanie: {script_name}", "info")
        
        # TUTAJ JEST KLUCZOWA ZMIANA - wywołanie execute_script
        success = self.execute_script(script_name, f'step_{self.current_step + 1}')
        
        if success:
            self.current_step += 1
            self.update_step_display()
            
            # Sprawdź czy oczekiwana warstwa została utworzona
            if expected_layer and not self.check_layer_exists(expected_layer):
                self.add_message(f"OSTRZEŻENIE: Nie znaleziono oczekiwanej warstwy '{expected_layer}'", "warning")
            
            self.add_message(f"Skrypt {script_name} wykonany pomyślnie", "success")
            
            # Przejdź do następnego kroku
            self.continue_workflow()
        else:
            self.add_message(f"Błąd wykonania skryptu {script_name}", "error")
            # Nie przechodź dalej jeśli skrypt się nie wykonał
         
    def continue_workflow(self):
        """Kontynuuj workflow na podstawie aktualnego kroku"""
        self.clear_buttons()
        
        if self.current_step == 1:
            # Krok 1: Granica terenu
            self.add_message("Krok 1: Przejdź do wtyczki GISsupport i znajdź teren inwestycji")
            self.show_continue_button("granica_terenu_zapis_wynikowULDK.py", "granica_terenu", 
                                     "Po znalezieniu terenu kliknij Dalej")
        
        elif self.current_step == 2:
            # Krok 2: Bufor obszaru
            self.add_message("Krok 2: Tworzenie bufora obszaru analizowanego")
            self.show_continue_button("front_dzialki_buffer.py", "granica_obszaru_analizowanego")
            QMessageBox.information(
                None, 
                "Wyznaczanie obszaru analizy", 
                "Proces rozpoczęty!\n\n"
                "Instrukcje:\n"
                "1. Warstwa 'front_dzialki' jest już aktywna i w trybie edycji\n"
                "2. Użyj narzędzia 'Dodaj obiekt liniowy'\n"
                "3. Narysuj linię front działki\n"
                "4. Długość zostanie automatycznie obliczona i zapisana\n"
            )
        
        elif self.current_step == 3:
            # Krok 3: Wymiary
            self.add_message("Krok 3: Rysowanie wymiarów")
            self.show_continue_button("wymiary.py", "wymiary")
            QMessageBox.information(
                None, 
                "Wymiarowanie frontu działki i promienia obszaru analizy", 
                "Proces rozpoczęty!\n\n"
                "Instrukcje:\n"
                "1. Warstwa 'granica_obszaru_analizowanego' jest już aktywna i w trybie edycji\n"
                "2. Użyj narzędzia 'Dodaj obiekt liniowy'\n"
                "3. Narysuj linię wymiarowe\n"
                "4. Długość zostanie automatycznie obliczona i zapisana\n"
                "5. Możesz rysować kolejne linie według potrzeby\n"
                "6. Zakończ edycję gdy skończysz (Ctrl+S lub kliknij 'Zapisz wymiary')"
            )
        
        elif self.current_step == 4:
            # Krok 4: Zapisywanie wymiarów
            self.add_message("Krok 4: Zapisywanie wymiarów")
            self.show_continue_button("zapis_wymiarow.py", None)
        
        elif self.current_step == 5:
            # Krok 5: Działki i budynki
            self.add_message("Krok 5: Wyznaczanie działek i budynków w obszarze")
            self.show_continue_button("wyznacz_dzialki_i_budynki.py", "dzialki_w_obszarze")
        
        elif self.current_step == 6:
            # Krok 6: Elewacje
            self.add_message("Krok 6: Pomiar elewacji frontowych")
            self.show_continue_button("qgis_elewacja_drawing_more_safe.py", "budynki_z_szer_elew_front")
        
        elif self.current_step == 7:
            # Krok 7: Przetwarzanie chmury punktów
            self.add_message("Krok 7: Przetwarzanie chmury punktów")
            utworz_folder(f"{Path(project_directory)}/chmura")
            QMessageBox.information(
                None, 
                "Wymiarowanie frontu działki i promienia obszaru analizy", 
                "Proces rozpoczęty!\n\n"
                "Instrukcje:\n"
                "1. Po dokonaniu pomiarów elewacji frontowych pobierz chmurę punktów dla projektu.\n"
                "2. Ustaw chmurę jako aktywną warstwę przed kliknięciem 'Dalej'\n"
                )
            self.show_continue_button("pointcloud_processing_script.py", "Classification_2")
        
        elif self.current_step == 8:
            # Krok 8: Klasyfikacja PBC
            self.add_message("Krok 8: Klasyfikacja PBC")
            self.execute_pbc_classification()
        
        elif self.current_step == 9:
            # Krok 9: Weryfikacja
            self.add_message("Krok 9: Weryfikacja punktów PBC")
            self.show_verification_step()
        
        elif self.current_step == 10:
            # Krok 10: Wskaźniki
            self.add_message("Krok 10: Obliczanie wskaźników działek")
            self.execute_wskazniki()
        
        elif self.current_step == 11:
            # Krok 11: Parametry budynków
            self.add_message("Krok 11: Obliczanie parametrów budynków")
            self.execute_parametry_budynkow()
        
        elif self.current_step == 12:
            # Krok 12: Linie zabudowy
            self.add_message("Krok 12: Wyznaczanie linii zabudowy")
            self.show_line_measurement_controls()
        
        elif self.current_step == 13:
            # Krok 13: Dane działki
            self.add_message("Krok 13: Dane działki inwestora")
            self.ask_about_data_sheet()
        
        elif self.current_step == 14:
            # Krok 14: Wyniki końcowe
            self.add_message("Krok 14: Generowanie wyników końcowych")
            self.execute_final_scripts()
        
        else:
            # Workflow zakończony
            self.show_completion()    
     
    def save_checkpoint(self, step_name):
        """Zapisz checkpoint workflow"""
        checkpoint_data = {
            'step': self.current_step,
            'step_name': step_name,
            'timestamp': datetime.now().isoformat(),
            'liczba_budynkow': self.liczba_budynkow,
            'rozne_funkcje': self.rozne_funkcje
        }
        
        try:
            with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2)
            self.add_message(f"Zapisano checkpoint: {step_name}")
        except Exception as e:
            self.add_message(f"Błąd zapisu checkpoint: {str(e)}")
    
    def go_back_to_checkpoint(self):
        """Cofnij się do poprzedniego kroku"""
        if self.current_step > 0:
            self.current_step -= 1
            self.add_message(f"Cofnięto do kroku {self.current_step}")
            self.update_step_display()
            if self.current_step == 0:
                self.show_main_menu()
        else:
            self.add_message("Jesteś już na początku workflow")
    
    def load_checkpoint_data(self):
        """Wczytaj tylko dodatkowe dane z checkpoint (nie krok)"""
        if os.path.exists(CHECKPOINT_FILE):
            try:
                with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
                
                self.liczba_budynkow = checkpoint_data.get('liczba_budynkow', 0)
                self.rozne_funkcje = checkpoint_data.get('rozne_funkcje', False)
                
            except Exception as e:
                self.add_message(f"Błąd wczytywania danych checkpoint: {str(e)}")
    
    def reset_workflow(self):
        """Resetuj workflow do początku"""
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
        
        self.current_step = 0
        self.liczba_budynkow = 0
        self.rozne_funkcje = False
        self.update_step_display()
        self.add_message("Workflow został zresetowany")
        self.start_workflow()
               
    def zakoncz_pomiar_linii_step(self):
        """Zakończ pomiar linii i sprawdź czy warstwa powstała"""
        self.zakoncz_pomiar_linii()
        
        if self.check_layer_exists("linie_zabudowy"):
            self.add_message("Linia zabudowy została zapisana")
            self.current_step += 1  # Przejdź do kroku 13
            self.clear_buttons()
            
            # Pokaż przycisk do następnego kroku
            dalej_btn = QPushButton("Dalej")
            dalej_btn.clicked.connect(lambda: self.continue_workflow())
            self.button_layout.addWidget(dalej_btn)
        else:
            self.add_message("BŁĄD: Nie utworzono warstwy 'linie_zabudowy'")
            f"{Path(project_directory)}"
            
            
    def uruchom_pomiar_linii(self):
        """Uruchom pomiar linii - placeholder"""
        # Tu będzie Twoja implementacja
        """Główna funkcja uruchamiająca proces pomiaru linii zabudowy"""
        global line_controller
        
        try:
            print("🔄 Uruchamianie procesu pomiaru linii zabudowy...")
            
            # Utwórz nowy kontroler
            print("🔄 Tworzenie nowego kontrolera...")
            line_controller = LineMeasurementController()
            
            # Rozpocznij proces
            print("🔄 Rozpoczynanie procesu mierzenia...")
            success = line_controller.start_measurement_process()
            
            if success:
                print("✅ Proces uruchomiony pomyślnie")
                # Instrukcje dla użytkownika
                try:
                    QMessageBox.information(
                        None, 
                        "Pomiar linii zabudowy", 
                        "Proces rozpoczęty!\n\n"
                        "Instrukcje:\n"
                        "1. Warstwa 'linie_zabudowy' jest już aktywna i w trybie edycji\n"
                        "2. Użyj narzędzia 'Dodaj obiekt liniowy'\n"
                        "3. Narysuj linię zabudowy\n"
                        "4. Długość zostanie automatycznie obliczona i zapisana w polu 'distance'\n"
                        "5. Możesz rysować kolejne linie według potrzeby\n"
                        "6. Zakończ edycję gdy skończysz (Ctrl+S lub kliknij 'Zapisz zmiany')"
                    )
                except Exception as dialog_error:
                    print(f"⚠️ Problem z dialogiem informacyjnym: {dialog_error}")
            else:
                print("❌ Nie udało się uruchomić procesu")
                try:
                    QMessageBox.warning(None, "Błąd", "Nie udało się uruchomić procesu pomiaru!")
                except Exception as dialog_error:
                    print(f"⚠️ Problem z dialogiem błędu: {dialog_error}")
                
        except Exception as e:
            print(f"❌ Błąd w uruchom_pomiar_linii: {e}")
            import traceback
            traceback.print_exc()
            try:
                QMessageBox.critical(None, "Błąd krytyczny", f"Wystąpił błąd: {str(e)}")
            except:
                print("❌ Nie można wyświetlić dialogu błędu")

        self.add_message("Uruchomiono tryb pomiaru linii")
        pass
    
    def zakoncz_pomiar_linii(self):
        """Zakończ pomiar linii - placeholder"""  
        # Tu będzie Twoja implementacja
        """Funkcja do ręcznego zakończenia procesu pomiaru"""
        global line_controller
        try:
            if line_controller:
                line_controller.finish_measurement()
                line_controller = None
            else:
                print("❌ Kontroler nie jest zainicjalizowany")
        except Exception as e:
            print(f"❌ Błąd w zakoncz_pomiar_linii: {e}")
        self.add_message("Zakończono pomiar linii")
        layers = QgsProject.instance().mapLayersByName("linie_zabudowy")
        if layers:
            layer = layers[0]
        zapis_do_gpkg(layer)
        remove_memory_layers()
        pass        
    
    
    def show_line_measurement_controls(self):
        """Pokaż kontrolki dla pomiaru linii zabudowy"""
        self.clear_buttons()
        
        start_btn = ModernButton("🚀 Rozpocznij pomiar linii", "success")
        start_btn.clicked.connect(self.uruchom_pomiar_linii)
        self.button_layout.addWidget(start_btn)
        
        finish_btn = ModernButton("✅ Zakończ pomiar linii", "primary") 
        finish_btn.clicked.connect(self.zakoncz_pomiar_linii_step)
        self.button_layout.addWidget(finish_btn)
    
        
    def save_memory_layer(self):
        self.execute_script("zapisanie_warstwy_tymczasowej.py")
        
    def add_fields_script(self):
        self.execute_script("add_fields_script.py")
        
    def generator_analiz_opisowych(self):
        self.execute_script("generator_analiz_opisowych.py")
        
    
    def standardowa_analiza(self):
        """Rozpocznij standardową analizę"""
        self.current_step = 1
        self.save_checkpoint('standardowa_analiza')
        
        if not self.sprawdz_warstwy_i_projekt():
            return
        
        self.add_message("Użyj wtyczki GISsupport do odnalezienia terenu planowanej inwestycji")
        self.show_continue_button("granica_terenu_zapis_wynikowULDK.py", "granica_terenu")
    
    
    
    def execute_pbc_classification(self):
        """Wykonaj klasyfikację PBC"""
        if self.execute_script("fixed_qgis_hex_predictor.py") and \
           self.execute_script("fixed_qgis_hex_predictor_teren_inwestycji.py"):
            self.current_step += 1
            self.continue_workflow()
    
    def execute_wskazniki(self):
        """Wykonaj obliczanie wskaźników"""
        if self.execute_script("oblicz_wskazniki_dzialek.py") and \
           self.execute_script("wskazniki_teren_inwestycji.py"):
            self.current_step += 1
            self.continue_workflow()
    
    def execute_parametry_budynkow(self):
        """Wykonaj obliczanie parametrów budynków"""
        if not self.execute_script('oblicz_parametry_budynkow.py'):
            return
        if not self.execute_script('przygotuj_dachy_do_klasyfikacji.py'):
            return  
        if not self.execute_script("roof_classification.py"):
            return
        
        self.ask_building_count()
        
    
    def show_verification_step(self):
        """Pokaż krok weryfikacji punktów"""
        self.clear_buttons()
        self.add_message("Zweryfikuj czy punkty zostały prawidłowo zaklasyfikowane")
        
        dalej_btn = QPushButton("Dalej - punkty OK")
        dalej_btn.clicked.connect(self.verification_ok)
        self.button_layout.addWidget(dalej_btn)
    
    def verification_ok(self):
        """Kontynuuj po weryfikacji"""
        self.current_step += 1
        self.continue_workflow()
    
    def ask_building_count(self):
        """Zapytaj o liczbę budynków"""
        self.clear_buttons()
        count, ok = QInputDialog.getInt(None, "Liczba budynków", 
                                       "Ilu budynków dotyczy wniosek?", 1, 1, 100)
        if ok:
            self.liczba_budynkow = count
            if count > 1:
                self.ask_different_functions()
            else:
                # tworzy folder
                utworz_folder(f"{Path(project_directory)}/budynki_parametry_dachy")
                zestawienie_dachow("budynki_parametry")
                self.current_step += 1
                self.continue_workflow()
    
    def ask_different_functions(self):
        """Zapytaj o różne funkcje budynków"""
        reply = QMessageBox.question(None, "Różne funkcje", 
                                   "Czy w skład planowanej inwestycji wchodzą budynki o różnych funkcjach?",
                                   QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.rozne_funkcje = True
            
            try:
                # Wykonaj podział budynków
                context, feedback = setup_processing()
                output_folder = f"{Path(project_directory)}/budynki_parametry_dachy"
                
                print("Rozpoczynam podział budynków...")
                warstwy_budynkow = podziel_budynki(
                    buildings_layer_name="budynki_parametry", 
                    output_dir=output_folder, 
                    context=context, 
                    feedback=feedback,
                )
                
                print(f"Przetwarzam {len(warstwy_budynkow)} warstw budynków...")
               
                # Przetwarzaj każdą utworzoną warstwę
                for i, warstwa_path in enumerate(warstwy_budynkow):
                    print(f"Przetwarzam warstwę {i+1}/{len(warstwy_budynkow)}: {warstwa_path}")
                    
                    # Załaduj warstwę do projektu jeśli nie jest załadowana
                    layer = None
                    layer_name = Path(warstwa_path).stem  # Nazwa pliku bez rozszerzenia
                    
                    # Sprawdź czy warstwa już istnieje w projekcie
                    existing_layers = QgsProject.instance().mapLayersByName(layer_name)
                    if existing_layers:
                        layer = existing_layers[0]
                        print(f"Używam istniejącej warstwy: {layer_name}")
                    else:
                        # Załaduj warstwę z pliku
                        layer = QgsVectorLayer(warstwa_path, layer_name, "ogr")
                        if layer.isValid():
                            QgsProject.instance().addMapLayer(layer)
                            print(f"Załadowano nową warstwę: {layer_name}")
                        else:
                            print(f"Błąd: Nie można załadować warstwy z pliku: {warstwa_path}")
                            continue
                    
                    # Wykonaj zestawienie dla tej warstwy
                    if layer and layer.isValid():
                        try:
                            zestawienie_dachow(layer_name)  # Przekaż nazwę warstwy
                            print(f"✓ Zestawienie dla warstwy {layer_name} zakończone")
                        except Exception as e:
                            print(f"Błąd podczas tworzenia zestawienia dla warstwy {layer_name}: {str(e)}")
                    
                print("✓ Wszystkie zestawienia zakończone")
                zmien_nazwy_plikow(output_folder, mapa_nazw)

            except Exception as e:
                print(f"Błąd podczas przetwarzania budynków: {str(e)}")
                QMessageBox.critical(None, "Błąd", f"Wystąpił błąd podczas przetwarzania budynków:\n{str(e)}")
        else:
            self.rozne_funkcje = False
            print("Użytkownik wybrał, że budynki mają tę samą funkcję")
        
        self.current_step += 1
        self.continue_workflow()
        
        
    def ask_about_data_sheet(self):
        """Zapytaj o arkusz danych działki"""
        reply = QMessageBox.question(None, "Arkusz danych", 
                                   "Czy wypełniłeś arkusz 'dane_dzialki_inwestora'?",
                                   QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.current_step += 1
            self.continue_workflow()
        else:
            self.add_message("Wypełnij arkusz 'dane_dzialki_inwestora' przed kontynuowaniem")
    
    def execute_final_scripts(self):
        """Wykonaj końcowe skrypty"""
        if self.execute_script("output_and_results_multi.py") and \
           self.execute_script("generator_analiz_opisowych.py"):
            self.monitor_final_files()
    
    def monitor_final_files(self):
        """Monitoruj tworzenie końcowych plików"""
        project_dir = os.path.dirname(QgsProject.instance().fileName())
        
        # Sprawdź plik docx z analizą
        docx_found = False
        pdf_found = False
        
        for file in os.listdir(project_dir):
            if "analiza" in file.lower() and file.endswith('.docx'):
                docx_found = True
                break
        
        if docx_found:
            self.add_message("Nie zapomnij wyeksportować tabeli do PDF oraz wykonać załącznik graficzny")
            
            # Sprawdź plik PDF
            for file in os.listdir(project_dir):
                if "analiza graficzna" in file.lower() and file.endswith('.pdf'):
                    pdf_found = True
                    break
            
            if pdf_found:
                self.show_completion()
    
    def show_completion(self):
        """Pokaż komunikat zakończenia z obrazkiem"""
        self.clear_buttons()
        self.add_message("Dziękuję! Analiza została zakończona pomyślnie! 😊")
        
        # Tu można dodać obrazek Sophie Tatcher lub Scarlett Johansson
        # Ale ze względów praktycznych zostawiam tylko emoji
        completion_label = QLabel("🎉 WORKFLOW ZAKOŃCZONY POMYŚLNIE! 🎉")
        completion_label.setAlignment(Qt.AlignCenter)
        completion_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.button_layout.addWidget(completion_label)
    
    def focus_on_layer(self, layer_name):
        """Skieruj widok mapy na określoną warstwę"""
        try:
            layers = QgsProject.instance().mapLayersByName(layer_name)
            if layers:
                layer = layers[0]
                iface.mapCanvas().setExtent(layer.extent())
                iface.mapCanvas().refresh()
        except Exception as e:
            self.add_message(f"Nie można skierować widoku na warstwę {layer_name}: {str(e)}")
    
    def execute_script(self, script_name, step_name=None):
        """Wykonaj skrypt z obsługą błędów"""
        script_path = os.path.join(SCRIPTS_PATH, script_name)
        
        if not os.path.exists(script_path):
            self.handle_script_error(f"Nie znaleziono skryptu: {script_name}", step_name)
            return False
        
        try:
            self.add_message(f"Wykonuję skrypt: {script_name}")
            with open(script_path, 'r', encoding='utf-8') as f:
                script_content = f.read()
            
            exec(script_content, globals())
            
            # Zapisz checkpoint po pomyślnym wykonaniu
            if step_name:
                self.save_checkpoint(step_name)
            
            self.add_message(f"Skrypt {script_name} wykonany pomyślnie")
            return True
            
        except Exception as e:
            self.handle_script_error(f"Błąd w skrypcie {script_name}:\n{str(e)}", step_name)
            return False
    
    def handle_script_error(self, error_message, step_name=None):
        """Obsłuż błąd skryptu"""
        self.add_message(f"BŁĄD: {error_message}")
        QMessageBox.critical(None, "Błąd skryptu", error_message)
    
    def check_layer_exists(self, layer_name):
        """Sprawdź czy warstwa istnieje w projekcie"""
        layers = QgsProject.instance().mapLayersByName(layer_name)
        existing_layers = [layer.name() for layer in QgsProject.instance().mapLayers().values()]
        print(f"DEBUG: Szukam warstwy '{layer_name}'")
        print(f"DEBUG: Dostępne warstwy: {existing_layers}")
        return len(layers) > 0
    
   


def create_wz_workflow_dock():
    """Funkcja do utworzenia i wyświetlenia dock widget"""
    try:
        if not IFACE_AVAILABLE:
            raise Exception("iface nie jest dostępne - nie można utworzyć dock widget")
        
        dock_widget = WZWorkflowDockWidget()
        iface.addDockWidget(Qt.RightDockWidgetArea, dock_widget)
        dock_widget.show()
        
        print("WZ Workflow dock widget utworzony pomyślnie")
        return dock_widget
        
    except Exception as e:
        print(f"BŁĄD tworzenia dock widget: {e}")
        import traceback
        traceback.print_exc()
        raise
