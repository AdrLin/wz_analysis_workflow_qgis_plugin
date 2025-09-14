#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 11:50:34 2025

@author: adrian
"""

import pandas as pd
from qgis.core import QgsProject
import os

project_path = QgsProject.instance().fileName()
project_directory = os.path.dirname(project_path)


def lista_csv(folder):
    try:
        # Pobiera tylko pliki .csv
        pliki_csv = [
            plik for plik in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, plik)) and plik.lower().endswith('.csv')
        ]
        return pliki_csv
    except FileNotFoundError:
        print("Podany folder nie istnieje.")
        return []
    except Exception as e:
        print(f"Wystąpił błąd: {e}")
        return []


def process_layers():
    """
    Główna funkcja przetwarzająca warstwy QGIS i eksportująca wyniki do Excel
    """
    
    # 1. Pobieranie warstw z projektu QGIS
    project = QgsProject.instance()
    
    # Pobranie warstw
    dzialki_layer = None
    budynki_layer = None
    
    for layer in project.mapLayers().values():
        if layer.name() == 'dzialki_ze_wskaznikami':
            dzialki_layer = layer
        elif layer.name() == 'budynki_parametry':
            budynki_layer = layer
    
    if dzialki_layer is None:
        print("Błąd: Nie znaleziono warstwy 'dzialki_ze_wskaznikami'")
        return
    
    if budynki_layer is None:
        print("Błąd: Nie znaleziono warstwy 'budynki_parametry'")
        return
    
    print("Znaleziono obie warstwy")
    
    # Konwersja warstw QGIS na DataFrame pandas
    dzialki_df = layer_to_dataframe(dzialki_layer)
    budynki_df = layer_to_dataframe(budynki_layer)
    budynki_df['nachylenie'] = pd.to_numeric(budynki_df['nachylenie'], errors='coerce').fillna(0).astype(int)

    
    print(f"Liczba działek: {len(dzialki_df)}")
    print(f"Liczba budynków: {len(budynki_df)}")
    
    # 2. Pivot tabeli budynków
    budynki_pivot = create_budynki_pivot(budynki_df)
    print(f"Liczba pogrupowanych działek z budynkami: {len(budynki_pivot)}")
    
    # 3. Łączenie tabel
    merged_df = merge_tables(dzialki_df, budynki_pivot)
    print(f"Liczba rekordów po połączeniu: {len(merged_df)}")
    
    # 4. Mapowanie pól według schematu
    output_table = create_output_table(merged_df)
    
    # 5. Zapis do Excel
    output_path = os.path.join(QgsProject.instance().homePath(), "output_table.xlsx")
    save_to_excel(output_table, output_path)
    output_path2 = "/home/adrian/Documents/JXPROJEKT/analiza_schemat/output_table.xlsx"
    save_to_excel(output_table, output_path2)
    print(f"Tabela została zapisana w: {output_path}")

def layer_to_dataframe(layer):
    """
    Konwertuje warstwę QGIS na DataFrame pandas
    """
    data = []
    fields = [field.name() for field in layer.fields()]
    
    for feature in layer.getFeatures():
        row = {}
        for field in fields:
            row[field] = feature[field]
        data.append(row)
    
    return pd.DataFrame(data)

def create_budynki_pivot(budynki_df):
    """
    Tworzy pivot z tabeli budynków według ID_DZIALKI
    """
    if 'ID_DZIALKI' not in budynki_df.columns:
        print("Błąd: Brak kolumny ID_DZIALKI w tabeli budynków")
        return pd.DataFrame()
    
    print("Kolumny w tabeli budynków:", budynki_df.columns.tolist())
    print("Typy danych:", budynki_df.dtypes)
    
    # Funkcja do bezpiecznego uśredniania
    def safe_mean(x):
        # Konwertuj do numerycznych, ignoruj błędy
        numeric_vals = pd.to_numeric(x, errors='coerce')
        # Usuń NaN i oblicz średnią
        valid_vals = numeric_vals.dropna()
        return valid_vals.mean() if len(valid_vals) > 0 else None
    
    # Funkcja do łączenia tekstów
    def safe_join(x):
        # Usuń wartości NULL i puste, konwertuj do string
        valid_vals = [str(val) for val in x if pd.notna(val) and str(val).strip() != '']
        return '; '.join(valid_vals) if valid_vals else None
    
    # Przygotowanie słownika agregacji - sprawdź które kolumny istnieją
    agg_dict = {}
    
    if 'szer_elew_front' in budynki_df.columns:
        agg_dict['szer_elew_front'] = safe_mean
    if 'wysokosc' in budynki_df.columns:
        agg_dict['wysokosc'] = safe_mean
    if 'nachylenie' in budynki_df.columns:
        agg_dict['nachylenie'] = safe_join
    if 'Kategoria' in budynki_df.columns:
        agg_dict['Kategoria'] = safe_join
    
    if not agg_dict:
        print("Błąd: Nie znaleziono żadnej z wymaganych kolumn w tabeli budynków")
        return pd.DataFrame()
    
    # Grupowanie według ID_DZIALKI
    grouped = budynki_df.groupby('ID_DZIALKI').agg(agg_dict).reset_index()
    
    return grouped

def merge_tables(dzialki_df, budynki_pivot):
    """
    Łączy tabele działek z pivot budynków
    """
    # Łączenie po ID_DZIALKI
    merged = pd.merge(dzialki_df, budynki_pivot, on='ID_DZIALKI', how='left')
    
    return merged


def create_output_table(qgis_layer_df):
    """
    Tworzy tabelę wyjściową z mapowaniem pól według schematu
    """
    output_table = pd.DataFrame()
    
    # Mapowanie pól zgodnie ze schematem
    output_table['Lp.'] = pd.to_numeric(
        qgis_layer_df.get('Lp.', ''), errors='coerce'
    )
    output_table['nr działki'] = qgis_layer_df.get('NUMER_DZIALKI', '')
    output_table['nr obrębu'] = qgis_layer_df.get('NUMER_OBREBU', '')
    output_table['powierzchnia działki [m2]'] = pd.to_numeric(qgis_layer_df.get('POLE_EWIDENCYJNE', ''), errors='coerce').round(2)
    output_table['rodzaj zabudowy'] = qgis_layer_df.get('RODZAJ_ZABUDOWY', '')
    output_table['szerokość elewacji frontowej [m]'] = round(qgis_layer_df.get('szer_elew_front', ''),2)
    output_table['wysokość zabudowy [m]'] = round(qgis_layer_df.get('wysokosc', ''),2)
    output_table['rodzaj dachu'] = qgis_layer_df.get('Kategoria', '')
    output_table['kąt nachylenia połaci dachowych [o]'] = qgis_layer_df.get('nachylenie', '')
    output_table['powierzchnia zabudowy [m2]'] = pd.to_numeric(qgis_layer_df.get('S_POW_ZABUD', ''), errors='coerce').round(2)
    
    # Konwersja do numerycznych z obsługą błędów
    output_table['WIZ wskaźnik intensywności zabudowy'] = pd.to_numeric(
        qgis_layer_df.get('WIZ', ''), errors='coerce'
    )
    output_table['WNIZ wskaźnik nadziemnej intensywności zabudowy'] = pd.to_numeric(
        qgis_layer_df.get('WNIZ', ''), errors='coerce'
    )
    
    output_table['WPZ wskaźnik powierzchni zabudowy'] = qgis_layer_df.get('WPZ', '')
    output_table['WPBC wskaźnik powierzchni biologicznie czynnej'] = qgis_layer_df.get('WPBC', '')
    output_table['id_działki'] = qgis_layer_df.get('ID_DZIALKI', '')
    
    output_table['wpz_float'] = pd.to_numeric(
        qgis_layer_df.get('wpz_float', ''), errors='coerce'
    )
    output_table['wpbc_float'] = pd.to_numeric(
        qgis_layer_df.get('wpbc_float', ''), errors='coerce'
    )
    
    
    # Sortowanie według Lp.
    output_table = output_table.sort_values(by="Lp.")
    
    return output_table

def save_to_excel(df, output_path):
    """
    Zapisuje DataFrame do pliku Excel
    """
    try:
        # Zapisanie z określonymi parametrami
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Output', index=False)
        
        print(f"Plik Excel został zapisany pomyślnie: {output_path}")
        
    except Exception as e:
        print(f"Błąd podczas zapisywania pliku Excel: {str(e)}")




def create_results_analysis():
    """
    Tworzy tabelę wyników analizy na podstawie output_table.xlsx
    """
    
    # Wczytanie output_table.xlsx
    project_path = QgsProject.instance().homePath()
    input_path = os.path.join(project_path, "output_table.xlsx")

    if not os.path.exists(input_path):
        print(f"Błąd: Nie znaleziono pliku {input_path}")
        return
    
    try:
        output_table = pd.read_excel(input_path)
        print(f"Wczytano tabelę: {len(output_table)} rekordów")
        print("Dostępne kolumny:", output_table.columns.tolist())
    except Exception as e:
        print(f"Błąd podczas wczytywania pliku: {str(e)}")
        return
    
    # Tworzenie tabeli wyników
    results_df = create_results_table(output_table)
    
    # Pobieranie dodatkowych parametrów z warstw QGIS
    additional_params = get_additional_parameters()
    
    # Dodanie dodatkowych parametrów do results_df
    for param_name, param_value in additional_params.items():
        new_row = pd.DataFrame([{'nazwa_pola': param_name, 'wartosc': param_value}])
        results_df = pd.concat([results_df, new_row], ignore_index=True)
    
    # utworz geometrie dachow
    geometrie = []
    folder_dachy = f"{project_directory}/budynki_parametry_dachy"
    dachy_files = lista_csv(folder_dachy)
    for i in range(len(dachy_files)):
        geometriaDachow = utworz_geometria_dachow(dachy_files[i])
        geometrie.append(geometriaDachow)
        
        # Tworzenie DataFrame
    geometrie_df = pd.DataFrame([
        {'nazwa_pola': f"geometriaDachow_{i}", 'wartosc': geometrie[i]}
        for i in range(len(geometrie))
    ])
   
    results_df = pd.concat([results_df, geometrie_df], ignore_index=True)

    


    # Zapis do pliku Excel
    output_path = os.path.join(project_path, "results_df.xlsx")
    save_results_table(results_df, output_path)
    output_path2 = os.path.join("/home/adrian/Documents/JXPROJEKT/analiza_schemat/results_df.xlsx")
    save_results_table(results_df, output_path2)
    
    print(f"Tabela wyników została zapisana w: {output_path}")


def get_additional_parameters():
    """
    Pobiera dodatkowe parametry z warstw QGIS
    """
    project = QgsProject.instance()
    additional_params = {}
    
    # Pobieranie nr_obrebu i nr_dzialki z warstwy 'granica_terenu'
    granica_terenu_layer = None
    for layer in project.mapLayers().values():
        if layer.name() == 'granica_terenu':
            granica_terenu_layer = layer
            break
    
    if granica_terenu_layer:
        try:
            for feature in granica_terenu_layer.getFeatures():
                id_dzialki = feature['ID_DZIALKI']
                if id_dzialki:
                    # Parsowanie ID_DZIALKI (format: 301701_1.0169.72/6)
                    parts = str(id_dzialki).split('.')
                    if len(parts) >= 3:
                        # nr_obrebu - między pierwszą a drugą kropką
                        nr_obrebu = parts[1]
                        # nr_dzialki - za drugą kropką
                        nr_dzialki = parts[2]
                        
                        additional_params['nr_obrebu'] = nr_obrebu
                        additional_params['nr_dzialki'] = nr_dzialki
                        break  # Bierzemy pierwszy rekord
        except Exception as e:
            print(f"Błąd podczas pobierania danych z warstwy granica_terenu: {str(e)}")
            additional_params['nr_obrebu'] = ""
            additional_params['nr_dzialki'] = ""
    else:
        print("Nie znaleziono warstwy 'granica_terenu'")
        additional_params['nr_obrebu'] = ""
        additional_params['nr_dzialki'] = ""
    
    # Pobieranie dlugosc_frontu i promien_bufora z warstwy 'wymiary'
    wymiary_layer = None
    for layer in project.mapLayers().values():
        if layer.name() == 'wymiary':
            wymiary_layer = layer
            break
    
    if wymiary_layer:
        try:
            l_values = []
            for feature in wymiary_layer.getFeatures():
                l_value = feature['l']
                if l_value is not None:
                    try:
                        l_values.append(float(l_value))
                    except (ValueError, TypeError):
                        continue
            
            if len(l_values) >= 2:
                l_values.sort()  # Sortujemy aby mieć pewność co do min/max
                additional_params['dlugosc_frontu'] = f"{l_values[0]} m"  # mniejsza liczba
                additional_params['promien_bufora'] = f"{l_values[-1]} m"  # większa liczba
            elif len(l_values) == 1:
                additional_params['dlugosc_frontu'] = f"{l_values[0]} m"
                additional_params['promien_bufora'] = f"{l_values[0]} m"
            else:
                additional_params['dlugosc_frontu'] = "0 m"
                additional_params['promien_bufora'] = "0 m"
                
        except Exception as e:
            print(f"Błąd podczas pobierania danych z warstwy wymiary: {str(e)}")
            additional_params['dlugosc_frontu'] = "0 m"
            additional_params['promien_bufora'] = "0 m"
    else:
        print("Nie znaleziono warstwy 'wymiary'")
        additional_params['dlugosc_frontu'] = "0 m"
        additional_params['promien_bufora'] = "0 m"
    
    # Pobieranie Lz_min i Lz_max z warstwy 'linie_zabudowy'
    linie_zabudowy_layer = None
    for layer in project.mapLayers().values():
        if layer.name() == 'linie_zabudowy':
            linie_zabudowy_layer = layer
            break
    
    if linie_zabudowy_layer:
        try:
            distance_values = []
            for feature in linie_zabudowy_layer.getFeatures():
                distance_value = feature['distance']
                if distance_value is not None:
                    try:
                        distance_values.append(float(distance_value))
                    except (ValueError, TypeError):
                        continue
            
            if len(distance_values) > 0:
                additional_params['Lz_min'] = f"{min(distance_values)} m"
                additional_params['Lz_max'] = f"{max(distance_values)} m"
            else:
                additional_params['Lz_min'] = "0 m"
                additional_params['Lz_max'] = "0 m"
                
        except Exception as e:
            print(f"Błąd podczas pobierania danych z warstwy linie_zabudowy: {str(e)}")
            additional_params['Lz_min'] = "0 m"
            additional_params['Lz_max'] = "0 m"
    else:
        print("Nie znaleziono warstwy 'linie_zabudowy'")
        additional_params['Lz_min'] = "0 m"
        additional_params['Lz_max'] = "0 m"
    
    return additional_params

def create_results_table(output_table):
    """
    Tworzy tabelę wyników na podstawie danych z output_table
    """
    
    # Lista wszystkich zmiennych
    nazwa_pola = [
        'maks_inten_zab',
        'maks_nadz_inten_zab', 
        'min_nadz_inten_zab',
        'Sredni_wiz',
        'Sredni_wniz',
        'Sredni_wpz',
        'Sredni_wpbc',
        'wpz_min',
        'wpz_max',
        'SrElewFront_x',
        'srWysZab_x',
        'SrElewFront_y',
        'srWysZab_y',
        'linia_zabudowy_min',
        'linia_zabudowy_max',
        'wys_zab_min_x',
        'wys_zab_max_x',
        'wys_zab_min_y',
        'wys_zab_max_y',
        'MinXszerElewFront',
        'MaxXszerElewFront',
        'szerElewFront08x',
        'szerElewFront12x',
        'MinYszerElewFront',
        'MaxYszerElewFront',
        'szerElewFront08y',
        'szerElewFront12y',
        'wpbc_min',
        'wpbc_max',
    ]
    
    # Przygotowanie pomocniczych zmiennych
    wpz_float = prepare_wpz_float(output_table)
    wpbc_float = prepare_wpbc_float(output_table)
    # Obliczanie wartości
    wartosc = []
    
    for pole in nazwa_pola:
        try:
            if pole == 'maks_inten_zab':
                val = calculate_safe_mean(output_table, "WIZ wskaźnik intensywności zabudowy", multiplier=1.2)
                wartosc.append(val)
                
            elif pole == 'maks_nadz_inten_zab':
                val = calculate_safe_mean(output_table, "WNIZ wskaźnik nadziemnej intensywności zabudowy", multiplier=1.2)
                wartosc.append(val)
                
            elif pole == 'min_nadz_inten_zab':
                val = calculate_safe_min(output_table, "WNIZ wskaźnik nadziemnej intensywności zabudowy")
                wartosc.append(val)
                
            elif pole == 'Sredni_wiz':
                val = calculate_safe_mean(output_table, "WIZ wskaźnik intensywności zabudowy")
                wartosc.append(val)
                
            elif pole == 'Sredni_wniz':
                val = calculate_safe_mean(output_table, "WNIZ wskaźnik nadziemnej intensywności zabudowy")
                wartosc.append(val)
                
            elif pole == 'Sredni_wpz':
                if wpz_float:
                    val = f"{round(sum(wpz_float)/len(wpz_float)*100, 0):.0f}%"
                else:
                    val = "0%"
                wartosc.append(val)
                
            elif pole == 'Sredni_wpbc':
                if wpbc_float:
                    val = f"{round(sum(wpbc_float)/len(wpbc_float)*100, 0):.0f}%"
                else:
                    val = "0%"
                wartosc.append(val)
                
            elif pole == 'wpz_min':
                if wpz_float:
                    val = f"{round(min(wpz_float)*100, 0):.0f}%"
                else:
                    val = "0%"
                wartosc.append(val)
                
            elif pole == 'wpz_max':
                if wpz_float:
                    val = f"{round(max(wpz_float)*100, 0):.0f}%"
                else:
                    val = "0%"
                wartosc.append(val)
                
            elif pole == 'wpbc_min':
                if wpbc_float:
                    val = f"{round(min(wpbc_float)*100, 0):.0f}%"
                else:
                    val = "0%"
                wartosc.append(val)
                
            elif pole == 'wpbc_max':
                if wpbc_float:
                    val = f"{round(max(wpbc_float)*100, 0):.0f}%"
                else:
                    val = "0%"
                wartosc.append(val)
                
            # Parametry budynków X (podstawowa warstwa)
            elif pole == 'SrElewFront_x':
                val = calculate_safe_mean(output_table, "szerokość elewacji frontowej [m]")
                wartosc.append(f"{val} m" if val != 0 else "0 m")
                
            elif pole == 'srWysZab_x':
                val = calculate_safe_mean(output_table, "wysokość zabudowy [m]")
                wartosc.append(f"{val} m" if val != 0 else "0 m")
                
            elif pole == 'wys_zab_min_x':
                val = calculate_safe_min(output_table, "wysokość zabudowy [m]")
                wartosc.append(f"{val} m" if val != 0 else "0 m")
                
            elif pole == 'wys_zab_max_x':
                val = calculate_safe_max(output_table, "wysokość zabudowy [m]")
                wartosc.append(f"{val} m" if val != 0 else "0 m")
                
            elif pole == 'MinXszerElewFront':
                val = calculate_safe_min(output_table, "szerokość elewacji frontowej [m]")
                wartosc.append(f"{val} m" if val != 0 else "0 m")
                
            elif pole == 'MaxXszerElewFront':
                val = calculate_safe_max(output_table, "szerokość elewacji frontowej [m]")
                wartosc.append(f"{val} m" if val != 0 else "0 m")
                
            elif pole == 'szerElewFront08x':
                base_val = calculate_safe_mean(output_table, "szerokość elewacji frontowej [m]")
                val = round(base_val * 0.8, 2) if base_val != 0 else 0
                wartosc.append(f"{val} m")
                
            elif pole == 'szerElewFront12x':
                base_val = calculate_safe_mean(output_table, "szerokość elewacji frontowej [m]")
                val = round(base_val * 1.2, 2) if base_val != 0 else 0
                wartosc.append(f"{val} m")
                
            # Parametry budynków Y (warstwa z suffiksem 1)
            elif pole == 'SrElewFront_y':
                val = calculate_safe_mean(output_table, "szerokość elewacji frontowej [m]1")
                wartosc.append(f"{val} m" if val != 0 else "0 m")
                
            elif pole == 'srWysZab_y':
                val = calculate_safe_mean(output_table, "wysokość zabudowy [m]1")
                wartosc.append(f"{val} m" if val != 0 else "0 m")
                
            elif pole == 'wys_zab_min_y':
                val = calculate_safe_min(output_table, "wysokość zabudowy [m]1")
                wartosc.append(f"{val} m" if val != 0 else "0 m")
                
            elif pole == 'wys_zab_max_y':
                val = calculate_safe_max(output_table, "wysokość zabudowy [m]1")
                wartosc.append(f"{val} m" if val != 0 else "0 m")
                
            elif pole == 'MinYszerElewFront':
                val = calculate_safe_min(output_table, "szerokość elewacji frontowej [m]1")
                wartosc.append(f"{val} m" if val != 0 else "0 m")
                
            elif pole == 'MaxYszerElewFront':
                val = calculate_safe_max(output_table, "szerokość elewacji frontowej [m]1")
                wartosc.append(f"{val} m" if val != 0 else "0 m")
                
            elif pole == 'szerElewFront08y':
                base_val = calculate_safe_mean(output_table, "szerokość elewacji frontowej [m]1")
                val = round(base_val * 0.8, 2) if base_val != 0 else 0
                wartosc.append(f"{val} m")
                
            elif pole == 'szerElewFront12y':
                base_val = calculate_safe_mean(output_table, "szerokość elewacji frontowej [m]1")
                val = round(base_val * 1.2, 2) if base_val != 0 else 0
                wartosc.append(f"{val} m")
                
            # Linia zabudowy (jeśli istnieje kolumna)
            elif pole == 'linia_zabudowy_min':
                val = calculate_safe_min(output_table, "linia zabudowy")
                wartosc.append(f"{val} m" if val != 0 else "0 m")
                
            elif pole == 'linia_zabudowy_max':
                val = calculate_safe_max(output_table, "linia zabudowy")
                wartosc.append(f"{val} m" if val != 0 else "0 m")
                
            else:
                # Dla nieznanych pól
                wartosc.append("")
                
        except Exception as e:
            print(f"Błąd przy obliczaniu {pole}: {str(e)}")
            wartosc.append("")
    
    # Tworzenie DataFrame wyników
    results_df = pd.DataFrame({
        'nazwa_pola': nazwa_pola,
        'wartosc': wartosc
    })
    
    return results_df

def prepare_wpz_float(output_table):
    """Przygotowuje listę wartości wpz_float"""
    try:
        if 'wpz_float' in output_table.columns:
            wpz_values = pd.to_numeric(output_table['wpz_float'], errors='coerce').dropna()
            return wpz_values.tolist()
        else:
            return []
    except:
        return []

def prepare_wpbc_float(output_table):
    """Przygotowuje listę wartości wpbc_float"""
    try:
        if 'wpbc_float' in output_table.columns:
            wpbc_values = pd.to_numeric(output_table['wpbc_float'], errors='coerce').dropna()
            return wpbc_values.tolist()
        else:
            return []
    except:
        return []

def calculate_safe_mean(df, column_name, multiplier=1):
    """Bezpieczne obliczanie średniej"""
    try:
        if column_name in df.columns:
            values = pd.to_numeric(df[column_name], errors='coerce').dropna()
            if len(values) > 0:
                return round(values.mean() * multiplier, 2)
        return 0
    except:
        return 0

def calculate_safe_min(df, column_name):
    """Bezpieczne obliczanie minimum"""
    try:
        if column_name in df.columns:
            values = pd.to_numeric(df[column_name], errors='coerce').dropna()
            if len(values) > 0:
                return round(values.min(), 2)
        return 0
    except:
        return 0

def calculate_safe_max(df, column_name):
    """Bezpieczne obliczanie maksimum"""
    try:
        if column_name in df.columns:
            values = pd.to_numeric(df[column_name], errors='coerce').dropna()
            if len(values) > 0:
                return round(values.max(), 2)
        return 0
    except:
        return 0

def save_results_table(df, output_path):
    """Zapisuje tabelę wyników do pliku Excel"""
    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Results', index=False)
        
        print("Tabela wyników została zapisana pomyślnie")
        print(f"Liczba parametrów: {len(df)}")
        
        # Wyświetl próbkę wyników
        print("\nPróbka wyników:")
        for i in range(min(10, len(df))):
            print(f"{df.iloc[i]['nazwa_pola']}: {df.iloc[i]['wartosc']}")
        
    except Exception as e:
        print(f"Błąd podczas zapisywania pliku Excel: {str(e)}")


def utworz_geometria_dachow(dachy_csv_path):
    """
    Funkcja tworzy zmienną geometriaDachow na podstawie pliku 'dachy.csv'
    """
    # Wczytanie pliku CSV
    dachy = pd.read_csv(dachy_csv_path)
    dachy_layer_name = dachy_csv_path.split("_")[0]
    # Obliczenie łącznej liczby budynków
    total_budynki = dachy['liczba_wystapien'].sum()
    
    # Lista fragmentów opisu
    opisy = [f"Dla budynków zaklasyfikowanych jako {dachy_layer_name}"]
    
    for index, row in dachy.iterrows():
        kategoria = row['Kategoria']
        liczba = row['liczba_wystapien']
        min_nach = row['min_nachylenie']
        max_nach = row['max_nachylenie']
        
        # Obliczenie procentu
        procent = (liczba / total_budynki) * 100
        
        # Określenie przedrostka w zależności od częstości występowania
        if procent > 50:
            przedrostek = "występują przeważnie dachy typu"
        elif procent < 10:
            przedrostek = "sporadycznie występują dachy typu"
        else:
            przedrostek = "występują dachy typu"
        
        # Tworzenie opisu dla kategorii
        if int(min_nach) == int(max_nach): 
            opis = f"{przedrostek} {kategoria} o kącie nachylenia połaci dachowych ok. {min_nach} stopni"
        else:
            opis = f"{przedrostek} {kategoria} o kącie nachylenia połaci dachowych od {min_nach} do {max_nach} stopni"
        opisy.append(opis)
    
    # Łączenie wszystkich opisów
    geometriaDachow = ", ".join(opisy) + ","
    return geometriaDachow


# GENERUJE TABELE WYJSCIOWA
try:
    process_layers()
except Exception as e:
    print(f"Błąd podczas wykonywania skryptu: {str(e)}")
    

# GENERUJE WYNIKI
try:
    create_results_analysis()
except Exception as e:
    print(f"Błąd podczas wykonywania skryptu: {str(e)}")







 