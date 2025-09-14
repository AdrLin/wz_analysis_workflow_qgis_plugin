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
    Obsługuje wiele warstw budynków z różnymi funkcjami
    """
    
    # 1. Pobieranie warstw z projektu QGIS
    project = QgsProject.instance()
    
    # Pobranie warstwy działek
    dzialki_layer = None
    all_budynki_layers = {}  # Wszystkie warstwy budynków
    base_layer = None  # Warstwa podstawowa (bez suffixu)
    suffix_layers = {}  # Warstwy z suffixami
    
    for layer in project.mapLayers().values():
        if layer.name() == 'dzialki_ze_wskaznikami':
            dzialki_layer = layer
        elif layer.name().startswith('budynki_parametry'):
            all_budynki_layers[layer.name()] = layer
            
            # Rozróżnij warstwę podstawową od warstw z suffixami
            if layer.name() == 'budynki_parametry':
                base_layer = layer
            else:
                suffix_layers[layer.name()] = layer
    
    if dzialki_layer is None:
        print("Błąd: Nie znaleziono warstwy 'dzialki_ze_wskaznikami'")
        return
    
    if not all_budynki_layers:
        print("Błąd: Nie znaleziono żadnej warstwy budynków (budynki_parametry*)")
        return
    
    # LOGIKA WYBORU WARSTW: 
    # Jeśli istnieją warstwy z suffixami - używaj TYLKO ich
    # Jeśli nie ma warstw z suffixami - używaj warstwy podstawowej
    if suffix_layers:
        budynki_layers = suffix_layers
        print("Znaleziono warstwy z suffixami - używam tylko ich (pomijam warstwę podstawową)")
        print(f"Warstwy do przetworzenia: {list(budynki_layers.keys())}")
        if base_layer:
            print(f"Pomijam warstwę podstawową: {base_layer.name()}")
    else:
        if base_layer:
            budynki_layers = {'budynki_parametry': base_layer}
            print("Nie znaleziono warstw z suffixami - używam warstwy podstawowej")
        else:
            print("Błąd: Nie znaleziono ani warstwy podstawowej ani warstw z suffixami")
            return
    
    print(f"Znaleziono warstwę działek: {dzialki_layer.name()}")
    print(f"Znaleziono warstwy budynków: {list(budynki_layers.keys())}")
    
    # Konwersja warstwy działek na DataFrame
    dzialki_df = layer_to_dataframe(dzialki_layer)
    print(f"Liczba działek: {len(dzialki_df)}")
    
    # Przetwarzanie każdej warstwy budynków
    all_budynki_pivots = []
    
    for layer_name, layer in budynki_layers.items():
        print(f"\nPrzetwarzanie warstwy: {layer_name}")
        
        # Konwersja warstwy na DataFrame
        budynki_df = layer_to_dataframe(layer)
        budynki_df['nachylenie'] = pd.to_numeric(budynki_df['nachylenie'], errors='coerce').fillna(0).astype(int)
        
        print(f"Liczba budynków w {layer_name}: {len(budynki_df)}")
        
        # Określenie suffixu na podstawie nazwy warstwy
        if layer_name == 'budynki_parametry':
            # Warstwa podstawowa - używana tylko gdy brak warstw z suffixami
            suffix = ""  
        else:
            # Wyciągnij suffix z nazwy warstwy (np. budynki_parametry_1 -> "1")
            parts = layer_name.split('_')
            if len(parts) > 2:
                suffix = '_' + '_'.join(parts[2:])  # Zachowaj pełny suffix z podkreślnikami
            else:
                suffix = "_1"  # Domyślny suffix jeśli nie można wyciągnąć
        
        # Pivot tabeli budynków z odpowiednim suffixem
        budynki_pivot = create_budynki_pivot_with_suffix(budynki_df, suffix)
        
        if not budynki_pivot.empty:
            print(f"Liczba pogrupowanych działek z budynkami ({layer_name}): {len(budynki_pivot)}")
            all_budynki_pivots.append(budynki_pivot)
        else:
            print(f"Brak danych po pivot dla warstwy: {layer_name}")
    
    # Łączenie wszystkich pivot tabel budynków
    if all_budynki_pivots:
        # Rozpocznij od pierwszej tabeli pivot
        combined_budynki_pivot = all_budynki_pivots[0]
        
        # Łącz kolejne tabele pivot po ID_DZIALKI
        for i in range(1, len(all_budynki_pivots)):
            combined_budynki_pivot = pd.merge(
                combined_budynki_pivot, 
                all_budynki_pivots[i], 
                on='ID_DZIALKI', 
                how='outer'
            )
        
        print(f"Łączna liczba rekordów po połączeniu pivot tabel: {len(combined_budynki_pivot)}")
    else:
        print("Błąd: Brak danych z pivot tabel budynków")
        return
    
    # 3. Łączenie z tabelą działek
    merged_df = merge_tables(dzialki_df, combined_budynki_pivot)
    print(f"Liczba rekordów po połączeniu z działkami: {len(merged_df)}")
    
    # 4. Mapowanie pól według schematu
    output_table = create_output_table_multi_layers(merged_df, budynki_layers.keys())
    
    # 5. Zapis do Excel
    output_path = os.path.join(QgsProject.instance().homePath(), "output_table.xlsx")
    save_to_excel(output_table, output_path)
    output_path2 = "/home/adrian/Documents/JXPROJEKT/analiza_schemat/output_table.xlsx"
    save_to_excel(output_table, output_path2)
    print(f"Tabela została zapisana w: {output_path}")


def create_budynki_pivot_with_suffix(budynki_df, suffix):
    """
    Tworzy pivot z tabeli budynków według ID_DZIALKI z odpowiednim suffixem w nazwach kolumn
    """
    if 'ID_DZIALKI' not in budynki_df.columns:
        print("Błąd: Brak kolumny ID_DZIALKI w tabeli budynków")
        return pd.DataFrame()
    
    print("Kolumny w tabeli budynków:", budynki_df.columns.tolist())
    print("Typy danych:", budynki_df.dtypes)
    
    # Funkcja do bezpiecznego uśredniania
    def safe_mean(x):
        numeric_vals = pd.to_numeric(x, errors='coerce')
        valid_vals = numeric_vals.dropna()
        return valid_vals.mean() if len(valid_vals) > 0 else None
    
    # Funkcja do łączenia tekstów
    def safe_join(x):
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
    
    # Dodanie suffixu do nazw kolumn (oprócz ID_DZIALKI)
    if suffix:
        new_columns = ['ID_DZIALKI']  # ID_DZIALKI pozostaje bez suffixu
        for col in grouped.columns[1:]:  # Pomijamy pierwszą kolumnę (ID_DZIALKI)
            new_columns.append(f"{col}{suffix}")
        grouped.columns = new_columns
    
    return grouped


def create_output_table_multi_layers(qgis_layer_df, layer_names):
    """
    Tworzy tabelę wyjściową z mapowaniem pól według schematu
    Obsługuje wiele warstw budynków z suffixami
    """
    output_table = pd.DataFrame()
    
    # Mapowanie podstawowych pól (bez zmian)
    output_table['Lp.'] = pd.to_numeric(
        qgis_layer_df.get('Lp.', ''), errors='coerce'
    )
    output_table['nr działki'] = qgis_layer_df.get('NUMER_DZIALKI', '')
    output_table['nr obrębu'] = qgis_layer_df.get('NUMER_OBREBU', '')
    output_table['powierzchnia działki [m2]'] = pd.to_numeric(
        qgis_layer_df.get('POLE_EWIDENCYJNE', ''), errors='coerce'
    ).round(2)
    output_table['rodzaj zabudowy'] = qgis_layer_df.get('RODZAJ_ZABUDOWY', '')
    output_table['powierzchnia zabudowy [m2]'] = pd.to_numeric(
        qgis_layer_df.get('S_POW_ZABUD', ''), errors='coerce'
    ).round(2)
    
    # Wskaźniki (bez zmian)
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
    
    # Mapowanie parametrów budynków dla każdej warstwy
    for layer_name in layer_names:
        if layer_name == 'budynki_parametry':
            # Podstawowa warstwa - używana tylko gdy brak warstw z suffixami
            suffix = ""
            col_suffix = ""
        else:
            # Określ suffix na podstawie nazwy warstwy
            parts = layer_name.split('_')
            if len(parts) > 2:
                suffix = '_' + '_'.join(parts[2:])  # Pełny suffix z podkreślnikami
                col_suffix = '_' + '_'.join(parts[2:])  # Suffix dla nazw kolumn
            else:
                suffix = "_1"
                col_suffix = "_1"
        
        # Nazwy kolumn w DataFrame po pivot
        szer_elew_col = f"szer_elew_front{suffix}"
        wysokosc_col = f"wysokosc{suffix}"
        nachylenie_col = f"nachylenie{suffix}"
        kategoria_col = f"Kategoria{suffix}"
        
        # Nazwy kolumn w output_table
        output_szer_col = f"szerokość elewacji frontowej [m]{col_suffix}"
        output_wys_col = f"wysokość zabudowy [m]{col_suffix}"
        output_dach_col = f"rodzaj dachu{col_suffix}"
        output_nach_col = f"kąt nachylenia połaci dachowych [o]{col_suffix}"
        
        # Mapowanie danych
        if szer_elew_col in qgis_layer_df.columns:
            output_table[output_szer_col] = pd.to_numeric(
                qgis_layer_df[szer_elew_col], errors='coerce'
            ).round(2)
        else:
            output_table[output_szer_col] = ''
            
        if wysokosc_col in qgis_layer_df.columns:
            output_table[output_wys_col] = pd.to_numeric(
                qgis_layer_df[wysokosc_col], errors='coerce'
            ).round(2)
        else:
            output_table[output_wys_col] = ''
            
        if kategoria_col in qgis_layer_df.columns:
            output_table[output_dach_col] = qgis_layer_df[kategoria_col]
        else:
            output_table[output_dach_col] = ''
            
        if nachylenie_col in qgis_layer_df.columns:
            output_table[output_nach_col] = qgis_layer_df[nachylenie_col]
        else:
            output_table[output_nach_col] = ''
    
    # Sortowanie według Lp.
    output_table = output_table.sort_values(by="Lp.")
    output_table = fix_building_columns_suffixes(output_table)
    return output_table


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



def merge_tables(dzialki_df, budynki_pivot):
    """
    Łączy tabele działek z pivot budynków
    """
    # Łączenie po ID_DZIALKI
    merged = pd.merge(dzialki_df, budynki_pivot, on='ID_DZIALKI', how='left')
    
    return merged




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
        dachy_csv_path = f"{folder_dachy}/{dachy_files[i]}"
        geometriaDachow = utworz_geometria_dachow(dachy_csv_path)
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
    Dynamicznie wykrywa suffixy warstw budynków
    """
    
    # Wykrycie dostępnych suffixów dla parametrów budynków
    building_suffixes = detect_building_suffixes(output_table)
    print(f"Wykryte suffixy warstw budynków: {building_suffixes}")
    
    # Przygotowanie pomocniczych zmiennych
    wpz_float = prepare_wpz_float(output_table)
    wpbc_float = prepare_wpbc_float(output_table)
    
    # Lista wyników
    results_data = []
    
    # Podstawowe wskaźniki (bez suffixów)
    basic_indicators = [
        ('maks_inten_zab', lambda: calculate_safe_mean(output_table, "WIZ wskaźnik intensywności zabudowy", multiplier=1.2)),
        ('maks_nadz_inten_zab', lambda: calculate_safe_mean(output_table, "WNIZ wskaźnik nadziemnej intensywności zabudowy", multiplier=1.2)),
        ('min_nadz_inten_zab', lambda: calculate_safe_min(output_table, "WNIZ wskaźnik nadziemnej intensywności zabudowy")),
        ('Sredni_wiz', lambda: calculate_safe_mean(output_table, "WIZ wskaźnik intensywności zabudowy")),
        ('Sredni_wniz', lambda: calculate_safe_mean(output_table, "WNIZ wskaźnik nadziemnej intensywności zabudowy")),
        ('Sredni_wpz', lambda: f"{round(sum(wpz_float)/len(wpz_float)*100, 0):.0f}%" if wpz_float else "0%"),
        ('Sredni_wpbc', lambda: f"{round(sum(wpbc_float)/len(wpbc_float)*100, 0):.0f}%" if wpbc_float else "0%"),
        ('wpz_min', lambda: f"{round(min(wpz_float)*100, 0):.0f}%" if wpz_float else "0%"),
        ('wpz_max', lambda: f"{round(max(wpz_float)*100, 0):.0f}%" if wpz_float else "0%"),
        ('wpbc_min', lambda: f"{round(min(wpbc_float)*100, 0):.0f}%" if wpbc_float else "0%"),
        ('wpbc_max', lambda: f"{round(max(wpbc_float)*100, 0):.0f}%" if wpbc_float else "0%"),
    ]
    
    # Dodanie podstawowych wskaźników
    for nazwa, func in basic_indicators:
        try:
            wartosc = func()
            results_data.append({'nazwa_pola': nazwa, 'wartosc': wartosc})
        except Exception as e:
            print(f"Błąd przy obliczaniu {nazwa}: {str(e)}")
            results_data.append({'nazwa_pola': nazwa, 'wartosc': ""})
    
    # Dynamiczne generowanie parametrów budynków dla każdego wykrytego suffixu
    for suffix in building_suffixes:
        # Definicje parametrów budynków dla danego suffixu
        building_params = [
            ('SrElewFront', 'szerokość elewacji frontowej [m]', lambda col: f"{calculate_safe_mean(output_table, col)} m"),
            ('srWysZab', 'wysokość zabudowy [m]', lambda col: f"{calculate_safe_mean(output_table, col)} m"),
            ('wys_zab_min', 'wysokość zabudowy [m]', lambda col: f"{calculate_safe_min(output_table, col)} m"),
            ('wys_zab_max', 'wysokość zabudowy [m]', lambda col: f"{calculate_safe_max(output_table, col)} m"),
            ('MinszerElewFront', 'szerokość elewacji frontowej [m]', lambda col: f"{calculate_safe_min(output_table, col)} m"),
            ('MaxszerElewFront', 'szerokość elewacji frontowej [m]', lambda col: f"{calculate_safe_max(output_table, col)} m"),
            ('szerElewFront08', 'szerokość elewacji frontowej [m]', lambda col: f"{round(calculate_safe_mean(output_table, col) * 0.8, 2)} m"),
            ('szerElewFront12', 'szerokość elewacji frontowej [m]', lambda col: f"{round(calculate_safe_mean(output_table, col) * 1.2, 2)} m"),
        ]
        
        # Generowanie parametrów dla danego suffixu
        for param_name, column_base, calc_func in building_params:
            try:
                # Nazwa kolumny w output_table
                column_name = f"{column_base}{suffix}"
                
                # Nazwa pola w results_table
                result_field_name = f"{param_name}{suffix}"
                
                # Obliczenie wartości jeśli kolumna istnieje
                if column_name in output_table.columns:
                    wartosc = calc_func(column_name)
                else:
                    wartosc = "0 m"
                
                results_data.append({
                    'nazwa_pola': result_field_name, 
                    'wartosc': wartosc
                })
                
            except Exception as e:
                print(f"Błąd przy obliczaniu {param_name}{suffix}: {str(e)}")
                results_data.append({
                    'nazwa_pola': f"{param_name}{suffix}", 
                    'wartosc': ""
                })
    
    # Dodanie linii zabudowy (jeśli istnieje)
    if 'linia zabudowy' in output_table.columns:
        try:
            lz_min = calculate_safe_min(output_table, "linia zabudowy")
            lz_max = calculate_safe_max(output_table, "linia zabudowy")
            results_data.extend([
                {'nazwa_pola': 'linia_zabudowy_min', 'wartosc': f"{lz_min} m"},
                {'nazwa_pola': 'linia_zabudowy_max', 'wartosc': f"{lz_max} m"}
            ])
        except Exception as e:
            print(f"Błąd przy obliczaniu linii zabudowy: {str(e)}")
            results_data.extend([
                {'nazwa_pola': 'linia_zabudowy_min', 'wartosc': "0 m"},
                {'nazwa_pola': 'linia_zabudowy_max', 'wartosc': "0 m"}
            ])
    
    # Tworzenie DataFrame wyników
    results_df = pd.DataFrame(results_data)
    
    return results_df


def detect_building_suffixes(output_table):
    """
    Wykrywa wszystkie suffixy używane w parametrach budynków w output_table
    """
    building_param_columns = [
        'szerokość elewacji frontowej [m]',
        'wysokość zabudowy [m]',
        'rodzaj dachu',
        'kąt nachylenia połaci dachowych [o]'
    ]
    
    suffixes = set()
    
    for column in output_table.columns:
        for param in building_param_columns:
            if column.startswith(param):
                # Wyciągnij suffix
                suffix = column[len(param):]
                if suffix:
                    suffixes.add(suffix)
                else:
                    # Brak suffixu - dodaj domyślny "_0"
                    suffixes.add("_0")
    
    # Jeśli nie znaleziono żadnych parametrów budynków, dodaj domyślny suffix
    if not suffixes:
        suffixes.add("_0")
    
    return sorted(list(suffixes))


def fix_building_columns_suffixes(output_table):
    """
    Poprawia nazwy kolumn parametrów budynków - dodaje suffix "_0" tam gdzie go brak
    """
    building_param_columns = [
        'szerokość elewacji frontowej [m]',
        'wysokość zabudowy [m]',
        'rodzaj dachu',
        'kąt nachylenia połaci dachowych [o]'
    ]
    
    columns_to_rename = {}
    
    for param in building_param_columns:
        if param in output_table.columns:
            # Znaleziono kolumnę bez suffixu - dodaj "_0"
            new_name = f"{param}_0"
            columns_to_rename[param] = new_name
    
    if columns_to_rename:
        output_table = output_table.rename(columns=columns_to_rename)
        print(f"Dodano suffix '_0' do kolumn: {list(columns_to_rename.keys())}")
    
    return output_table

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
    dachy = pd.read_csv(dachy_csv_path, encoding='utf-8')
    dachy_layer_name = dachy_csv_path.split("/")[-1]
    # Obliczenie łącznej liczby budynków
    total_budynki = dachy['liczba_wystapien'].sum()
    
    # Lista fragmentów opisu
    opisy = [f"dla budynków zaklasyfikowanych jako {dachy_layer_name}"]
    
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







 