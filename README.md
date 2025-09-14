# WZ Workflow - Plugin QGIS

**Narzędzie do zarządzania workflow analiz wniosków o warunki zabudowy (WZ)**

![QGIS](https://img.shields.io/badge/QGIS-3.0%2B-green)
![Python](https://img.shields.io/badge/Python-3.6%2B-blue)
![License](https://img.shields.io/badge/License-GPL-red)

## Opis

WZ Workflow to zaawansowana wtyczka QGIS zaprojektowana do automatyzacji i standaryzacji procesu tworzenia analiz do wniosków o warunki zabudowy. Plugin oferuje kompletny 14-stopniowy workflow, który prowadzi użytkownika przez wszystkie etapy analizy - od identyfikacji terenu inwestycji po generowanie końcowych raportów.

## Główne funkcjonalności

### 🔄 Zarządzanie Workflow
- **14-stopniowy proces** analizy WZ
- **System checkpointów** - możliwość wznowienia pracy
- **Automatyczne wykrywanie** aktualnego kroku
- **Intuitive GUI** z dock widget
- **Monitorowanie postępu** w czasie rzeczywistym

### 🏗️ Analiza przestrzenna
- Wyznaczanie granic terenu inwestycji
- Tworzenie buforów obszarów analizowanych
- Pomiary wymiarów i linii zabudowy
- Klasyfikacja chmur punktów
- Obliczanie wskaźników urbanistycznych
- Analiza parametrów budynków

### 📊 Generowanie raportów
- Automatyczne zestawienia dachów
- Analizy opisowe
- Eksport do formatów DOCX i PDF
- Załączniki graficzne

### 🛠️ Narzędzia pomocnicze
- Zapisywanie warstw tymczasowych
- Dodawanie pól do warstw
- Przetwarzanie chmur punktów
- Klasyfikacja PBC (Pokrycie i Użytkowanie Terenu)

## Wymagania systemowe

### Minimalne wymagania
- **QGIS**: 3.0 lub nowszy
- **Python**: 3.6+
- **System**: Windows/Linux/macOS

### Wymagane biblioteki Python
- `PyQt5` (standardowo w QGIS)
- `pandas` - do analizy danych
- `pathlib` - do zarządzania ścieżkami
- `json` - do checkpointów

### Zalecane wtyczki QGIS
- **GISsupport** - do identyfikacji terenów z ULDK
- Wtyczki do przetwarzania chmur punktów

## Instalacja

### Metoda 1: Instalacja z pliku ZIP

1. Pobierz najnowszą wersję z [Releases](../../releases)
2. W QGIS: `Wtyczki` → `Zarządzaj wtyczkami` → `Zainstaluj z ZIP`
3. Wybierz pobrany plik ZIP
4. Włącz wtyczkę w liście wtyczek

### Metoda 2: Instalacja manualna

1. Sklonuj repozytorium:
```bash
git clone https://github.com/AdrLin/AdrLin-wz_workflow_qgis_plugin.git
```

2. Skopiuj folder do katalogu wtyczek QGIS:
- **Windows**: `%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\`
- **Linux**: `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/`
- **macOS**: `~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/`

3. Zrestartuj QGIS i włącz wtyczkę

## Użytkowanie

### Uruchomienie wtyczki

1. Po instalacji znajdziesz ikonę WZ Workflow na pasku narzędzi
2. Kliknij ikonę lub przejdź do `Wtyczki` → `WZ Workflow`
3. Otworzy się dock widget z interfejsem workflow

### Rozpoczęcie nowej analizy

1. **Przygotowanie projektu**:
   - Utwórz nowy projekt QGIS
   - Zapisz projekt w odpowiednim folderze

2. **Wybór typu analizy**:
   - `Nowa standardowa analiza` - pełny 14-stopniowy proces
   - `Użyj dostępnych funkcji` - dostęp do pojedynczych narzędzi
   - `Kontynuuj od kroku X` - wznowienie przerwanych prac

3. **Wykonanie kroków workflow**:
   - Plugin prowadzi przez kolejne etapy
   - Każdy krok ma jasne instrukcje
   - System automatycznie sprawdza wyniki

### Główne etapy workflow

| Krok | Opis | Wynik |
|------|------|-------|
| 1 | Odnalezienie terenu inwestycji | `granica_terenu` |
| 2 | Bufor obszaru analizowanego | `granica_obszaru_analizowanego` |
| 3-4 | Rysowanie i zapis wymiarów | `wymiary` |
| 5 | Wyznaczanie działek i budynków | `dzialki_w_obszarze` |
| 6 | Pomiar elewacji frontowych | `budynki_z_szer_elew_front` |
| 7 | Przetwarzanie chmury punktów | `Classification_2` |
| 8 | Klasyfikacja PBC | `punkty_pbc_wyniki_predykcji` |
| 9 | Weryfikacja punktów | - |
| 10 | Obliczanie wskaźników | `dzialki_ze_wskaznikami` |
| 11 | Parametry budynków | `budynki_parametry` |
| 12 | Linie zabudowy | `linie_zabudowy` |
| 13 | Dane działki inwestora | - |
| 14 | Generowanie wyników końcowych | Raporty PDF/DOCX |

## Szczegółowe funkcjonalności

### Pomiar linii zabudowy
```python
# Automatyczne uruchomienie narzędzia do pomiaru
- Tworzenie warstwy linie_zabudowy
- Automatyczne obliczanie długości
- Zapisywanie w formacie GPKG
```

### Analiza budynków
- Automatyczny podział według funkcji zabudowy
- Generowanie zestawień dachów
- Obliczanie parametrów technicznych
- Klasyfikacja rodzajów pokryć

### System checkpointów
Plugin automatycznie zapisuje postęp w pliku `workflow_checkpoint.json`:
```json
{
  "step": 5,
  "step_name": "wyznacz_dzialki",
  "timestamp": "2025-01-15T10:30:00",
  "liczba_budynkow": 3,
  "rozne_funkcje": true
}
```

## Struktura plików

```
wz_workflow_plugin/
├── __init__.py                 # Inicjalizacja wtyczki
├── wz_workflow_plugin.py       # Główna klasa wtyczki
├── improved_wz_workflow.py     # Logika workflow
├── metadata.txt               # Metadane wtyczki
├── icon.png                   # Ikona wtyczki
└── README.md                  # Dokumentacja
```

## Konfiguracja

### Struktura folderów projektu
Plugin automatycznie tworzy strukturę folderów:
```
projekt_wz/
├── projekt.qgz
├── budynki_parametry_dachy/
├── chmura/
├── wyniki/
└── workflow_checkpoint.json
```

### Wymagane warstwy bazowe
- Warstwy z ULDK (działki, budynki)
- Chmura punktów LiDAR
- Warstwy referencyjne WMS/WMTS

## Rozwiązywanie problemów

### Najczęstsze problemy

**Problem**: Plugin nie uruchamia się
```
Rozwiązanie:
1. Sprawdź czy wszystkie zależności są zainstalowane
2. Sprawdź logi QGIS (Menu -> Zobacz -> Panele -> Log Messages)
3. Upewnij się że projekt jest zapisany
```

**Problem**: Brak warstwy po wykonaniu kroku
```
Rozwiązanie:
1. Sprawdź czy poprzedni krok się wykonał poprawnie
2. Użyj funkcji "Poprzedni krok" w panelu kontroli
3. Sprawdź komunikaty w panelu wtyczki
```

**Problem**: Błąd importu modułów
```
Rozwiązanie:
1. Sprawdź instalację pandas: pip install pandas
2. Zrestartuj QGIS
3. Reinstaluj wtyczkę
```

### Debug mode
Aby włączyć szczegółowe logowanie, odkomentuj linie debug w kodzie:
```python
# DEBUG = True  # Odkomentuj dla debugowania
```

## API i rozszerzenia

### Dodawanie własnych funkcji
```python
def moja_funkcja(self):
    """Własna funkcja do workflow"""
    self.add_message("Wykonuję własną funkcję", "info")
    # Twoja logika
    return True

# Dodanie do mapy funkcji
self.funkcje_map["Moja funkcja"] = self.moja_funkcja
```

### Tworzenie własnych skryptów
Plugin może wykonywać zewnętrzne skrypty Python umieszczone w folderze wtyczki.

## Licencja

Ten projekt jest licencjonowany na podstawie licencji GPL v3. Zobacz plik [LICENSE](LICENSE) po szczegóły.

## Autor

- **Email**: link.mapy@gmail.com
- **GitHub**: [@AdrLin](https://github.com/AdrLin)

## Wsparcie

- **Issues**: [GitHub Issues](https://github.com/AdrLin/AdrLin-wz_workflow_qgis_plugin/issues)
- **Dokumentacja**: [Wiki](https://github.com/AdrLin/AdrLin-wz_workflow_qgis_plugin/wiki)
- **Email**: link.mapy@gmail.com

## Changelog

### v1.0.0 (2025-01-15)
- Pierwsza wersja public release
- Kompletny 14-stopniowy workflow
- System checkpointów
- Automatyczne generowanie raportów
- Pomiar linii zabudowy z automatycznym obliczaniem

## Roadmap

- [ ] Integracja z bazami danych przestrzennymi
- [ ] Eksport do formatów CAD
- [ ] Wsparcie dla analiz 3D
- [ ] API REST dla automatyzacji
- [ ] Interfejs webowy

---

⭐ **Jeśli ten plugin Ci pomógł, zostaw gwiazdkę na GitHubie!**
