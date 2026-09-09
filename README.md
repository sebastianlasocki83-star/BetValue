# BetValue 5.5.2 — Mobile Edition

Wersja przygotowana dokładnie pod upload pojedynczych plików z telefonu.

## Do GitHub wgraj tylko 3 pliki do głównego katalogu repozytorium
- `app.py`
- `requirements.txt`
- `README.md`

Nie tworzysz folderu `betvalue/` ani `.streamlit/`.

## Streamlit Community Cloud
1. Wgraj 3 pliki do ROOT repozytorium GitHub.
2. W Streamlit wybierz repozytorium i `app.py`.
3. **Settings → Secrets** i dodaj:

```toml
ODDS_API_KEY = "TWÓJ_KLUCZ_Z_THE_ODDS_API"
```

4. Zapisz i zrestartuj aplikację.

Klucza API nie publikuj w GitHub.
