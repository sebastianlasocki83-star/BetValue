# BetValue 5.5.3 — World Leagues

Wersja mobilna BetValue z rozszerzonym katalogiem lig.

## Co dodano
- 42 obsługiwane przez The Odds API rozgrywki piłkarskie: ligi Tier 1 oraz dostępne Tier 2.
- Filtr **Kraj / region**.
- Wybór ligi w ramach wybranego kraju.
- Drugi poziom jest dodany tam, gdzie The Odds API faktycznie udostępnia daną ligę (np. Championship, Ligue 2, 2. Bundesliga, Serie B, La Liga 2, Superettan, Brazil Série B).
- Dla Polski obecnie The Odds API udostępnia Ekstraklasę; **1 Liga Polska nie znajduje się na aktualnej liście sportów The Odds API**, więc nie można jej pobierać z tego źródła bez dodatkowego dostawcy danych.

## Model
BetValue nie udaje wyceny, gdy brakuje historii wyników. Dla lig bez wystarczającej historii Football-Data aplikacja wyświetla informację zamiast generować fikcyjne prawdopodobieństwa.

## Wdrożenie z telefonu
Do repozytorium GitHub wgraj tylko:
- `app.py`
- `requirements.txt`
- `README.md`

W Streamlit Cloud ustaw `app.py` jako Main file.

W **Manage app → Settings → Secrets**:

```toml
ODDS_API_KEY = "TWÓJ_PRAWDZIWY_KLUCZ"
```

Nie publikuj klucza API w GitHub.
