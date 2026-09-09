import os
import sqlite3
import json
import math
import statistics
import time
from datetime import datetime, timezone, timedelta

import streamlit as st
import pandas as pd
import numpy as np
import requests

VERSION = "5.5.2"

# ===== data.py =====

def football_data(season='2526',league='E0'):
    url=f'https://www.football-data.co.uk/mmz4281/{season}/{league}.csv'
    r=requests.get(url,timeout=25); r.raise_for_status()
    return pd.read_csv(io.BytesIO(r.content))

def clean(df):
    d=df.copy(); d['Date']=pd.to_datetime(d['Date'],dayfirst=True,errors='coerce'); d=d.dropna(subset=['Date','HomeTeam','AwayTeam','FTHG','FTAG'])
    return d.sort_values('Date')

# ===== odds.py =====

BASE='https://api.the-odds-api.com/v4'
SPORTS={'Premier League':'soccer_epl','La Liga':'soccer_spain_la_liga','Bundesliga':'soccer_germany_bundesliga','Serie A':'soccer_italy_serie_a','Ligue 1':'soccer_france_ligue_one','Eredivisie':'soccer_netherlands_eredivisie','Ekstraklasa':'soccer_poland_ekstraklasa'}
MARKETS=['h2h','totals','spreads']

def api_key():
    try:
        import streamlit as st
        if 'ODDS_API_KEY' in st.secrets:return st.secrets['ODDS_API_KEY']
    except Exception:pass
    return os.getenv('ODDS_API_KEY','')

def request_api(path, params):
    key=api_key()
    if not key: raise RuntimeError('Brak ODDS_API_KEY. Dodaj klucz w Streamlit Secrets lub zmiennej środowiskowej.')
    p=dict(params); p['apiKey']=key
    r=requests.get(f'{BASE}{path}',params=p,timeout=30)
    r.raise_for_status()
    return r.json(),{k:v for k,v in r.headers.items() if k.lower().startswith('x-')}

def fetch_odds(sport_key,regions='eu',markets='h2h,totals,spreads'):
    return request_api(f'/sports/{sport_key}/odds/',{'regions':regions,'markets':markets,'oddsFormat':'decimal','dateFormat':'iso'})

def fetch_historical_odds(sport_key,date_iso,regions='eu',markets='h2h,totals,spreads'):
    return request_api(f'/historical/sports/{sport_key}/odds/',{'regions':regions,'markets':markets,'oddsFormat':'decimal','dateFormat':'iso','date':date_iso})

def remove_margin(probs):
    vals=[max(0,1/x) for x in probs if x and x>1]; s=sum(vals)
    return [v/s for v in vals] if s else []

def market_consensus(outcomes):
    rows=[(o['name'],o['price']) for o in outcomes if o.get('price',0)>1]
    if not rows:return {}
    raw=[1/p for _,p in rows]; s=sum(raw)
    return {name:r/s for (name,_),r in zip(rows,raw)}

def parse_events(events):
    rows=[]
    for e in events:
        for b in e.get('bookmakers',[]):
            for m in b.get('markets',[]):
                for o in m.get('outcomes',[]):
                    rows.append({'event_id':e['id'],'commence_time':e.get('commence_time'),'home':e.get('home_team'),'away':e.get('away_team'),'bookmaker':b.get('title'),'market':m.get('key'),'outcome':o.get('name'),'price':o.get('price'),'point':o.get('point')})
    return rows

def best_prices(rows):
    best={}
    for r in rows:
        if not r.get('price') or r['price']<=1:continue
        key=(r['event_id'],r['market'],r['outcome'],r.get('point'))
        if key not in best or r['price']>best[key]['price']:best[key]=r
    return list(best.values())

def group_peers(rows,event_id,market,outcome,point):
    return [x for x in rows if x['event_id']==event_id and x['market']==market and x['outcome']==outcome and x.get('point')==point and x.get('price',0)>1]

# ===== models.py =====

def poisson_pmf(k, lam):
    return math.exp(-lam) * lam**k / math.factorial(k)

def score_matrix(home_lambda, away_lambda, max_goals=8):
    h = np.array([poisson_pmf(i, home_lambda) for i in range(max_goals+1)])
    a = np.array([poisson_pmf(i, away_lambda) for i in range(max_goals+1)])
    return np.outer(h, a)

def dc_tau(i, j, lh, la, rho=-0.06):
    if i == 0 and j == 0: return 1 - lh*la*rho
    if i == 0 and j == 1: return 1 + lh*rho
    if i == 1 and j == 0: return 1 + la*rho
    if i == 1 and j == 1: return 1 - rho
    return 1.0

def dc_matrix(lh, la, rho=-0.06, max_goals=8):
    m = score_matrix(lh, la, max_goals)
    for i in range(max_goals+1):
        for j in range(max_goals+1): m[i,j] *= dc_tau(i,j,lh,la,rho)
    return m / m.sum()

def probs_from_matrix(m):
    home = np.tril(m, -1).sum()
    draw = np.trace(m)
    away = np.triu(m, 1).sum()
    return {'home': float(home), 'draw': float(draw), 'away': float(away)}

def totals_probs(m, line=2.5):
    over = sum(m[i,j] for i in range(m.shape[0]) for j in range(m.shape[1]) if i+j > line)
    return {'over': float(over), 'under': float(1-over)}

def btts_probs(m):
    yes = sum(m[i,j] for i in range(1,m.shape[0]) for j in range(1,m.shape[1]))
    return {'yes': float(yes), 'no': float(1-yes)}

def fit_team_strengths(df):
    d = df.copy().sort_values('Date')
    teams = sorted(set(d.HomeTeam) | set(d.AwayTeam))
    atk = {t:1.0 for t in teams}; deff = {t:1.0 for t in teams}
    for _ in range(12):
        new_atk={}; new_def={}
        for t in teams:
            hm=d[d.HomeTeam==t]; aw=d[d.AwayTeam==t]
            scored=list(hm.FTHG)+list(aw.FTAG); conceded=list(hm.FTAG)+list(aw.FTHG)
            if len(scored):
                new_atk[t]=max(.35, min(2.8, np.mean(scored)/(np.mean([x for _,r in d.iterrows() for x in [r.FTHG,r.FTAG]])/2)))
                new_def[t]=max(.35, min(2.8, np.mean(conceded)/(np.mean([x for _,r in d.iterrows() for x in [r.FTHG,r.FTAG]])/2)))
            else: new_atk[t]=new_def[t]=1.0
        atk,deff=new_atk,new_def
    return atk,deff

def expected_goals(df, home, away):
    league_home=max(.2, df.FTHG.mean()); league_away=max(.2, df.FTAG.mean())
    h=df[df.HomeTeam==home]; a=df[df.AwayTeam==away]
    home_scored=h.FTHG.mean() if len(h) else league_home
    home_conc=h.FTAG.mean() if len(h) else league_away
    away_scored=a.FTAG.mean() if len(a) else league_away
    away_conc=a.FTHG.mean() if len(a) else league_home
    lh=.55*home_scored+.45*away_conc
    la=.55*away_scored+.45*home_conc
    return max(.15,lh), max(.15,la)

def elo_ratings(df, k=22, home_adv=55):
    r={t:1500.0 for t in set(df.HomeTeam)|set(df.AwayTeam)}
    for _,x in df.sort_values('Date').iterrows():
        h,a=x.HomeTeam,x.AwayTeam; rh,ra=r[h],r[a]
        eh=1/(1+10**((ra-(rh+home_adv))/400)); res=1 if x.FTHG>x.FTAG else .5 if x.FTHG==x.FTAG else 0
        margin=max(1,abs(x.FTHG-x.FTAG)); k_eff=k*(1+0.15*math.log1p(margin))
        r[h]+=k_eff*(res-eh); r[a]+=k_eff*((1-res)-(1-eh))
    return r

def ensemble(df, home, away):
    lh,la=expected_goals(df,home,away); m=dc_matrix(lh,la); p=probs_from_matrix(m)
    elo=elo_ratings(df); rh=elo.get(home,1500); ra=elo.get(away,1500)
    pe=1/(1+10**((ra-(rh+55))/400))
    # conservative ELO blend: 75% goal model, 25% ELO
    p['home']=.75*p['home']+.25*pe
    p['away']=.75*p['away']+.25*(1-pe)
    p['draw']=max(0.0,1-p['home']-p['away'])
    return p, m, (lh,la)

# ===== engine.py =====

DB=os.getenv('BETVALUE_DB','betvalue.db')

def fair_odds(p): return 1/p if p and p>0 else float('inf')
def ev(p, odds): return p*odds-1 if p and odds and odds>0 else None

def kelly(p, odds, fraction=.25, cap=.10):
    if not p or not odds or odds<=1: return 0.0
    b=odds-1; q=1-p; f=(b*p-q)/b
    return max(0.0,min(cap,f*fraction))

def value_score(ev_pct, consensus_gap, sources, steam_pct=0, uncertainty=0):
    if ev_pct is None: return 0
    score=50 + min(30,max(-20,ev_pct*1.35))
    score += min(10,max(-10,consensus_gap*100))
    score += min(7,sources*1.4)
    score += min(6,max(-6,steam_pct*0.8))
    score -= min(15,uncertainty*100)
    return int(max(0,min(100,round(score))))

def clv(open_odds, close_odds):
    # Positive when the price taken was better than closing price.
    if not open_odds or not close_odds or open_odds<=1 or close_odds<=1:return None
    return ((1/close_odds)/(1/open_odds)-1)*100

def init_db():
    c=sqlite3.connect(DB)
    c.execute('''CREATE TABLE IF NOT EXISTS odds_snapshots(
        ts TEXT,event_id TEXT,home TEXT,away TEXT,commence_time TEXT,
        market TEXT,outcome TEXT,point REAL,bookmaker TEXT,price REAL)''')
    c.execute('''CREATE INDEX IF NOT EXISTS idx_snap_event ON odds_snapshots(event_id,market,outcome,point,ts)''')
    c.execute('''CREATE TABLE IF NOT EXISTS signals(
        created_at TEXT,event_id TEXT,home TEXT,away TEXT,commence_time TEXT,
        market TEXT,outcome TEXT,point REAL,bookmaker TEXT,price REAL,
        model_prob REAL,fair_odds REAL,ev_pct REAL,score INTEGER,kelly25_pct REAL,
        signal TEXT)''')
    c.execute('''CREATE INDEX IF NOT EXISTS idx_signal_event ON signals(event_id,market,outcome,point,created_at)''')
    c.commit(); c.close()

def save_snapshot(rows):
    init_db(); now=datetime.now(timezone.utc).isoformat()
    c=sqlite3.connect(DB)
    vals=[]
    for r in rows:
        if r.get('price'):
            vals.append((now,r['event_id'],r.get('home'),r.get('away'),r.get('commence_time'),r.get('market'),r.get('outcome'),r.get('point'),r.get('bookmaker'),r.get('price')))
    c.executemany('INSERT INTO odds_snapshots VALUES (?,?,?,?,?,?,?,?,?,?)',vals)
    c.commit(); c.close()

def save_signals(rows):
    if not rows:return
    init_db(); now=datetime.now(timezone.utc).isoformat(); c=sqlite3.connect(DB)
    vals=[]
    for r in rows:
        vals.append((now,r.get('event_id'),r.get('home'),r.get('away'),r.get('commence_time'),r.get('market'),r.get('outcome'),r.get('point'),r.get('bookmaker'),r.get('price'),r.get('P(model)'),r.get('Fair'),r.get('EV %'),r.get('Score'),r.get('Kelly 25%'),r.get('Signal')))
    c.executemany('INSERT INTO signals VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',vals)
    c.commit(); c.close()

def steam_for(event_id,market,outcome,point=None):
    init_db(); c=sqlite3.connect(DB)
    q='''SELECT ts,price FROM odds_snapshots WHERE event_id=? AND market=? AND outcome=?
         AND ((point=? ) OR (point IS NULL AND ? IS NULL)) ORDER BY ts ASC'''
    df=pd.read_sql_query(q,c,params=(event_id,market,outcome,point,point)); c.close()
    if len(df)<2:return {'move_pct':0.0,'samples':len(df),'first':None,'last':None}
    first=float(df.iloc[0].price); last=float(df.iloc[-1].price)
    return {'move_pct':(last/first-1)*100,'samples':len(df),'first':first,'last':last}

def snapshot_history(event_id,market,outcome,point=None,limit=200):
    init_db(); c=sqlite3.connect(DB)
    q='''SELECT ts,bookmaker,price FROM odds_snapshots WHERE event_id=? AND market=? AND outcome=?
         AND ((point=? ) OR (point IS NULL AND ? IS NULL)) ORDER BY ts DESC LIMIT ?'''
    df=pd.read_sql_query(q,c,params=(event_id,market,outcome,point,point,limit)); c.close()
    if not df.empty: df['ts']=pd.to_datetime(df['ts'],utc=True)
    return df

def signal_history(limit=500):
    init_db(); c=sqlite3.connect(DB)
    df=pd.read_sql_query('SELECT * FROM signals ORDER BY created_at DESC LIMIT ?',c,params=(limit,)); c.close()
    return df

def performance_from_signals(df):
    if df.empty:return {}
    bets=df[df['Signal']=='BET'].copy()
    if bets.empty:return {'bets':0}
    return {'bets':len(bets),'avg_ev':float(bets['ev_pct'].mean()),'avg_score':float(bets['score'].mean()),'avg_kelly25':float(bets['kelly25_pct'].mean())}

# ===== performance.py =====

def settle_row(row, results):
    if results.empty: return None
    home, away = row['home'], row['away']
    r = results[(results.HomeTeam==home)&(results.AwayTeam==away)]
    if r.empty:
        r = results[(results.HomeTeam.astype(str).str.lower()==str(home).lower())&(results.AwayTeam.astype(str).str.lower()==str(away).lower())]
    if r.empty: return None
    x=r.iloc[-1]; hg,ag=int(x.FTHG),int(x.FTAG)
    market=row['market']; outcome=row['outcome']; point=row.get('point')
    if market=='h2h':
        actual='Home' if hg>ag else 'Draw' if hg==ag else 'Away'
        return {'win':actual==outcome,'actual':actual,'goals':hg+ag}
    if market=='totals' and point is not None:
        total=hg+ag; actual='Over' if total>float(point) else 'Under'
        return {'win':actual==outcome,'actual':actual,'goals':total}
    if market=='spreads' and point is not None:
        adj = hg + float(point) if outcome==home else ag + float(point)
        opp = ag if outcome==home else hg
        return {'win':adj>opp,'actual':f'{hg}-{ag}','goals':hg+ag}
    if market=='btts':
        actual='Yes' if hg>0 and ag>0 else 'No'
        return {'win':actual==outcome,'actual':actual,'goals':hg+ag}
    return None

def max_drawdown(profits):
    if not profits:return 0.0
    curve=np.cumsum(profits); peak=np.maximum.accumulate(np.r_[0,curve])
    return float((np.r_[0,curve]-peak).min())

def evaluate_settled(df):
    if df.empty:return {}
    bets=df[df.Signal=='BET'].copy()
    if bets.empty:return {'bets':0,'settled':0}
    bets=bets.dropna(subset=['win']).copy()
    if bets.empty:return {'bets':len(df[df.Signal=='BET']),'settled':0}
    bets['profit']=np.where(bets.win==True,bets.price-1,-1.0)
    bets['stake']=1.0
    wins=int(bets.win.sum()); n=len(bets)
    p=bets['P(model)'].astype(float).clip(.000001,.999999)
    y=bets.win.astype(float)
    brier=float(np.mean((p-y)**2))
    logloss=float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))
    return {'bets':int(len(df[df.Signal=='BET'])),'settled':n,'wins':wins,'hit_rate':wins/n*100,'profit_units':float(bets.profit.sum()),'roi_pct':float(bets.profit.sum()/bets.stake.sum()*100),'max_drawdown_units':abs(max_drawdown(bets.profit.tolist())),'brier':brier,'log_loss':logloss,'avg_ev':float(bets.ev_pct.mean()),'avg_clv':float(bets.clv_pct.mean()) if 'clv_pct' in bets and bets.clv_pct.notna().any() else None}

def score_buckets(df):
    """Performance by Value Score. Only settled BET signals are included."""
    if df.empty or 'Score' not in df.columns: return pd.DataFrame()
    x=df[(df.Signal=='BET') & df.win.notna()].copy()
    if x.empty:return pd.DataFrame()
    x['bucket']=pd.cut(x.Score.astype(float),bins=[-1,60,70,80,90,101],labels=['<60','60–70','70–80','80–90','90–100'],right=False)
    rows=[]
    for bucket,g in x.groupby('bucket',observed=False):
        if len(g)==0: continue
        profit=np.where(g.win.astype(bool),g.price.astype(float)-1,-1).sum()
        rows.append({'Value Score':str(bucket),'Bets':len(g),'Hit rate %':g.win.mean()*100,'ROI %':profit/len(g)*100,'Śr. EV %':g.ev_pct.mean(),'Śr. CLV %':g.clv_pct.mean() if g.clv_pct.notna().any() else np.nan})
    return pd.DataFrame(rows)

# ===== APPLICATION =====

def _get_odds_api_key():
    try:
        key = st.secrets.get("ODDS_API_KEY", "")
    except Exception:
        key = ""
    if not key:
        key = os.environ.get("ODDS_API_KEY", "")
    return str(key).strip()

ODDS_API_KEY = _get_odds_api_key()

def _require_api_key():
    if not ODDS_API_KEY:
        st.title("BetValue 5.5.2")
        st.warning("Brakuje klucza ODDS_API_KEY.")
        st.markdown("Wejdź w **Streamlit Cloud → Settings → Secrets** i dodaj:")
        st.code('ODDS_API_KEY = "TWÓJ_KLUCZ_Z_THE_ODDS_API"')
        st.info("Zapisz sekret i uruchom ponownie aplikację.")
        st.stop()

_require_api_key()
import streamlit as st, pandas as pd, numpy as np
from datetime import datetime, timezone, timedelta

st.set_page_config(page_title='BetValue 5.5',page_icon='🎯',layout='wide')
st.title('🎯 BetValue 5.5')
st.caption('Wersja PRO: skaner value + settlement + CLV + analiza jakości Value Score')
st.caption('Market → consensus → model → fair odds → EV → CLV → settlement → performance')
init_db()

with st.sidebar:
    st.header('⚙️ Scanner')
    league=st.selectbox('Liga',list(SPORTS))
    min_ev=st.slider('Minimalne EV %',0,30,5)
    min_score=st.slider('Minimalny Value Score',0,100,65)
    max_events=st.slider('Liczba wydarzeń',5,100,30)
    if st.button('🔄 Odśwież kursy'):
        st.cache_data.clear(); st.rerun()

with st.sidebar.expander('🧰 Diagnostyka / konfiguracja', expanded=False):
    import os
    api_ok = bool(os.getenv('ODDS_API_KEY') or st.secrets.get('ODDS_API_KEY',''))
    st.write('The Odds API:', '✅ klucz wykryty' if api_ok else '❌ brak klucza')
    st.write('Baza SQLite:', '✅ aktywna')
    st.caption('Jeżeli klucza nie ma, aplikacja nie pobierze kursów z The Odds API.')

@st.cache_data(ttl=300)
def get_feed(league):
    return fetch_odds(SPORTS[league],regions='eu',markets='h2h,totals,spreads')

try:
    events,headers=get_feed(league); rows=parse_events(events); save_snapshot(rows)
except Exception as e:
    st.error(str(e)); st.stop()

try:
    code={'Premier League':'E0','Ekstraklasa':'POL','La Liga':'SP1','Bundesliga':'D1','Serie A':'I1','Ligue 1':'F1','Eredivisie':'N1'}.get(league,'E0')
    hist=clean(football_data('2526',code))
except Exception:
    hist=pd.DataFrame(columns=['Date','HomeTeam','AwayTeam','FTHG','FTAG'])

best=best_prices(rows); results=[]
for event_id in sorted(set(r['event_id'] for r in best))[:max_events]:
    evrows=[r for r in best if r['event_id']==event_id]
    if not evrows or len(hist)<20: continue
    home,away=evrows[0]['home'],evrows[0]['away']
    try:p,m,(lh,la)=ensemble(hist,home,away)
    except Exception: continue
    tp=totals_probs(m); bp=btts_probs(m)
    market_probs={'h2h':{'Home':p['home'],'Draw':p['draw'],'Away':p['away']},'totals':{'Over':tp['over'],'Under':tp['under']},'btts':{'Yes':bp['yes'],'No':bp['no']}}
    for r in evrows:
        if r['market']=='h2h': mp=market_probs['h2h'].get(r['outcome'])
        elif r['market']=='totals' and r.get('point')==2.5: mp=market_probs['totals'].get(r['outcome'])
        elif r['market']=='btts': mp=market_probs['btts'].get(r['outcome'])
        else: continue
        if not mp: continue
        peers=group_peers(rows,event_id,r['market'],r['outcome'],r.get('point'))
        consensus=float(np.mean([1/x['price'] for x in peers])) if peers else mp
        steam=steam_for(event_id,r['market'],r['outcome'],r.get('point'))
        e_pct=ev(mp,r['price'])*100
        score=value_score(e_pct,(mp-consensus),len(peers),steam_pct=steam['move_pct'],uncertainty=max(0,.12-len(peers)*.01))
        signal='BET' if e_pct>=min_ev and score>=min_score else 'NO BET'
        results.append({'Mecz':f'{home} – {away}','Rynek':r['market'],'Wybór':r['outcome'],'Punkt':r.get('point'),'Bukmacher':r['bookmaker'],'Kurs':r['price'],'P(model)':mp,'Fair':fair_odds(mp),'EV %':e_pct,'Consensus':consensus,'Gap %':(mp-consensus)*100,'Steam %':steam['move_pct'],'Sources':len(peers),'Score':score,'Kelly 25%':kelly(mp,r['price'])*100,'Signal':signal,'event_id':event_id,'home':home,'away':away,'commence_time':r.get('commence_time')})

if not results:
    st.warning('Brak wystarczających danych do wyceny.'); st.stop()
df=pd.DataFrame(results).sort_values(['Signal','Score','EV %'],ascending=[True,False,False])
save_signals(df.to_dict('records'))

c1,c2,c3,c4,c5=st.columns(5)
c1.metric('VALUE ≥ próg',int((df.Signal=='BET').sum())); c2.metric('Najwyższe EV',f'{df["EV %"].max():.1f}%'); c3.metric('Najwyższy Score',int(df.Score.max())); c4.metric('Max źródeł',int(df.Sources.max())); c5.metric('Sygnały',len(signal_history(5000)))

st.subheader('🔥 TOP VALUE')
show=df[(df['EV %']>=min_ev)&(df['Score']>=min_score)].head(25).copy()
for col in ['P(model)','Fair','Consensus','Kelly 25%','Gap %','Steam %']: show[col]=show[col].round(3)
show['EV %']=show['EV %'].round(2)
st.dataframe(show.drop(columns=['event_id','home','away','commence_time'],errors='ignore'),use_container_width=True,hide_index=True)
st.download_button('📥 Eksport TOP VALUE',show.to_csv(index=False).encode('utf-8'),file_name='betvalue55_top_value.csv',mime='text/csv')

tabs=st.tabs(['📊 Performance','📈 CLV / historia','🧪 Historyczny snapshot','🧾 Sygnały','⚙️ Metodyka'])
with tabs[0]:
    st.subheader('🏦 Backtest / settlement sygnałów')
    sh=signal_history(5000)
    if sh.empty: st.info('Brak sygnałów.')
    else:
        # Match stored signals to Football-Data results; this is a settlement ledger, not a synthetic backtest.
        settled=[]
        for _,r in sh.iterrows():
            s=settle_row(r,hist)
            z=r.to_dict(); z.update(s or {'win':np.nan,'actual':None,'goals':None})
            # CLV uses the latest locally observed price for the same event/selection when available.
            try:
                hs=snapshot_history(r.event_id,r.market,r.outcome,r.point,limit=200)
                z['clv_pct']=clv(float(r.price),float(hs.iloc[0].price)) if len(hs) else np.nan
            except Exception:z['clv_pct']=np.nan
            settled.append(z)
        perf=pd.DataFrame(settled); metrics=evaluate_settled(perf)
        a,b,c,d,e=st.columns(5)
        a.metric('Rozliczone BET',metrics.get('settled',0)); b.metric('Hit rate',f"{metrics.get('hit_rate',0):.1f}%"); c.metric('ROI',f"{metrics.get('roi_pct',0):.1f}%"); d.metric('Profit',f"{metrics.get('profit_units',0):.2f}u"); e.metric('Max DD',f"-{metrics.get('max_drawdown_units',0):.2f}u")
        a,b,c,d=st.columns(4); a.metric('Brier',f"{metrics.get('brier',0):.4f}"); b.metric('Log Loss',f"{metrics.get('log_loss',0):.4f}"); c.metric('Śr. EV',f"{metrics.get('avg_ev',0):.2f}%"); d.metric('Śr. CLV',f"{metrics.get('avg_clv',0):.2f}%" if metrics.get('avg_clv') is not None else '—')
        st.subheader('📊 Czy wyższy Value Score rzeczywiście działa?')
        buckets=score_buckets(perf)
        if buckets.empty: st.info('Za mało rozliczonych BET do analizy przedziałów Score.')
        else: st.dataframe(buckets,use_container_width=True,hide_index=True)
        st.dataframe(perf[['created_at','home','away','market','outcome','price','model_prob','ev_pct','score','Signal','win','profit','clv_pct']].head(1000),use_container_width=True,hide_index=True)
        st.download_button('📥 Eksport performance',perf.to_csv(index=False).encode('utf-8'),file_name='betvalue55_performance.csv',mime='text/csv')
        st.caption('Rozliczenie jest wykonywane względem wyników Football-Data dla tej samej pary drużyn. Jeżeli wynik nie zostanie dopasowany, zakład pozostaje nierozliczony.')
with tabs[1]:
    st.subheader('CLV i ruch kursu')
    if len(show):
        idx=st.selectbox('Wybierz sygnał',show.index.tolist()); r=df.loc[idx]; sh=snapshot_history(r.event_id,r['Rynek'],r['Wybór'],r['Punkt'])
        if sh.empty: st.info('Brak historii snapshotów.')
        else:
            st.metric('Obserwowany CLV',f'{clv(float(r.Kurs),float(sh.iloc[0].price)):.2f}%'); st.dataframe(sh,use_container_width=True,hide_index=True)
with tabs[2]:
    st.subheader('Historyczny snapshot kursów')
    st.caption('Historyczne snapshoty The Odds API wymagają płatnego planu.')
    dt=st.datetime_input('Czas UTC',datetime.now(timezone.utc)-timedelta(hours=1))
    if st.button('Pobierz historyczny snapshot'):
        try:
            h,_=fetch_historical_odds(SPORTS[league],dt.astimezone(timezone.utc).isoformat(),regions='eu',markets='h2h,totals,spreads'); hr=parse_events(h.get('data',h) if isinstance(h,dict) else h); save_snapshot(hr); st.success(f'Zapisano {len(hr)} kursów.')
        except Exception as e: st.error(f'Historyczny snapshot: {e}')
with tabs[3]:
    st.subheader('Historia sygnałów'); sh=signal_history(5000)
    st.dataframe(sh,use_container_width=True,hide_index=True) if not sh.empty else st.info('Brak zapisanych sygnałów.')
with tabs[4]:
    st.markdown('''**BetValue 5.5**

- EV = `P(model) × kurs − 1`
- Fair Odds = `1 / P(model)`
- Value Score 0–100 łączy EV, różnicę względem consensus, liczbę źródeł, steam i niepewność.
- Kelly 25% ograniczony do 10% bankrollu.
- Performance rozlicza zapisane sygnały po zakończeniu meczu; nie udaje historycznego backtestu, jeśli nie ma historycznych kursów.
- Score buckets pokazują ROI/CLV dla przedziałów Value Score 60–70, 70–80, 80–90 i 90–100.
- Brier i Log Loss mierzą jakość probabilistyczną, a ROI/CLV jakość ekonomiczną.

**BET ≠ gwarancja wygranej.** To narzędzie analityczne; jakość wyników zależy od danych, kalibracji modelu i dostępności kursów.''')
