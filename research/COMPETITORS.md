# Rozwiązania innych zespołów (GitHub) — dane zewnętrzne, papiery, metody

Stan na 13.11.2026. Przejrzane: ~200 kandydatów (forki upstreamu + wyszukiwanie po nazwie challenge),
z czego **161 repo** przeanalizowano dokładnie (≥300 linii Pythona lub ≥20 KB dokumentacji).
**141** zawiera własną pracę, **79** to rozwiązania dojrzałe (kompletny pipeline + dokument techniczny).

Zastrzeżenia, które trzeba mieć w głowie przy każdej liczbie poniżej:

- Wyniki **nie są porównywalne między zespołami**: każdy ma własny bench (inne mecze, inna degradacja,
  inna konwersja etykiet). "Clean" vs "degraded" różni się zwykle o 0.1–0.2 spellF1.
- Bench na **Metrica sample** jest wyraźnie łatwiejszy (etykiety z eventów, 3 mecze) niż IDSSE/SkillCorner.
- Okno publiczne ma **17 posiadań** — bootstrap CI typowo [0.39, 0.75]. Korelacja publiczne↔zewnętrzne
  jest słaba (np. AndaAnda: bench 0.776, publiczne 0.480; SAM-Analytics: 0.741 vs 0.764).
- Wszystko poniżej to deklaracje z README/dokumentów; nic nie było przez nas uruchamiane.
- Zliczenia repo (161, 141, 63, 55…) pochodzą z regexu po notatkach; skrypt i surowy wynik nie są
  w repo, więc liczby są **wyłącznie orientacyjne**. Tabeli wyników nie używać do rankingu.
- Żadna liczba o meczu BAR-ATH z tego dokumentu (dead-rate, granice połówek, progi) **nie jest
  priorem ani celem**: część zespołów stroiła na publicznym GT (§5), więc ich liczby niosą je
  pośrednio (AGENTS §2 reguła 2). Priory tylko z literatury piłkarskiej lub innych meczów.

## TL;DR

1. **Kanoniczna architektura** (≈ 2/3 dojrzałych repo): degradacja zewnętrznego trackingu do profilu ATD →
   per-frame klasyfikator(y) P(dead) i P(home | alive) bez ID zawodników → **Viterbi H/A/D per połowa**
   z kosztami przełączeń → reguły brzegowe (kick-off 2. połowy = D, snap do dotknięć).
2. **Dane zewnętrzne**: IDSSE/DFL (7 meczów Bundesligi) jest de facto standardem — jedyny zbiór z flagami
   `BallStatus`/`BallPossession` o tej samej semantyce co GT. Metrica sample (3 mecze, etykiety z eventów)
   jako drugi bench, SkillCorner (20 meczów A-League, 10 fps) głównie do kalibracji luk i walidacji
   cross-provider, PFF WC2022 rzadko i niemal tylko jako test.
3. **Najlepsze deklarowane wyniki** (bench zewnętrzny, degradowany): 0.74–0.85 spellF1 — ensemble'e
   sieci czasowych (TCN / MS-TCN / 1D-CNN / bi-GRU) + GBDT, multi-provider, z Viterbi lub MBR.
   Czyste GBDT + Viterbi: 0.55–0.70. Reguły + Viterbi bez uczenia: publicznie 0.42–0.58.
4. **Papiery**: realnie wpływowe są tylko Bassek et al. 2025 (dataset), Vidal-Codina 2022 i
   Link & Hoernig 2017 (dotknięcia/kontrola), Lang et al. 2022 (alive/dead), Borghesi et al. 2023 (TCN
   possession), PathCRF (KDD 2026, ball-free). Reszta to cytaty ozdobne.
5. **Co powtarzalnie NIE działało**: Smart Tagging jako wejście modelu (wiele zespołów wyrzuciło —
   tagi wyprzedzają koniec przerwy w GT), detekcja piłki z wideo (YOLO: mAP50 0.022 u jednego zespołu),
   pełny HSMM zamiast Viterbi (mieszane), self-training na dekodzie.
6. **Co powtarzalnie działało**: realistyczna degradacja (ukrywanie piłki 25–60%, rozkład luk z
   nieetykietowanych statystyk BAR-ATH), relabel etykiet DFL do "controlled possession" (debounce 2–3 s),
   symetria drużyn (swap H↔A jako augmentacja), multi-provider training, ważenie krótkich posiadań /
   granic, ensemble kilku seedów i widoków.

---

## 1. Dane zewnętrzne

### 1.1 Zbiory używane w kodzie

Liczby = repo spośród 141 nietrywialnych, w których zbiór pojawia się w ścieżce danych (nie tylko w planach).

| Zbiór | Repo | Do czego | Etykiety → H/A/D | Loader | Uwagi |
|---|---|---|---|---|---|
| **IDSSE / Sportec DFL** (Bassek et al. 2025), 7 meczów Bundesligi, 25 fps, CC-BY 4.0 | **63** | trening dead + team, strojenie dekodera, główny bench LOMO | natywne `BallStatus` + `BallPossession`; zwykle relabel do controlled possession (2–3 s debounce, `merge_brief` 1.6 s) | figshare 10.6084/m9.figshare.28196177, HF `pysport/idsse-data`, kloppy `sportec`, floodlight, często własny parser XML | dead-ball u DFL ma "quirki" providera → Anfield trenuje dwa zestawy modeli (z i bez artefaktów); COOKED celowo nie uczy dead z piłki DFL |
| **Metrica sample-data**, 3 mecze (2 CSV + 1 EPTS), 25 fps, commit `e706dd5` | **55** | drugi bench, trening klasyfikatorów, priory czasu trwania | **syntetyczne z eventów** (PASS/RECOVERY/BALL OUT…) → reguły zespołu; Senators: ~286k klatek | własne, kloppy `metrica` | łatwiejszy bench (wyniki 0.65–0.85); kilka zespołów wycofało go z powodu niejasnej licencji (ZesterJ, Dog Lovers, BAU) |
| **SkillCorner opendata**, 9–20 meczów A-League, 10 fps (MIT) | **25** | kalibracja rozkładu luk/missingness, fine-tune, test cross-provider | `possession.group` / "one-touch labels" / własne z eventów | GitHub `SkillCorner/opendata` (commit `4340d27`), upsampling 10→25 fps | piłka jest w danych "broadcast-like" — dobra do symulacji luk; Egyptian Team trenuje na nim model stoppage |
| **PFF FC WC2022**, 64 mecze | **7** | prawie wyłącznie **test / weto** (AndaAnda 0.529, Anfield 16 meczów, Data Kickers 28), Camp Nou Limits — trening | flagi posiadania PFF | własne | duży, ale wymaga rejestracji; nikt nie raportuje dużego zysku |
| **SoccerNet** (GAR `tracking-full`, FOOTPASS) | 3 | jekkiechan: 45 meczów treningowych, "silver" etykiety z touch-streamu; Camp Nou Limits: FOOTPASS lokalny holdout | srebrne etykiety z eventów | SoccerNet API | najbardziej egzotyczny i najliczniejszy zbiór w użyciu |
| **StatsBomb La Liga** (eventy) | 1 | Data Kickers: tylko prior długości posiadań | — | statsbombpy | |
| **WCSF / open_data_groningen** (SciSports/Groningen) | 1 | GabrielGausachs: trening Deep Sets + TCN ALIVE/DEAD i H/A/D, transfer do 25 Hz | etykiety WCSF | własne CSV→NPZ | jedyny zespół z tym zbiorem |
| **ISSIA-CNR** (wideo) | 1 | Mastaadmi: screen detektora WASB (porzucony) | — | — | |
| Modele pretrenowane: **Roboflow `sports` YOLOv8**, YOLO11, **FootAndBall**, Tesseract OCR, **TabPFN v2** | ~10 | detekcja piłki na wideo, OCR zegara, klasyfikator tabelaryczny | — | — | wideo prawie zawsze negatywne lub "experimental only" |

**52 repo nie używa żadnych danych zewnętrznych**: reguły + Viterbi z parametrami z "rozumowania
piłkarskiego", nieetykietowanych statystyk BAR-ATH albo (wbrew regułom) z okna publicznego.

### 1.2 Jak zespoły dopasowywały dane zewnętrzne do ATD

Ten etap okazał się ważniejszy niż wybór modelu. Powtarzające się techniki:

- **Ukrywanie piłki**: 20–60% klatek live (Espadas 25%, Agamemnon 20–60%, O(1)Burger ~41% = statystyka
  BAR-ATH). Najlepsi kopiują **rozkład długości luk** z nieetykietowanego trackingu BAR-ATH
  (NoTime, ZesterJ, Cota Cero `barath_gap_stats.py`, Melihozc) lub z SkillCorner (Kora).
- **Luki zależne od stanu**: piłka widoczna podczas dead tak jak w ATD (Barca-Hackathon),
  "moving dead-ball degrader" (AndaAnda), widoczność zależna od fazy (TercerHombre).
- **Szum pozycji**: AR(1) jitter zawodników (Perosinho), ¼ błędu pozycji zmierzonego na meczu
  (MassiuSoft), blur piłki (Espadas, COOKED: MA vs Savitzky–Golay → dwa modele).
- **Track slots**: tasowanie do 46 slotów bez ID, dropout tracków (jekkiechan `tracking_corruption.py`,
  Raumdeuters `stitch.py` 46→11v11 dla PathCRF).
- **Domain randomisation** zamiast jednego profilu (MassiuSoft, Team Sigue), mieszane copy z ciężkimi
  lukami (Kora J6).
- **Fake Smart Tagging** na meczach zewnętrznych z miss-rate BAR-ATH (Saryarka, Agamemnon, MassiuSoft,
  Rushilkr77 — generowanie XML z eventów IDSSE).
- **Relabel do definicji challenge**: DFL `BallPossession` przełącza się szybciej niż GT; najczęściej
  wybierane progi kontroli to 2 s (Wichai-pan, bh-14, Dog Lovers `align_control`) i 3 s
  (Anfield N=3 s, ZesterJ).

---

## 2. Papiery i metody

### 2.1 Literatura faktycznie wykorzystana

| Praca | Do czego | Kto (przykłady) |
|---|---|---|
| **Bassek, Rein, Weber, Memmert 2025**, *Sci. Data* 12:195 — IDSSE | źródło danych | ~44 repo cytuje |
| **Vidal-Codina et al. 2022**, *Automatic event detection in football using tracking data* | definicja dotknięć/kontroli, switch-at-touch | MassiuSoft, ramith001, svidoz (cały pipeline), MRH2818 |
| **Link & Hoernig 2017**, *PLoS ONE* — individual ball possession | cechy zespołu, cel piłki, piki dotknięć | Frame Control, HSA Connection, TercerHombre, MRH2818 |
| **Lang, Wild, Isenko, Link 2022**, *Sci. Rep.* 12:16291 — alive/dead | cechy alive/dead, kontekst wieloskalowy | HSA Connection, Team Nexus, ZainnB |
| **Borghesi et al. 2023**, *ESWA* 223:119780 — TCN possession | reimplementacja sieci kontaktu | **Barca4Ever** (TCN/attention + HGB), Team Nexus, TercerHombre |
| **Kim et al., PathCRF** (KDD 2026, arXiv 2602.12080) — posiadanie bez piłki | H/A bez piłki, pretrenowany na IDSSE | **Raumdeuters** (w submisji), Anfield (przetestowany, odrzucony), ZainnB (research) |
| **Kim et al. 2023** (KDD, ball trajectory imputation / "Ball Radar") | imputacja piłki z zawodników | Perosinho (transformer ball model), Camp Nou Limits (ball-radar head) |
| **ELASTIC** (Kim et al. 2025) | detekcja dotknięć | TercerHombre |
| **TranSPORTmer** (ACCV 2024) | set-transformer na trajektoriach | Perosinho (cytat) |
| **MS-TCN** (Farha & Gall 2019), ASFormer | segmentacja akcji | Camp Nou Limits, TercerHombre |
| **Spearman 2018 / Fernández & Bornn 2018** — pitch control | cechy kontroli przestrzeni | Pompeu, Suleyman, debjyoty |
| **Augste & Cordes 2016**, StatsBomb/La Liga effective playing time | priory czasu przerw / dead-rate | yelhakeem, jazzybruno, Data Kickers |
| Kaplan 2020 (scaling laws), Hernandez 2022 (repeated data) | uzasadnienie doboru danych | Camp Nou Limits |
| Deep Sets (Zaheer 2017) | permutation-invariant encoder | GabrielGausachs, Derekko-web (player-set encoder) |
| WASB / FootAndBall (Komorowski 2020) | detekcja piłki na wideo | Mastaadmi, MassiuSoft, Cochinae — wszystko porzucone |
| Li & Link 2026 (arXiv 2606.09289), Bauer & Anzer 2021 | reguły stref/przejść do syntetycznych tagów | Rushilkr77 |
| IFAB Laws of the Game | reguły kick-off/goal kick/out | wiele repo |

### 2.2 Metody (zliczenia przybliżone, regex po notatkach, 141 repo)

| Metoda | Repo | Komentarz |
|---|---|---|
| Viterbi / HMM (3-stan H/A/D lub 2-etapowy: alive/dead → H/A) | ~104 | standard; koszty przełączeń zwykle 8–45 nats, osobne H↔A i alive↔dead |
| Reguły / maszyna stanów (control radius 1.5–2.5 m, hysteresis, backdating) | ~88 | baza prawie wszystkich; jako jedyna metoda u ~50 |
| Degradacja zewnętrznych danych | ~47 | patrz 1.2 |
| GBDT (LightGBM / XGBoost / HistGradientBoosting / CatBoost) | ~43 | najczęstszy learner; zwykle **dwa modele**: P(dead), P(home \| alive) |
| Broadcast ATD fusion | ~25 | +1–3.5% pokrycia piłki, mały zysk; Frame Control i Yamal Fan Boys mają najmocniejszy sygnał (cutaway kamery = dead) |
| Regresja logistyczna (dead / shape) | ~20 | Mantas: 2× LR + Viterbi → Metrica LOMO 0.761 |
| Pseudo-etykiety na BAR-ATH (self-training shape/dead) | ~18 | szara strefa reguł — patrz §5 |
| TCN / 1D-CNN / MS-TCN | ~15 | w czołówce |
| HSMM / semi-Markov / explicit-duration | ~12 | wyniki mieszane; Data Kickers i Pompeu porzucili na rzecz prostszego |
| YOLO / detekcja piłki z wideo | ~10 | prawie zawsze negatywne |
| GRU / bi-GRU / LSTM | ~6 | Team Sigue, jekkiechan (Fable), CoreOrigin |
| Transformer | ~5 | Dog Lovers (imputer piłki + possession), Senators, Perosinho |
| MBR (expected-spellF1) decoding | 3 | Camp Nou Limits w submisji, HellThrall2000, Més Que Data (studium) |
| Audio (gwizdek) | 2 | ZainnB (tactical audio + LR dead), Data Kickers (tylko audyt) |
| PathCRF | 2–3 | Raumdeuters |

### 2.3 Taksonomia podejść i typowe wyniki

| Klasa | Przykłady | Deklarowane spellF1 |
|---|---|---|
| **A. Reguły + Viterbi, bez uczenia** | YT, INVICTA, VISCA BARCA, alessandrotavolato, Suleyman, OPK, Matra | publiczne 0.42–0.58; brak benchu zewnętrznego |
| **B. 1–2 modele GBDT/LR + Viterbi**, IDSSE/Metrica | COOKED, Wichai-pan, ZesterJ, yelhakeem, O(1)Burger, Mantas, OffsideFC, HSA, palashpawar | zewn. 0.55–0.77 (Metrica wyżej, DFL niżej) |
| **C. Sieci czasowe (TCN/MS-TCN/CNN/GRU) + GBDT, ensemble, multi-provider** | Barca-Hackathon, Team Sigue, SAM-Analytics, TercerHombre, Kora, AndaAnda, Camp Nou Limits, Anfield, Agamemnon, Fable | zewn. **0.70–0.85** |
| **D. Ball-free** (PathCRF / shape models) | Raumdeuters, Data Kickers (in-gap shape decoder) | DFL degraded ~0.60 |
| **E. Fuzja wideo/broadcast** | Frame Control, Letinsh, Yamal Fan Boys, Bodring, Pompeu B8 | publiczne 0.55–0.65; zysk głównie na dead |

---

## 3. Czołówka (dojrzałe repo, sortowane po deklarowanym benchu zewnętrznym)

| Zespół (repo) | Model | Dane zewn. | Bench zewn. spellF1 | Publiczne 5 min |
|---|---|---|---|---|
| Barca-Hackathon (`Barca-Hackathon/more-than-a-hack-2027`) | dead-ball HMM + 8× temporal CNN ensemble + Viterbi z min. czasem D | Metrica, DFL, SkillCorner | **0.853** held-out | — |
| CoreOrigin (`7dracoder`) | HGB + CNN-BiGRU-TCN, 780 cech, residual TCN | DFL, Metrica, SkillCorner | 0.857 (1 mecz dev), 0.732 test | 0.446 |
| Team Sigue (`MarkMolnar232926`) | bi-GRU ×3 seed × 4 widoki, boundary-weighted CE, Viterbi 16/24 | DFL, SkillCorner, Metrica | DFL 0.741 / SC 0.747 / Metrica 0.855 | 0.662 |
| TercerHombre (`TEKAB2`) | GBDT + multi-seed MS-TCN, touch peaks, Viterbi | DFL, Metrica | 0.775 (LOO 10 meczów) | — |
| AndaAnda (`JulienErbland`) | LightGBM + TCN, średnia geometryczna 4 modeli, Viterbi 40 | DFL, Metrica; PFF tylko eval | 0.776 LOMO; PFF 0.529 | 0.480 |
| SAM-Analytics-2027 | HGB + 3× 1D-CNN @5 Hz, Viterbi z min. czasem | DFL, Metrica, SkillCorner (waga 0.25) | 0.741 (10 meczów), SC 0.777 | 0.764 |
| Kora Analytics (`younis-y`) | GBT + temporal CNN, HSMM-style dekoder | DFL (jedyne etykiety), SkillCorner (tylko missingness) | 0.767 ± 0.029 nested 7-fold | nie używane |
| O(1)Burger (`AyanBhardwaj1`) | 2× HistGB + symetryczne widoki + Viterbi | Metrica | 0.768 LOO (harsh 0.715) | — |
| Camp Nou Limits (`hfarid19`) | TCN + MS-TCN + ball-radar + 19.3M possessor-pointer, **MBR decode** | DFL, PFF, SkillCorner, Metrica, FOOTPASS | 0.762 (selekcja) | 0.55–0.71 (warianty) |
| Mantas Team | 2× regresja logistyczna + Viterbi | Metrica | 0.761 LOMO | — |
| Més Que Data (`AndriesDKock`) | 2-etapowe GBDT + auto-context ±10 s + Viterbi | Metrica, DFL | Metrica 0.723 / DFL 0.642 | — |
| Barca4Ever (`aqaPayam`) | Borghesi TCN/attention + HGB, boundary detector, Viterbi | DFL | clean 0.711 / noisy 0.533 | — |
| Anfield (`Derekko-web/mtah2027-submission`) | XGBoost + 2× TCN, 238 przepisów ensemble, Viterbi N=3 s | DFL, SkillCorner; PFF/Metrica weto | 0.709 | — |
| MassiuSoft (`EnricGS`) | 3× LightGBM + 5× MLP + q(change), Viterbi λ=32 | DFL, Metrica; SC luki; PFF bench | 0.706 (¼ pos-err), clean 0.809 | tylko check |
| Agamemnon (`agamemnon-labs`) | dilated 1D-CNN + Viterbi + touch snap; Smart Tagging +0.086 | DFL, SkillCorner | 0.703 (DFL 0.748, SC 0.657) | 0.519 |
| ZesterJ | LightGBM 78 cech + min-duration 3 s | DFL | 0.688 LOMO | ~0.57 |
| AOM (`ra189zor`) | XGBoost + TCN 85k, Viterbi 40/8 | DFL; Metrica eval | clean 0.675 / moderate 0.555 | — |
| Fable (`jekkiechan`) | 204 cechy → bi-GRU H/A + player-only GRU D, Viterbi 3 | DFL, Metrica, SkillCorner, **SoccerNet-GAR** 45 | 0.647 clean / 0.506 stress | 0.630 |
| Frame Control (`danieltyukov`) | 3× LightGBM + broadcast shot-type, 4-stan Viterbi | DFL | 0.610 | 0.650 |
| Raumdeuters (`stefrazv`) | **PathCRF** H/A + HGB dead + ball-vote | DFL | ~0.604 degraded | — |
| COOKED (`akss06`) | 2× LightGBM (MA vs SG ball) + Viterbi | DFL, Metrica | 0.601 (worst 0.540) | tylko check |

Deklarowane wyniki na oknie publicznym (17 posiadań) to szum i nie mówią nic o zgodności z regułami;
nie rankujemy po nich.

---

## 4. Wnioski techniczne z cudzych eksperymentów

Co zespoły zgodnie raportują jako działające:

- **Rozdzielenie dead i team**: osobny model P(dead) bez piłki (kinematyka zawodników: spowolnienie,
  rozproszenie, ustawienie pod stały fragment) i P(home | alive). Fable: D liczone z samych zawodników.
- **Symetria**: cechy bez ID, swap H↔A jako augmentacja / test-time averaging (CoreOrigin, Sigue,
  O(1)Burger, SAM).
- **Ważenie granic i krótkich posiadań** w stracie (Sigue boundary-weighted CE, SAM short-spell weights,
  Saryarka, Mantas ×40 na klatkach przekazania). Bezpośrednio adresuje spellF1.
- **Viterbi jest wystarczający**; zysk z dekodera +0.06–0.13 spellF1 (Dog Lovers). HSMM/MBR dają mało
  ponad dobrze dobrane koszty + minimalny czas trwania.
- **Snap granic** do wykrytych dotknięć / wyjść piłki (Agamemnon, DSBA, Barca4Ever ±3 klatki,
  FC Medallo boundary detector +1.2 pp).
- **Okno przed kick-offem 2. połowy jako twarde D** — powszechne. Cudze numery klatek nie są do
  kopiowania: u nas początek połowy i wznowienie wynikają wyłącznie z `valid_runs` i kotwicy
  `kickoff_2nd` w `games.py`, nigdy z literału (AGENTS §2 reguła 1).
- **Multi-provider**: trening na DFL + SkillCorner + Metrica uśrednia "quirki" dead-ball każdego providera
  (Anfield, Sigue, SAM). Ocena na providerze niewidzianym w treningu (Fable, Kora).

Co raportują jako niedziałające lub marginalne:

- **Smart Tagging** jako wejście modelu: często porzucany (MG team: tagi wyprzedzają koniec przerwy w GT;
  Frame Control: niska zgodność; Data Kickers −0.001). Wyjątek: Agamemnon (+0.086). Używany głównie
  do kick-offów i restartów.
- **Wideo / YOLO piłki**: mAP50 0.022 (ezramathias14), "mostly negative" (CoreOrigin), porzucone
  (Cochinae, Mastaadmi WASB, Tiki Taka Masala).
- **Broadcast fill piłki**: +1.2–3.5% klatek; Fable/Sigue/AOM go nie używają.
- **Self-training na własnym dekodzie** (DSBA: szkodzi), **PathCRF w ensemble** (Anfield: odrzucony),
  **joint fusion dead+team** (AOM: psuje dead).
- **Pełny 5-stanowy HSMM** vs Viterbi: Pompeu hsmm_v1 0.449 < rules 0.506 publicznie.
- Częsty błąd: zespoły raportują wyraźnie za dużo albo za mało D. Udział D to osobny problem;
  oczekiwany poziom bierzemy z literatury (np. Siegle & Lames 2012) i innych meczów, nie z liczb
  innych zespołów o BAR-ATH.

---

## 5. Flagi zgodności z regułami (z dokumentacji samych zespołów)

Nie oceniamy innych zespołów i nie wymieniamy ich tutaj (to wnioski z README, niezweryfikowane).
Notujemy tylko kategorie ryzyka, których mamy unikać:

| Kategoria ryzyka | Ile repo (orientacyjnie) |
|---|---|
| Trening/strojenie na **publicznym GT** | ~10 |
| Wagi/progi dobrane na oknie publicznym przy innym modelu | ~3 |
| Ręczne etykiety na meczu BAR-ATH (u nas zakazane w każdej ilości) | ~5 |
| Fit nienadzorowany / pseudo-etykiety / EM na meczu konkursowym | ~8 |
| Udostępnienie statystyk meczu hostowanemu LLM | ~2 |

---

## 6. Implikacje dla naszego rozwiązania

1. **Dane**: IDSSE (7) jako rdzeń + SkillCorner (20) do kalibracji luk i testu cross-provider
   + Metrica (3) jako trzeci provider. PFF WC2022 jako test zewnętrzny, jeśli zdążymy pobrać —
   nikt nie pokazał, że trening na nim pomaga. SoccerNet-GAR to jedyna droga do dziesiątek meczów
   więcej (Fable), ale ze "srebrnymi" etykietami.
2. **Symulator ATD** (nasz pomysł z `RESEARCH.md`) jest potwierdzony przez czołówkę jako kluczowy.
   Kalibracja z nieetykietowanych statystyk BAR-ATH (rozkład luk, 46 slotów, jitter) to **szara
   strefa** względem pkt 7 i §5: **do decyzji zespołu** (ew. pytanie do organizatorów), zapisanej
   w docs/DECISIONS.md przed implementacją. Do tego czasu: domain randomisation z SkillCorner
   i szerokich zakresów, bez dopasowania do tego meczu.
3. **Model**: GBDT na cechach bez ID jako szybka baza (cel ~0.6 na degradowanym DFL), potem
   temporalna sieć (TCN / bi-GRU, kontekst ±5–10 s) w ensemble z GBDT. Dwa wyjścia: P(dead), P(home | alive)
   + opcjonalnie P(change) do kosztów przełączeń.
4. **Strata i dekoder pod spellF1**: ważenie granic, Viterbi per połowa z min. czasem D i kosztami
   strojonymi na benchu zewnętrznym; snap do dotknięć. MBR jako eksperyment.
5. **Ball-free** (PathCRF albo własny shape model) tylko jako składnik dla luk piłki (41% klatek),
   nie jako główny model.
6. **Walidacja**: LOMO na providerach + "provider held-out"; publiczne 5 minut tylko raz na końcu
   jako sanity check (17 posiadań = szum ±0.18).
7. **Nie robić**: tuning na publicznym GT, ręczne etykiety BAR-ATH do selekcji, fit EM/pseudo-label
   na meczu konkursowym, inwestycji w YOLO piłki.
