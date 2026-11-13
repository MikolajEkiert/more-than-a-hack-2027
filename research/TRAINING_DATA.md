# Na czym i jak trenować model — publiczne dane i metody uczenia

Stan: 13.11.2026. Uzupełnia `RESEARCH.md` (metryka, architektura) i `COMPETITORS.md` (co zrobili inni).
Wszystkie fakty o zbiorach zweryfikowane w źródłach (linki na końcu) lub w kodzie `kloppy` 3.19.1.

## TL;DR

1. **Rdzeń treningowy = IDSSE (7 meczów) + PFF WC2022 (64 mecze).** Tylko te dwa otwarte zbiory dają
   **per-klatkowe** `ball_state` + `ball_owning_team` pasujące semantyką do naszego GT. PFF ma ~9× więcej
   meczów niż IDSSE, a w `COMPETITORS.md` prawie nikt go nie użył do treningu — to nasza największa
   niewykorzystana przewaga. IDSSE trzymamy jako bench o semantyce "złotej" (natywne flagi TRACAB).
2. **Drugi krąg (słabsze etykiety, inny provider)**: SkillCorner (10 meczów, 10 fps), Metrica sample (3,
   **ten sam provider co ATD**, etykiety z eventów), FOOTPASS (54 mecze z top-lig, pozycje z broadcastu +
   play-by-play), SoccerTrack v2 (10 meczów amatorskich, CC BY, panorama + eventy).
3. **Najlepsza metoda uczenia** = nadzorowany trening na zharmonizowanych etykietach + **realistyczna
   degradacja do profilu ATD z domain randomisation** + **samonadzorowany pretraining (masked trajectory
   modelling) na wszystkich trajektoriach, łącznie z nieetykietowanym BAR-ATH** — to ostatnie jest wprost
   dozwolone przez `RULES.md` i jest jedyną legalną formą adaptacji do meczu testowego.
4. **Czego oczekiwać** (literatura, dane własnościowe): alive/dead z samych zawodników 92% acc. per klatka
   (Lang 2022, 102 mecze); 4 stany (pass/possession/uncontrolled/out) bez piłki 80.8%, z piłką ~89%
   (TranSPORTmer, 283 mecze LaLiga). Skala danych pomaga — dlatego PFF.
5. **Wideo** (opcjonalnie, tylko dla D): SoccerNet-v2 — 500 meczów, ~30.5k znaczników "Ball out of play",
   13 klas ujęć kamery (replay/close-up ≈ przerwa). Do detektora przerw na broadcaście, nie do H/A.

---

## 1. Co mówią reguły (granice dla danych i metod)

Z `RULES.md` §1–2 i `FAQ.md`:

| Dozwolone | Zabronione |
|---|---|
| Dowolne dane zewnętrzne o **innych meczach**, w tym własne ręczne etykiety (muszą być dostępne dla organizatorów i opisane w dokumencie) | Jakiekolwiek ręczne etykiety na BAR-ATH (trening, walidacja, poprawki) |
| Na BAR-ATH: **metody nienadzorowane / samonadzorowane / algorytmiczne pseudo-etykiety**, o ile żadnej nie zdecydował człowiek | Pełne GT, cokolwiek z niego wyprowadzone; publiczne 5 min jako zbiór treningowy lub do strojenia |
| Modele pretrenowane, LLM-y | Dane ludzkie o BAR-ATH od kogokolwiek (Opta, StatsBomb, LaLiga, …) |

Wniosek metodyczny: **pretraining samonadzorowany na ATD BAR-ATH i statystyki nieetykietowane** (rozkład luk
piłki, szum pozycji) są legalne i są najbardziej bezpośrednią drogą do zmniejszenia domain gap.

---

## 2. Publiczne zbiory — szczegółowa ocena

### 2.1 Zbiory z per-klatkową etykietą posiadania / stanu piłki

| | **IDSSE / Sportec DFL** | **PFF FC WC2022** | **SkillCorner opendata** |
|---|---|---|---|
| Mecze | 7 (Bundesliga 1+2, 2022/23) | **64** (MŚ 2022) | 10 (A-League 2024/25) |
| Źródło trackingu | optyczny TRACAB, pełne boisko | **broadcast tracking**, zawodnicy poza kadrem ekstrapolowani | broadcast tracking, `is_detected` per zawodnik |
| fps | 25 | ~29.97 | 10 |
| Piłka | 3D, prawie zawsze | XYZ, z lukami jak w broadcaście | z lukami |
| Etykieta stanu | **natywne `BallStatus`** | w kloppy wyprowadzone z eventów: DEAD po `BALL_OUT` / `PERIOD_END`, ALIVE od kick-offu / `ON_THE_BALL` | `ALIVE` ⇔ `possession.group` ≠ null (zgrubne) |
| Etykieta drużyny | **natywne `BallPossession`** | `home_ball` z każdego eventu z piłką (eventy mają `frameNum`) | `possession.group` (home/away) |
| Licencja | CC-BY 4.0 | darmowe po formularzu (warunki do sprawdzenia przy pobraniu) | MIT |
| Loader | `kloppy.sportec.load_open_tracking_data(match_id=…)`, HF `pysport/idsse-data` | `kloppy.pff.load_tracking(meta, roster, raw)`; ~3 GB zip, ~1 GB/mecz | `kloppy.skillcorner`, GitHub LFS (~90 MB/mecz) |
| Klatek (≈) | ~1.0 M | **~11 M** | ~0.6 M (×2.5 po upsamplingu) |
| Rola u nas | **bench o złotej semantyce + trening** | **główny wolumen treningowy** | test cross-provider, kalibracja luk |

Uwagi do harmonizacji:

- **IDSSE**: `BallPossession` przełącza się szybciej niż nasza definicja "controlled possession"; relabel
  z progiem 2–3 s (wnioski z `COMPETITORS.md`). Dead-ball DFL ma quirki providera (Anfield trenował dwa
  warianty).
- **PFF**: semantyka stanu piłki jest *eventowa*. W kloppy DEAD włącza się tylko po wyjściu piłki i końcu
  połowy — **faule, spalone, kontuzje i VAR nie dają D**. Trzeba dołożyć przerwy z eventów PFF
  (faul → D do następnego `ON_THE_BALL`; sprawdzić typy w specyfikacji PDF dołączonej do zbioru) albo
  trenować głowę dead tylko na IDSSE, a głowę H/A na IDSSE+PFF. Resampling 29.97 → 25 fps.
  Broadcast tracking z ekstrapolowanymi zawodnikami jest **bliższy jakościowo naszemu ATD** niż czysty
  TRACAB — to dodatkowy argument za PFF.
- **SkillCorner**: brak odróżnienia "dead" od "nikt nie ma piłki w locie" → nie do treningu głowy D;
  do H/A i do statystyk luk/widoczności.

### 2.2 Zbiory z eventami (etykiety trzeba wyprowadzić)

| Zbiór | Co jest | Jak zrobić H/A/D | Ocena |
|---|---|---|---|
| **Metrica sample-data** (3 mecze, 25 fps; G1–2 CSV, G3 EPTS) | tracking + eventy, zsynchronizowane | reguły event→klatka (PASS/RECOVERY/BALL LOST → drużyna; BALL OUT/FAULT → D do SET PIECE) | **ten sam provider co ATD** (format, układ współrzędnych), ale brak flag w kloppy (`ball_state=None`). Licencja nieformalna ("be responsible, acknowledge the source"). Dobry trzeci provider do walidacji |
| **FOOTPASS** (SoccerNet 2026; 54 mecze L1/BL/SA/LL/UCL 2023–24) | pozycje + prędkości z rekonstrukcji broadcastu (niewidoczni zawodnicy imputowani), 103k eventów play-by-play z klatką (Drive 39%, Pass 50%, Throw-in, …) | Drive/Pass/Tackle → drużyna; brak jawnych przerw — D tylko z restartów (Throw-in) i luk | **największy zbiór top-lig z trackingiem i eventami**; pozycje z broadcastu ≈ nasz szum. Dostęp przez HF SoccerNet, wideo za NDA. Warto do H/A i do pretrainingu |
| **SoccerTrack v2** (10 meczów uniwersyteckich, ~934 min, CC BY 4.0) | per-klatkowe pozycje z panoramy (GSR), 21.4k eventów BAS z drużyną i graczem, rekonstrukcja piłki, surowy tracking i eventy providera (BePro) | eventy → drużyna; piłka z interpolacji między eventami | amatorzy (inna dynamika), ale **panorama pełnego boiska ≈ nasza kamera taktyczna** i otwarta licencja. Jedyny otwarty zbiór, na którym da się *legalnie i tanio ręcznie oznaczyć stan piłki z wideo* |

### 2.3 Wideo (tylko dla detekcji przerw / D)

| Zbiór | Co daje | Użycie |
|---|---|---|
| **SoccerNet-v2** (500 meczów broadcast) | 110k znaczników akcji, w tym ~30.5k "Ball out of play", ~13.4k Throw-in, ~2.3k Kick-off, Foul, Offside, Corner…; 13 klas ujęć kamery; replay grounding; gotowe cechy 2 fps | detektor "piłka poza grą / restart" na broadcaście BAR-ATH + klasyfikator ujęć (replay/close-up ≈ przerwa, co u Letinsh/Yamal Fan Boys było najmocniejszym sygnałem D). Wymaga synchronizacji broadcast↔tactical |
| SoccerNet Ball Action Spotting / GSR 2025 | gęste akcje z piłką, klipy GSR 30 s | niskie znaczenie dla nas |

### 2.4 Syntetyczne

- **Google Research Football** (pipeline z arXiv 2503.19809): nieograniczona liczba meczów z dokładnym
  posiadaczem i trybem gry. Duży sim-to-real gap (fizyka, taktyka); Kora (za Radke & Tilbury, AAMAS 2025)
  próbował i porzucił. Ewentualnie tylko do pretrainingu.
- **SoccER** (synthetic events) — przestarzałe, pomijamy.

### 2.5 Niedostępne, ale ważne jako punkt odniesienia

- Lang et al. 2022: 102 mecze Bundesligi z ręcznym in-game status — AdaBoost na samych zawodnikach:
  **92.0% acc** per klatka, stoppage F1 0.80 (≥2 s), granice w ±2 s w 77% (start) / 81% (koniec).
- TranSPORTmer (ACCV 2024): 283 mecze LaLiga, 4 stany per klatka: **~89% z piłką, 80.8% bez piłki**.
- WACV 2026 (Peral et al.): ten sam typ danych + crop'y zawodników z kamery taktycznej, masked pre-training.

Te liczby mówią, że z samych trajektorii da się dojść wysoko, ale przy **dziesiątkach–setkach meczów**.
Stąd priorytet wolumenu: PFF + FOOTPASS.

### 2.6 Ranking zbiorów dla naszego zadania

| Priorytet | Zbiór | Głowa D | Głowa H/A | Pretraining | Walidacja |
|---|---|---|---|---|---|
| 1 | IDSSE (7) | ✔ (złota) | ✔ | ✔ | ✔ LOMO, złota semantyka |
| 1 | PFF WC2022 (64) | ⚠ po uzupełnieniu przerw | ✔ | ✔ | ✔ provider-held-out |
| 2 | Metrica sample (3) | ⚠ z eventów | ✔ | ✔ | ✔ ten sam provider co ATD |
| 2 | SkillCorner (10) | ✘ | ✔ | ✔ | ✔ broadcast-like |
| 2 | FOOTPASS (54) | ⚠ tylko restarty | ✔ | ✔ | opcjonalnie |
| 3 | SoccerTrack v2 (10) | ⚠ (ew. ręczne etykiety) | ✔ | ✔ | opcjonalnie |
| 3 | SoccerNet-v2 (500, wideo) | ✔ detektor wideo | ✘ | — | — |
| — | **BAR-ATH ATD (bez etykiet)** | — | — | ✔ **(dozwolone)** | ✘ |

---

## 3. Metody uczenia na istniejących danych

### 3.1 Nadzorowane na zharmonizowanych etykietach (fundament)

- Wspólny schemat klatki: 25 fps, metry, środek boiska (0,0), kierunek ataku znormalizowany per połowa,
  **bez ID zawodników** (zbiory 22 punktów + piłka z maską widoczności).
- Etykiety → nasza definicja: H/A/D + "controlled possession" (debounce 2–3 s), D zgodnie z GT (restart
  kończy D). Osobny parametr per provider, sprawdzany na IDSSE (złota semantyka).
- Dwie głowy: **P(dead)** (uczona tam, gdzie D jest wiarygodne: IDSSE, PFF po uzupełnieniu, Metrica) i
  **P(home | alive)** (uczona na wszystkim). Opcjonalnie trzecia: P(zmiana posiadania w ±k klatek).
- Strata pod spellF1: ważenie granic i krótkich posiadań, symetria H↔A (swap jako augmentacja i TTA).

### 3.2 Sim-to-real: degradacja do profilu ATD (najważniejszy pojedynczy element)

Profil liczony **wyłącznie z nieetykietowanego ATD BAR-ATH**: rozkład długości luk piłki (41% klatek
bez piłki), widoczność zależna od fazy, jitter pozycji, 46 slotów zamiast ID, gubienie zawodników.
Zamiast jednego profilu — **domain randomisation** wokół niego (MassiuSoft, Team Sigue, SAM w
`COMPETITORS.md`). Każdy mecz treningowy w wielu kopiach o różnym stopniu zniszczenia.

### 3.3 Samonadzorowany pretraining na trajektoriach

- **T-MAE** (Machine Learning 2024; Leuphana): masked autoencoder na trajektoriach wielu agentów,
  faktoryzowany transformer (czas × agenci), permutation-equivariant. Pretraining poprawia klasyfikację
  zdarzeń zwłaszcza przy małej liczbie etykiet.
- **TranSPORTmer** (ACCV 2024): jeden model do imputacji piłki, predykcji trajektorii i klasyfikacji stanu
  per klatka; maska wejściowa naturalnie modeluje brak piłki.
- Dla nas: pretraining (rekonstrukcja zamaskowanych segmentów zawodników **i piłki**) na wszystkich
  trajektoriach z §2 **+ ATD BAR-ATH (tactical i broadcast)**, potem fine-tune głów z §3.1. Maskowanie
  piłki w pretrainingu = dokładnie nasze 41% luk. To legalna adaptacja do domeny testowej.

### 3.4 Zadania pomocnicze (multi-task)

- **Imputacja piłki** z zawodników (Ball Radar, KDD 2023; TranSPORTmer) jako głowa pomocnicza — uczy
  "gdzie jest piłka" dokładnie tam, gdzie ATD jej nie ma.
- **Wskaźnik posiadacza / ścieżka posiadania** (PathCRF, KDD 2026: Set Attention + CRF po krawędziach
  gracz→gracz, pretrenowane checkpointy na IDSSE w repo autorów; jest też fine-tune na SkillCorner na HF).
  Drużyna = drużyna wybranego węzła.
- **Detekcja dotknięć / zmian** (ELASTIC 2025 synchronizuje eventy z trackingiem — przydaje się też do
  poprawy etykiet event→klatka dla Metrica/FOOTPASS/PFF).

### 3.5 Słaby nadzór i własne etykiety na innych meczach

- Etykiety z eventów (Metrica, FOOTPASS, SoccerTrack, PFF) są zaszumione w czasie → wyrównać ELASTIC-iem
  albo traktować granice miękko (label smoothing w oknie ±0.5 s).
- **Ręczne etykiety na innych meczach są dozwolone bez limitu.** Najtańszy cel: SoccerTrack v2 (CC BY,
  wideo panoramiczne + pozycje) — ręczne oznaczenie przerw (D) na 10 meczach uzupełnia jedyną słabą
  głowę. Trzeba opisać w dokumencie technicznym (kto, jak, ile) i udostępnić organizatorom.

### 3.6 Adaptacja do BAR-ATH bez etykiet

- Pretraining z §3.3 (pewny zysk, zgodny z regułami).
- Dopasowanie statystyk normalizacji (AdaBN / test-time BN) na ATD BAR-ATH — tanie, bez etykiet.
- Algorytmiczne pseudo-etykiety / self-training są dozwolone, ale w `COMPETITORS.md` raczej szkodziły
  (DSBA) — tylko z twardym progiem pewności i walidacją na zewnętrznym benchu.
- Domain-adversarial (DANN: dyskryminator provider vs BAR-ATH) — opcjonalnie, gdy degradacja nie
  wystarczy.

### 3.7 Struktura sekwencji i dekodowanie

- Model per klatka (TCN / bi-GRU / transformer, kontekst ±5–10 s) → **Viterbi / semi-Markov** z
  minimalnymi czasami trwania, koszty strojone na benchu zewnętrznym (nie na publicznych 5 min).
- Alternatywa end-to-end: CRF na wyjściu sieci (jak PathCRF) uczony log-likelihoodem sekwencji.

### 3.8 Walidacja

- **Leave-one-match-out + leave-one-provider-out** (IDSSE / PFF / Metrica / SkillCorner) z oficjalnym
  `possession_metrics`, po degradacji do profilu ATD.
- Decyzje (cechy, hiperparametry, wybór modelu) tylko na tym benchu; publiczne 5 min raz na końcu.

---

## 4. Rekomendowany plan

| Krok | Dane | Model | Cel |
|---|---|---|---|
| 0 | IDSSE + Metrica, degradacja v1 | 2× LightGBM (dead, home \| alive) + Viterbi | bench LOMO działa, ~0.55–0.65 na degradowanym DFL |
| 1 | + PFF (64), harmonizacja przerw, resampling | te same głowy, więcej danych | zysk z wolumenu; test provider-held-out |
| 2 | + SkillCorner, FOOTPASS do H/A; domain randomisation | TCN / bi-GRU per klatka (ensemble z GBDT) | ≥0.70 na benchu zewnętrznym (poziom czołówki) |
| 3 | wszystkie trajektorie + ATD BAR-ATH bez etykiet | masked-trajectory pretraining → fine-tune; głowa imputacji piłki | odporność na 41% luk, adaptacja do domeny |
| 4 | SoccerNet-v2 (wideo) | detektor "ball out / restart" + klasa ujęcia na broadcaście | lepsze D, jeśli broadcast da się zsynchronizować |

Ryzyka: harmonizacja przerw w PFF (bez niej głowa D się zepsuje), synchronizacja broadcastu (65.8%
klatek z homografią), licencje (PFF formularz, Metrica nieformalna, FOOTPASS wideo NDA).

---

## Źródła

- IDSSE: Bassek et al. 2025, *Sci. Data* 12:195, doi:10.1038/s41597-025-04505-y; HF `pysport/idsse-data`.
- PFF WC2022: blog.fc.pff.com (release), kloppy `pff` loader (`kloppy/infra/serializers/tracking/pff.py`).
- SkillCorner: github.com/SkillCorner/opendata (10 meczów A-League 2024/25, MIT).
- Metrica: github.com/metrica-sports/sample-data.
- FOOTPASS: arXiv 2511.16183; github.com/JeremieOchin/FOOTPASS.
- SoccerTrack v2: huggingface.co/datasets/atomscott/soccertrack-v2 (CC BY 4.0).
- SoccerNet-v2: Deliège et al., CVPRW 2021.
- Lang, Wild, Isenko, Link 2022, *Sci. Rep.* 12:16291.
- Capellera et al., TranSPORTmer, ACCV 2024 (arXiv 2410.17785).
- T-MAE: *Machine Learning* 2024, doi:10.1007/s10994-024-06647-3.
- Kim et al., Ball Radar / ball trajectory inference (arXiv 2306.08206, KDD 2023).
- Kim et al., PathCRF, KDD 2026 (arXiv 2602.12080; github.com/hyunsungkim-ds/pathcrf).
- Kim et al., ELASTIC 2025 (arXiv 2508.09238).
- Peral et al., Multi-Modal Soccer Scene Analysis with Masked Pre-Training, WACV 2026.
- Simulated tracking from Google Research Football, arXiv 2503.19809.
