# Deep research — Phase 1: Ball Possession

Notatki badawcze zespołu. Źródła: dokumentacja i kod tego repozytorium, literatura naukowa, publiczne
zbiory danych z **innych** meczów. Celowo **nie** analizowaliśmy rozwiązań innych drużyn.
Publiczne 5 minut GT służą tu wyłącznie do zrozumienia *semantyki* etykiet — żaden parametr nie
został z nich wyznaczony.

---

## 0. TL;DR

1. **Zadanie rozkłada się na trzy podproblemy**: (a) piłka żywa/martwa, (b) która drużyna ma piłkę,
   gdy jest żywa, (c) dekodowanie sekwencji na *segmenty* z ostrymi granicami. Metryka rankingowa
   (`spellF1`) najmocniej karze za (c): nieprecyzyjne granice i migotanie.
2. **Istnieją otwarte dane z innych meczów z etykietami w tej samej semantyce co GT** —
   flaga statusu piłki + drużyna przy piłce na każdą klatkę (IDSSE/DFL: 7 meczów, 25 fps, CC-BY 4.0;
   PFF WC2022: 64 mecze). To daje legalny zbiór treningowy i — co ważniejsze — **legalny zbiór
   walidacyjny do tuningu** z oficjalną metryką, czego publiczne 5 minut dać nie może.
3. **Luka domenowa**: tamte dane są skorygowane, nasze ATD jest automatyczne (piłka brak w 41%,
   brak ID, fragmentacja tracków). Trzeba **zaszumić dane zewnętrzne tak, żeby wyglądały jak nasze
   ATD** (rozkład luk piłki, liczba obiektów, fragmentacja) — statystyki szumu wolno mierzyć na
   naszym meczu, bo nie używają etykiet.
4. **Na brakującą piłkę**: modele „ball-free” (Ball Radar, PathCRF — 92% trafności drużyny per
   klatka z samych trajektorii zawodników) oraz **pseudo-etykiety** na naszym meczu z klatek, gdzie
   piłka jest widoczna i jednoznacznie przy zawodniku (dozwolone przez RULES.md).
5. **Na martwą piłkę**: geometria (piłka poza boiskiem, piłka nieruchoma na punkcie stałego
   fragmentu), zachowanie zespołowe (spadek prędkości, ustawienia do stałych fragmentów — Vidal-Codina
   2022), Smart Tagging `SET PIECES` jako kotwice, opcjonalnie gwizdek z audio i modele SoccerNet.
6. **Dekodowanie**: semi-Markov / HSMM (segmental Viterbi) z rozkładami długości i kosztem
   przejścia zamiast okna większości. Okno ±2 s na *idealnym* sygnale już obniża `spellF1` do 0,86.

---

## 1. Co dokładnie nagradza metryka

Źródło prawdy: `evaluation/possession_metrics.py`. Najważniejsze fakty z kodu:

- Posiadanie = **maksymalny ciąg `H` albo `A`** w obrębie połowy. `D` nie tworzy posiadania, ale
  **rozcina** je (`H H D D H H` = dwa posiadania).
- Dopasowanie predykcji do GT: jeden-do-jednego, zachłannie po malejącym IoU, **tylko ta sama
  drużyna**; F1 uśrednione po progach IoU 0,3/0,4/0,5/0,6/0,7.
- Każde posiadanie waży tyle samo — 0,8-sekundowe tyle co 40-sekundowe.
- `X` przed liczeniem `spellF1` jest wypełniane **ostatnią** etykietą w połowie (forward-fill), a na
  początku połowy **pierwszą znaną** (back-fill).
- **Pułapka**: 2. połowa jest oceniana od początku swojego bloku (`valid_runs` /
  `games.half_video_blocks`), a wznowienie następuje dopiero przy kotwicy `kickoff_2nd` z `games.py`,
  ~2 minuty później. Jeśli wpiszemy tam `X`, back-fill rozciągnie pierwsze posiadanie wstecz i je
  zabije. **Trzeba jawnie wpisać `D`**, z granicami liczonymi z tych funkcji, nigdy z literału.

### 1.1 Symulacja wrażliwości (`research/metric_sensitivity.py`)

Syntetyczne mecze generowane z priorów z literatury (sekcja 4.3 oraz stałe opisane w nagłówku
skryptu), nie z GT. Każdy wiersz psuje idealną
predykcję w jeden sposób; średnia z 6 meczów (~525 posiadań, ~119 przerw na mecz).

| uszkodzenie | spellF1 |
|---|---|
| idealnie | 1.000 |
| szum granic σ = 0,2 s / 0,5 s / 1 s / 2 s | 0.980 / 0.913 / 0.772 / 0.579 |
| stałe opóźnienie 0,5 s / 1 s | 0.885 / 0.696 |
| pominięte posiadania < 1 s / < 2 s / < 4 s | 0.965 / 0.870 / 0.646 |
| fałszywe 1-sekundowe „mignięcia” 0,5 / 1 / 3 na minutę gry | 0.932 / 0.872 / 0.686 |
| pominięte martwe piłki < 5 s / < 10 s | 0.993 / 0.963 |
| **nigdy nie mówimy `D`** (`X` na martwych) | **0.749** |
| martwa piłka wykryta 1 s / 3 s za krótko z każdej strony | 0.959 / 0.877 |
| zła drużyna po 10% / 30% wznowień | 0.980 / 0.933 |
| głosowanie większością ±1 s / ±2 s / ±5 s **na idealnym sygnale** | 0.963 / 0.857 / 0.547 |

Wnioski projektowe:

1. **Precyzja czasu przejść jest najważniejsza.** Przy σ = 1 s tracimy ~23 pkt. Docelowo granice
   z dokładnością < 0,5 s — to wymaga lokalnej, fizycznej detekcji momentu kontaktu (przyspieszenie
   piłki, zmiana kierunku), a nie wygładzania.
2. **Migotanie jest drogie**, ale **pomijanie krótkich posiadań też**. Kompromis rozstrzyga się na
   danych walidacyjnych z innych meczów, nie na intuicji.
3. **Trzeba przewidywać `D`** (kosztuje ~25 pkt, jeśli nie). Za to pominięcie bardzo krótkich przerw
   jest tanie — detektor martwej piłki może być konserwatywny dla krótkich przerw.
4. **Wygładzanie większością jest złym regularyzatorem** — zjada krótkie posiadania i przesuwa
   granice. Lepsze: dekodowanie segmentowe (sekcja 4.4).
5. Pomyłka drużyny po wznowieniu jest względnie tania — priorytet niższy niż timing.

Zastrzeżenie: rozkład długości posiadań w generatorze (mediana 6 s, ciężki ogon krótkich) to
przybliżenie z literatury; względna kolejność kosztów jest stabilna, wartości absolutne nie.

### 1.2 `X` — kiedy się opłaca

`X` = „powtórz poprzednią etykietę”. W środku posiadania (okluzja piłki) jest darmowe i
bezpieczniejsze niż zgadywanie. Na przejściu działa jak opóźnienie (≈ wiersz „stałe opóźnienie”).
Reguła: `X` tylko wtedy, gdy model jest niepewny **i** nie widzi przesłanek przejścia.
Uwaga: `macroF1` (tie-break) liczy `X` jako błąd — przy remisie warto mieć mało `X`.

---

## 2. Czym semantycznie jest ground truth

Z `CHALLENGE.md`: GT powstaje **mechanicznie ze skorygowanego trackingu** — `D` tam, gdzie
skorygowany tracking oznacza piłkę jako martwą, w przeciwnym razie drużyna, którą tracking wskazuje
jako „posiadającą”. Bez wygładzania.

To jest dokładnie format znany z feedów TRACAB/DFL/Metrica: per klatka `ball_status ∈ {alive, dead}`
oraz `ball_owning_team ∈ {home, away}` (Khaustov & Mozgovoy 2020; Bassek i in. 2025; PFF 2022).
Konsekwencje:

- W feedach TRACAB/DFL flaga właściciela zwykle **zmienia się na dotknięciu**, a nie dopiero po
  „kontrolowanym” posiadaniu. Czy GT tego meczu zachowuje się tak samo, sprawdzamy na danych
  z innych meczów z tą samą konwencją (IDSSE), nie na publicznych 5 minutach (17 posiadań).
- Flaga posiadania jest **ciągła podczas gry** (zawsze `H` albo `A`), więc w lotach piłki i
  zamieszaniu trwa ostatnia drużyna — tak jak modeluje to IDSSE/PFF.
- Przejście `alive → dead` ustawia operator/system w momencie przerwania (piłka za linią, gwizdek),
  a `dead → alive` w momencie wykonania stałego fragmentu.

**Właśnie dlatego dane z innych meczów z flagami `BallStatus`/`BallPossession` są tak cenne**: uczą
dokładnie tej konwencji.

---

## 3. Co mamy na wejściu (stan wiedzy bez bundla)

Bundla danych nie ma jeszcze w `data/`. Fakty z dokumentacji:

| asset | uwagi kluczowe dla metody |
|---|---|
| ATD tactical (199 319 klatek) | 46 slotów, brak ID, ~20,5 obiektu/klatkę, piłka 59,3% klatek ocenianych, `NaN` = brak |
| Smart Tagging (638) | `SET PIECES` z typem (bez drużyny przy wrzutach), fazy z drużyną; okna ~±3,5 s wokół wznowienia |
| Homografia tactical | 100% ważna; pozwala rzutować ATD na wideo i z powrotem |
| Wideo tactical | jedna szeroka kamera bez cięć — dobra do detekcji piłki i audio |
| Broadcast | inny zegar, cięcia, powtórki; mapowanie na tactical to osobny problem — niski priorytet |

Do zbadania na danych zaraz po rozpakowaniu (bez etykiet, więc dozwolone):
rozkład długości luk piłki, gdzie znikają (pole karne, powietrze, tłok), liczba obiektów w czasie,
częstość fragmentacji tracków, prędkości piłki (czy jest `z` — raczej nie), zgodność Smart Tagging
z przerwami w ATD.

---

## 4. Przegląd metod

### 4.1 Posiadanie z danych pozycyjnych (piłka widoczna)

- **Link & Hoernig 2017** (PLOS ONE, 60 meczów Bundesligi): wygładzanie RTS, start posiadania gdy
  najbliższy zawodnik < progu, opcjonalnie **lokalne maksimum przyspieszenia piłki ≥ 4 m/s²** jako
  detektor kopnięcia; koniec akcji przez predykcję toru piłki na 1 s; kontrola klasyfikowana siecią
  bayesowską (czas, prędkości, liczba rywali w 0,5–5 m). F ≈ 0,83–0,88.
- **Vidal-Codina i in. 2022** (Sports Engineering): deterministyczne drzewo decyzyjne. Strefa
  posiadania R_pz = 0,5–1 m, strefa pojedynku R_dz; zysk posiadania tylko gdy piłka **zmienia
  kierunek lub prędkość** w strefie zawodnika (odfiltrowuje przelatujące piłki); strata wymaga
  przemieszczenia piłki > ε. Martwa piłka z flagi **albo z braku piłki w trackingu**; stałe fragmenty
  rozpoznawane z ustawienia zawodników w przerwie. >90% detekcji większości zdarzeń.
  Implementacja referencyjna parametrów: `databallpy` (`add_individual_player_possession`).
- **Khaustov & Mozgovoy 2020** (Applied Sciences): histereza — przy kilku zawodnikach blisko piłki
  preferuj tego, kto miał ją w poprzedniej klatce; „ball far away” jako posiadanie podtrzymane.

Wniosek: przy widocznej piłce problemem nie jest „kto najbliżej”, tylko **kiedy dotknięcie zmienia
właściciela** — sygnałem jest zmiana wektora prędkości piłki w strefie zawodnika.

### 4.2 Posiadanie bez piłki (41% klatek)

- **Ball Radar — Kim i in., KDD 2023**: Set Transformer (niezmienniczy na permutację graczy — idealny,
  bo nie mamy ID) + hierarchiczny Bi-LSTM; najpierw posiadacz, potem tor piłki. Błąd < 3,7 m.
  Trening m.in. na danych Metrica.
- **PathCRF — Kim i in., 2026** (arXiv 2602.12080): graf dynamiczny zawodników + 4 węzły „aut”,
  w każdym kroku wybór jednej krawędzi (posiadanie / podanie), **CRF z maskowaniem niemożliwych
  przejść + Viterbi**. Trening na IDSSE (5 meczów!), **92,4% trafności drużyny per klatka**,
  udział posiadania 62,6% vs 62,1%. Działa tylko w fazach gry; między dostawcami spadek jakości —
  argument za augmentacją szumem.
- **Kolektywne cechy drużynowe** (bez sieci): drużyna z piłką rozciąga się, broniąca się kompaktuje
  i przesuwa w stronę piłki; centroidy, rozpiętość, linia obrony, ruch „w stronę bramki rywala”,
  pressing na najbliższym rywalu. Te cechy są wolnozmienne — dobre do wypełniania luk, słabe do
  precyzyjnego timingu.

Wniosek: model ball-free jako **emisja dla dekodera** w klatkach bez piłki, a piłka (gdy jest) do
ustalania momentu przejść.

### 4.3 Martwa piłka

- **Statystyki** (Siegle & Lames 2012, 16 meczów Bundesligi): ~108 przerw/mecz (40 wrzutów, 33
  rzuty wolne, 17 od bramki, 10 rożnych, 4 zmiany, 3 wznowienia od środka), gra 62% czasu,
  średnio **32,1 s gry i 18,7 s przerwy**; wrzut ~9 s, kontuzja ~82 s. Opta: LaLiga ma najkrótszy
  czas gry z top-5 lig.
- **Sygnały z trackingu**: piłka poza liniami (z marginesem na błąd homografii), piłka nieruchoma
  (stały fragment), gwałtowny spadek średniej prędkości drużyn, zawodnik przy linii bocznej
  z piłką (wrzut), ustawienie w polu karnym (rożny), bramkarz z piłką w polu bramkowym
  (od bramki), wszyscy na swoich połowach (od środka — Vidal-Codina 2022).
- **Smart Tagging `SET PIECES`**: każde wystąpienie = koniec przerwy w oknie ±kilka s; brak
  drużyny przy wrzutach → drużyna zawodnika przy piłce/linii w momencie wznowienia.
- **Audio — gwizdek**: pasmo ~3,3–4,5 kHz, STFT/Goertzel lub mały CNN na log-mel (UPC GTAV 2014;
  JDSAI). Gwizdek oznacza przerwę po faulu/spalonym, których geometria nie widzi. Działa lokalnie,
  bez wysyłania danych. Uwaga: najpierw sprawdzić, czy wideo taktyczne w ogóle ma ścieżkę audio.
- **Wideo — SoccerNet**: Action Spotting (17 klas, m.in. *Ball out of play*, *Throw-in*, *Corner*,
  *Foul*) i Ball Action Spotting 2024 (12 klas, m.in. *Out*, *Throw In*, *Free Kick*; zwycięzca
  T-DEED 73,4 mAP@1). Trenowane na innych meczach, więc dozwolone; ale to obraz z broadcastu —
  na szerokiej kamerze taktycznej będzie luka domenowa. Opcja drugiego rzutu.

### 4.4 Dekodowanie sekwencji

- **HSMM / segmental Viterbi** (Lea i in. 2016; Sarawagi & Cohen 2004): stan = {H, A, D},
  emisje per klatka z klasyfikatorów, **jawne rozkłady długości segmentów** (z danych zewnętrznych)
  i macierz przejść (np. `D→H/A` tylko przy wznowieniu, koszt `H↔A`). Złożoność O(T·D_max·C²) —
  przy 5 fps i D_max ~ 60 s to kilka sekund CPU.
- **CRF z maską przejść** (PathCRF) — to samo w wersji uczonej end-to-end.
- **Viterbi z minimalną długością stanu** (k-minimum-consecutive-states) — najprostsza wersja.
- Kluczowe: dekoder **nie przesuwa granic** tak jak okno większości; można dodatkowo „dociągnąć”
  każdą granicę do najbliższego zdarzenia fizycznego (maksimum przyspieszenia piłki).

### 4.5 Odzyskiwanie piłki z wideo

Wideo taktyczne + homografia 100% → detektor piłki (YOLO z kafelkowaniem, niski próg ufności +
filtr trajektorii; SoccerNet tracking 2023 3. miejsce) może uzupełnić część z 41% luk. Koszt:
dekodowanie ~200k klatek 1080p i GPU. Opłacalne dopiero, gdy wiemy, ile luk to okluzje „do odzyskania”,
a ile piłka w powietrzu/tłoku.

---

## 5. Dane zewnętrzne (inne mecze — dozwolone)

| zbiór | mecze | fps | etykiety per klatka | licencja / dostęp | uwagi |
|---|---|---|---|---|---|
| **IDSSE / Sportec Open DFL** (Bassek i in. 2025) | 7 (Bundesliga 1/2, 22/23) | 25 | **`BallStatus` (0/1) + `BallPossession` (1 home / 2 away)** + zdarzenia | CC-BY 4.0; `kloppy.sportec.load_open_tracking_data` | ta sama semantyka co GT; TRACAB, skorygowane |
| **PFF FC WC 2022** | 64 | ~30 | `ball_state`, `ball_owning_team` + zdarzenia | darmowe po rejestracji; `kloppy.pff` | broadcast tracking — bliżej naszego szumu |
| **SkillCorner Open Data** | 10 (A-League 24/25) | 10 | `possession.group` + fazy gry; flagi `is_detected` | publiczne repo SkillCorner | broadcast, ekstrapolowane pozycje |
| **Metrica Sample Data** | 3 | 25 | zdarzenia (podania, BALL OUT, SET PIECE) → etykiety wyprowadzalne | publiczne repo Metrica (sprawdzić warunki) | **ten sam dostawca**, gra 3 w formacie EPTS jak nasze ATD |

Uwaga regulaminowa: używamy wyłącznie danych **o innych meczach**; w dokumencie technicznym trzeba
podać źródło, kto etykietował i ile.

### 5.1 Jak zamknąć lukę domenową

Dane zewnętrzne są czyste. Nasze ATD nie. Przed treningiem/walidacją przepuszczamy dane zewnętrzne
przez **symulator szumu ATD** skalibrowany na *nieetykietowanych* statystykach naszego meczu:

- usuwanie piłki w seriach o rozkładzie długości jak w naszym ATD, z wyższym prawdopodobieństwem
  w tłoku/polu karnym/przy dużej prędkości piłki (lot);
- usuwanie ID: zawodnicy → anonimowe sloty drużynowe, losowe cięcie tracków i zamiany tożsamości;
- gubienie zawodników (średnio ~20,5 obiektu), duplikaty, jitter pozycji;
- normalizacja do 0–1 i wymiarów boiska.

---

## 6. Proponowana architektura

```
ATD ──► czyszczenie (RTS/Kalman, sklejanie tracków, interpolacja krótkich luk piłki)
     ├─► cechy per klatka: odległości piłka–zawodnicy, zmiana wektora prędkości piłki,
     │   piłka poza boiskiem / nieruchoma, prędkości drużyn, kształt drużyn, ...
     ├─► model alive/dead  ─┐
     ├─► model H/A (piłka) ─┼─► emisje P(H), P(A), P(D) per klatka (+ pewność)
     └─► model H/A (ball-free, set-model) ┘
Smart Tagging ──► kotwice wznowień, priory drużyny w fazach
(opcj.) audio ──► gwizdek;  (opcj.) wideo ──► dodatkowe detekcje piłki
                                   │
                                   ▼
          HSMM / segmental Viterbi (rozkłady długości, maska przejść)
                                   │
                                   ▼
          dociąganie granic do zdarzeń fizycznych ──► polityka X ──► CSV + walidator
```

Kolejność budowy (każdy krok mierzony na zewnętrznych meczach z oficjalną metryką):

1. **v0 — fundament**: czytnik ATD, poprawna obsługa połówek (w tym `D` przed wznowieniem 2. połowy),
   rozszerzony baseline z histerezą zamiast okna większości. Cel: >> 0,24.
2. **v1 — reguły + dekoder**: detektor martwej piłki z geometrii + Smart Tagging; zmiana posiadania
   wg Vidal-Codina (zmiana wektora piłki w strefie zawodnika); HSMM z rozkładami długości z IDSSE.
3. **v2 — modele uczone**: gradient boosting / mały Set Transformer na cechach per klatka,
   trenowany na IDSSE (+PFF) z augmentacją szumem; ball-free model do luk; pseudo-etykiety na
   naszym meczu (self-training) z klatek o wysokiej pewności.
4. **v3 — dodatkowe modalności** (jeśli czas pozwoli): gwizdek z audio, detektor piłki z wideo.

---

## 7. Protokół walidacji (zgodny z regulaminem)

1. **Główny sygnał do decyzji**: leave-one-match-out na IDSSE (i PFF), dane zaszumione symulatorem,
   oficjalna `possession_metrics.evaluate`, `--bootstrap` dla przedziałów. Tu wolno tuningować.
2. **Publiczne 5 minut**: tylko test poprawności pipeline'u (wyrównanie klatek, format, sens
   etykiet). Nie wybieramy na nich wariantów ani progów — dokumentacja pokazuje, że wybierają źle.
3. **Kontrole spójności na naszym meczu bez etykiet**: liczba posiadań, średnia długość, udział gry
   vs priory z literatury; liczba przerw vs liczba `SET PIECES` w Smart Tagging; brak `X` w złym
   miejscu; `validate_submission.py` na finalnym pliku.
4. **Odtwarzalność**: stałe ziarna, przypięte wersje, jedno polecenie regenerujące identyczny plik.

---

## 8. Zgodność z regulaminem — checklista

- [ ] Żadnych ręcznych etykiet ani korekt na tym meczu (także „na przejściach”).
- [ ] Żadnych reguł/stałych przypiętych do konkretnych klatek tego meczu (poza publicznymi granicami
      połówek z `games.py`).
- [ ] Publiczne GT nie jest używane do treningu ani wyboru parametrów.
- [ ] Brak danych zdarzeniowych/skorygowanego trackingu **tego** meczu (Opta, StatsBomb, LaLiga…).
- [ ] Nie wysyłamy danych/wideo meczu do hostowanych LLM-ów.
- [ ] `data/` w `.gitignore`; nie publikujemy pochodnych odtwarzających dane.
- [ ] Repo regeneruje identyczny plik; dokument opisuje dane zewnętrzne (źródło, kto, ile).
- [ ] Pseudo-etykiety tylko algorytmiczne (żadna osoba nie decyduje o żadnej etykiecie).

---

## 9. Ryzyka i otwarte pytania

- **Luka domenowa** clean → noisy: główne ryzyko podejścia uczonego; mitigacja: symulator szumu,
  walidacja na PFF (broadcast tracking).
- **Różnice konwencji między dostawcami** w momencie przełączania flagi (dotknięcie vs kontrola,
  moment „dead”): PathCRF raportuje spadek przy zmianie dostawcy. Metrica sample (ten sam dostawca)
  może pomóc skalibrować konwencję.
- **LaLiga** ma krótszy czas gry niż Bundesliga — priory długości przerw mogą wymagać korekty
  (np. z PFF/innych lig), nie z GT.
- **Koszt obliczeń** wideo/audio — decyzja po pomiarze, ile luk piłki to okluzje możliwe do odzyskania.
- Pytanie do organizatorów (issue), jeśli będzie potrzebne: czy użycie modeli SoccerNet i danych PFF
  jest OK w świetle „organisers can access and inspect it” (PFF wymaga rejestracji).

---

## 10. Bibliografia

- Bassek M., Rein R., Weber H., Memmert D. (2025). *An integrated dataset of spatiotemporal and event
  data in elite soccer.* Scientific Data 12, 195. doi:10.1038/s41597-025-04505-y
- Kim H. i in. (2023). *Ball Trajectory Inference from Multi-Agent Sports Contexts Using Set
  Transformer and Hierarchical Bi-LSTM.* KDD 2023. arXiv:2306.08206
- Kim H. i in. (2026). *PathCRF: Ball-Free Soccer Event Detection via Possession Path Inference from
  Player Trajectories.* arXiv:2602.12080
- Khaustov V., Mozgovoy M. (2020). *Recognizing Events in Spatiotemporal Soccer Data.* Applied
  Sciences 10(22), 8046.
- Lea C., Reiter A., Vidal R., Hager G. (2016). *Segmental Spatiotemporal CNNs for Fine-grained
  Action Segmentation.* ECCV. arXiv:1602.02995
- Link D., Hoernig M. (2017). *Individual ball possession in soccer.* PLOS ONE 12(7): e0179953.
- UPC GTAV (2014). *The Importance of Audio Descriptors in Automatic Soccer Highlights
  Generation.* (detekcja gwizdka, pasmo 3,5–4,5 kHz, algorytm Goertzla).
- *Detecting Referee Whistle Sounds in Soccer Videos Using Machine Learning Models.* Journal of
  Data Science and Artificial Intelligence (ISODS), pasmo 3,3–4,2 kHz.
- PFF FC (2023/2025). *FIFA World Cup 2022 broadcast tracking + event data* (kloppy `pff`).
- Siegle M., Lames M. (2012). *Game interruptions in elite soccer.* J. Sports Sciences 30(7), 619–624.
- SkillCorner Open Data (A-League 2024/25), SoccerNet Action Spotting / Ball Action Spotting 2024
  (Cioppa i in., arXiv:2409.10587), T-DEED (Xarles i in.).
- Vidal-Codina F., Evans N., El Fakir B., Billingham J. (2022). *Automatic event detection in
  football using tracking data.* Sports Engineering 25, 18. arXiv:2202.00804
