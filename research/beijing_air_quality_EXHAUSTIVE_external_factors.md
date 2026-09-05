# Beijing Air Quality — Exhaustive External-Factor Audit (2013–2017)

Goal: before any feature engineering or modelling, have a complete, **evidence-graded** list of every plausible external driver of PM10/PM2.5/SO2/NO2/CO/O3/meteorology in this dataset. Every item below is tagged:

- 🟢 **Confirmed** — checked directly against `train.csv`/`test_1_.csv`, effect is clear and consistent
- 🟡 **Partially confirmed / mixed** — real, documented phenomenon, but the data shows an inconsistent or confounded signal (said honestly rather than forced to fit)
- ⚪ **Plausible but untestable here** — a real mechanism, but this dataset alone can't isolate it (needs external data)
- 🔧 **Data artifact** — not a real-world atmospheric event, but will masquerade as one if not handled

---

## A. Structural / secular (multi-year, changes the baseline itself)

### 🟢 A1. China's national "war on pollution" — progressive, verified year-over-year decline
Beijing's 2013 Clean Air Action Plan (following the catastrophic Jan 2013 "Airpocalypse", just before this dataset begins) mandated PM2.5 cuts, coal-to-gas boiler conversion, desulfurization retrofits, and stricter vehicle emissions standards, phased in 2013–2017.

Checked using only Mar–Aug of each year (to hold season constant) so the trend isn't just "less heating pollution":

| Year (Mar–Aug avg) | SO2 | NO2 | PM2.5 | CO | O3 |
|---|---|---|---|---|---|
| 2013 | 19.0 | 48.8 | 80.0 | 1098 | 73.6 |
| 2014 | 14.1 | 48.0 | 76.0 | 957 | 84.2 |
| 2015 | 9.0 | 38.8 | 63.2 | 886 | 86.0 |
| 2016 | 7.8 | 38.8 | 64.2 | 849 | 82.8 |

**SO2 fell 59%, NO2 fell 20%, PM2.5 fell 20%, CO fell 23%** over just 4 years — this is a real, policy-driven non-stationary trend, not noise.

![YoY trend](eda_charts/chart6_yoy.png)

**Modelling implication:** `year` (or a monotonic trend feature) matters independently of season — a model trained naively on 2013–2014 data and evaluated on 2016 data will systematically over-predict SO2/PM2.5 if it doesn't have some way to represent this decline.

### 🟢 A2. The "ozone rebound" — O3 rising while everything else falls
Same table above: **O3 rose ~15%** from 2013 to 2015 even as PM2.5/SO2/NO2 fell. This is a well-documented side effect of NOx-reduction policy in Chinese megacities during 2013–2017: less NOx means less "titration" (NO scavenging O3 at the surface), so peak ozone actually increases even as overall air quality improves on other measures. This is *not* a data error — it's a real atmospheric-chemistry consequence of the same policy driving A1.

**Modelling implication:** don't assume all pollutants move together over time — O3 needs to be modelled as decoupled from (even inversely related to) the general "pollution improving" trend.

### ⚪ A3. Beijing–Tianjin–Hebei industrial relocation / coal-to-gas conversion (mechanism behind A1)
The underlying policy mechanism — closing/relocating steel, cement and coal-boiler capacity out of Beijing into (or out of) neighboring Hebei, and converting ~hundreds of thousands of residential coal boilers to gas — is well documented but not separately verifiable as its own signal distinct from A1 in this dataset; it's the *cause*, A1 is the *effect* you can measure.

### ⚪ A4. El Niño 2015–2016 (one of the strongest on record)
A strong El Niño peaked in winter 2015–16, which is known to influence East Asian winter monsoon strength and can suppress the cold, dry northerly winds that normally flush pollution out of the Beijing basin — potentially compounding the severity of the Dec 2015 red-alert haze events (see B5). This is a plausible meteorological confound for the `TEMP`/`PRES`/`WSPM` columns in that window, but isolating El Niño's specific contribution would require external climate-index data this dataset doesn't include.

---

## B. Episodic / calendar-anchored events (days to ~2 weeks)

### 🟢 B1. Lunar New Year fireworks
Confirmed in the original pass — SO2 at 248% (2014), 184% (2015), 134% (2016) of the full-period average during Spring Festival windows, weakening year over year as firework restrictions tightened.

### 🟢 B2. APEC Blue (Nov 1–13, 2014)
Confirmed: SO2 49%, PM10 63%, PM2.5 54%, CO 75%, NO2 89% of average — coordinated multi-province shutdown for the summit.

### 🟢 B3. Parade Blue (Aug 20 – Sep 4, 2015)
Confirmed: SO2 16%, PM10 29%, PM2.5 25%, CO 51%, NO2 44% of average — the single largest and longest shutdown in the dataset, for the WWII 70th-anniversary parade.

### 🟢 B4. Spring dust storms (Mar–May), Gobi/Mongolian plateau
Confirmed via PM10:PM2.5 ratio spikes (4–6x on days like 2015-02-22, 2016-03-05, 2015-03-20, 2014-04-10) clustering in Mar–May — a physically distinct mechanism (coarse mineral dust) from combustion haze.

### 🟢 B5. Winter "red alert" haze episodes — now explicitly dated, including in the test set
Beijing's 4-tier alert system (introduced 2013) issued its first-ever **red alerts** in winter 2015:

| Episode | PM2.5 vs. average |
|---|---|
| Red Alert #1 (Dec 8–10, 2015) | **247%** |
| Red Alert #2 (Dec 19–22, 2015) | **268%** |

Checking the **test set** for the same phenomenon turns up two more, more severe episodes that fall inside your holdout period:

| Date | Daily avg PM10 |
|---|---|
| 2016-12-20/21 | 385 / 411 |
| 2016-12-31 – 2017-01-01 | 324 / **496** |

These are genuine, extreme, real-world haze events (consistent with publicly reported Beijing red alerts in Dec 2016 and a notably bad New Year's Day 2017 smog episode) sitting right in your test window — a model that hasn't seen anything like this magnitude in training could badly under-predict here.

### 🟡 B6. "Two Sessions" (NPC/CPPCC annual meetings, early March)
Widely reported in Chinese media as producing a "Two Sessions Blue" via temporary factory/traffic curbs in Beijing and Hebei. **Checked directly and the signal does not hold up cleanly**: comparing the session window to the rest of March each year, SO2 was actually *higher* during the sessions in all three years tested (e.g., 41 vs. 31 µg/m³ in 2014). Likely explanations: restrictions are reportedly concentrated more in surrounding Hebei industrial areas than in Beijing itself, the window overlaps the tail of heating season and the start of dust-storm season (both pushing levels up independently), and/or the effect is simply too small to see against natural March variability. **Verdict: real policy, unconfirmed signal — don't hard-code this as a feature without better evidence.**

### 🟡 B7. National Day "Golden Week" (Oct 1–7)
Reduced industrial/commuter activity during the holiday is plausible, but checked across 3 years the result is inconsistent — 2014 shows a real drop (SO2 29%, CO 84%), but 2013 and 2015 show *increases* (driven by unrelated stagnant-weather episodes that happened to coincide). **Verdict: real but weak/overwhelmed by weather noise at daily-aggregate level.**

### 🟡 B8. Qingming Festival (early April, tomb-sweeping, ritual paper-burning)
A smaller-scale analog to fireworks (localized combustion smoke). Checked across 2014–2016: no consistent city-wide signal (128% of average one year, 60-63% the other two). Likely too localized (specific cemeteries, early-morning hours) and too diluted across 12 city-wide stations to register in daily averages the way fireworks (a truly city-wide, synchronized event) do. **Verdict: plausible micro-effect, not detectable at this aggregation level.**

### ⚪ B9. Beijing Daxing airport construction (broke ground Dec 2014) and other major infrastructure works
Large construction projects are a known local dust/PM10 source. This would show up as a **station-specific** anomaly (stations nearest the site) rather than a city-wide one, and wasn't tested here — worth a per-station deep-dive if you need station-level rather than city-wide accuracy.

### ⚪ B10. G20 Hangzhou (Sept 2016) and other non-Beijing national events
China imposed regional shutdowns around Hangzhou/Yangtze Delta for G20 in Sept 2016 — geographically distant from Beijing, so unlikely to have a direct effect here, but flagged since it falls inside the `test_1_.csv` window and any unexplained late-2016 anomaly should be checked against it before being attributed to something Beijing-specific.

---

## C. Meteorological transport & physical mechanisms (every hour, every day)

### 🟢 C1. Wind direction — regional transport is one of the strongest single effects in the dataset
Southerly/southeasterly winds funnel pollution from Hebei/Tianjin/Shandong's industrial belt into the Beijing basin; northerly/northwesterly winds bring clean continental/Siberian air:

![Wind direction effect](eda_charts/chart5_winddir.png)

PM2.5 averages **~100 µg/m³ under ESE/E/SE winds vs. ~49–55 µg/m³ under NW/NNW winds** — essentially a 2x swing from wind direction alone, independent of season.

**Modelling implication:** `wd` (categorical, or decomposed into a "southerly component") is likely one of the single highest-value features available, on par with season.

### 🟢 C2. Precipitation washout
Rain scavenges particulates out of the air. Directly confirmed: mean PM2.5 next-hour is **78.9 µg/m³** on non-rain hours vs. **58.9 µg/m³** on rain hours.

### 🟢 C3. Nocturnal boundary-layer collapse + calmer overnight wind
Already covered in the original pass: PM2.5 peaks overnight (00:00) and troughs mid-afternoon (15:00), tracking wind speed (r = −0.28) and the daily heating/mixing cycle, not emissions timing.

### 🟢 C4. Diurnal traffic (NO2/CO) and photochemistry (O3)
Already covered: rush-hour NO2/CO elevation, midday/early-afternoon O3 peak driven by solar radiation.

### 🟢 C5. Weekday/weekend "ozone weekend effect"
Checked directly: NO2 is essentially flat between weekdays and weekends (49.46 vs. 49.57 µg/m³) — consistent with Beijing's weekday-only license-plate rotation restriction being offset by more leisure/weekend driving. More interestingly, **O3 is slightly *higher* on weekends** (61.95 vs. 60.39) despite similar NOx — this is the well-documented "ozone weekend effect" seen in polluted cities worldwide: less traffic-NOx on Sundays means less NO available to titrate away O3, so ozone doesn't fall in step with primary traffic pollutants (and can even rise slightly).

---

## D. Spatial / station-level heterogeneity

### 🟢 D1. Urban-core vs. suburban/rural station gradient
The 12 stations are not interchangeable — averaging across them (as the charts above do) hides a real, consistent gradient:

| Station tier | Stations | Avg PM2.5 |
|---|---|---|
| Urban core | Dongsi, Nongzhanguan, Wanshouxigong, Wanliu, Gucheng, Aotizhongxin, Guanyuan, Tiantan | 80–84 |
| Peri-urban | Shunyi | 78 |
| Suburban/rural | Changping, Huairou | 69–70 |
| Rural background | Dingling | **65** (lowest of all 12, across every pollutant) |

Dingling (in the mountains north of Beijing) is consistently the cleanest site on every pollutant — it functions as a "background" reference station rather than a typical urban-exposure site. `station` (or an engineered urban/rural or distance-from-center feature) meaningfully changes the baseline independent of time.

---

## E. Data-collection artifacts (not atmospheric events — will bias a model if untreated)

### 🔧 E1. CO sensor rollout gap, 2013
Monthly CO missingness spikes to 20–28% in Oct–Dec 2013 versus <2% from 2014 onward — the monitoring network was still being installed/calibrated in year one. Don't let a blanket imputation strategy quietly treat 2013 CO as "normal but sparse."

### 🔧 E2. Sensor/reporting ceilings
`CO` caps at exactly 10,000 (241 rows), `PM10`/`PM2_5_next_hour` cap at exactly 999 (1–2 rows). These are instrument/reporting maxima — treat as censored ("≥ x"), not exact readings, especially since they cluster inside the most extreme real haze events (B5) where the true concentration may exceed the cap.

---

## F. Practical checklist before modelling

1. **Season / heating-season flag** and **month** — dominant driver (A1, but seasonal not secular).
2. **`year` or a trend term** — captures the A1 secular decline; without it, a model calibrated on 2013–14 will overshoot 2016 pollution levels.
3. **`wd` (wind direction)** — arguably as important as season; consider collapsing to a "southerly transport index."
4. **`WSPM` and `RAIN`** — dispersion/washout mechanisms (C1–C3).
5. **`hour`** (cyclically encoded) — diurnal boundary-layer + traffic + photochemistry (C3, C4).
6. **`station`** or an urban/rural grouping — real, persistent baseline differences (D1).
7. **Calendar flags for CNY, APEC (train only), Parade (train only), and the Dec 2015/Dec 2016–Jan 2017 red-alert windows** — genuine step-changes no weather variable can explain; better to flag than expect a model to infer from first principles.
8. **PM10 − PM2.5 (or their ratio)** as an engineered feature — separates dust events from combustion haze (B4 vs. B5), which otherwise look identical in a raw PM10 spike.
9. **Treat "Two Sessions," Golden Week, and Qingming as *not* reliable features** (B6–B8) — resist the temptation to hard-code these just because they're famous; the data doesn't support them at this aggregation level.
10. **Clip/flag sensor-ceiling values** (E2) and **be cautious with pre-2014 CO** (E1) rather than trusting them at face value.
11. **Test set (`test_1_.csv`) covers Aug 2016–Feb 2017 only** — no spring, no full summer, but it *does* contain a severe Dec 2016–Jan 2017 haze episode (B5) and the Jan 2017 CNY. Validate on a similarly-shaped time slice, not random k-fold.
