# PetBionic Analyser

Ferramentas Python para limpeza e visualização dos CSVs gerados pela prótese
canina petBionic (GUI Qt com gráficos, features de marcha e modelo 3D).

---

## Estrutura do repositório (layout `src`)

```
PetBionicAnalyser/
├── pyproject.toml                  — metadados do pacote + entry point
├── requirements.txt                — dependências (também no pyproject)
├── run.sh                          — launcher (cria venv fora do iCloud na 1.ª vez)
├── README.md
├── assets/                         — recursos
│   ├── app_icon.png                — ícone da app (cão + prótese)
│   └── app_icon_orig.png
├── scripts/                        — utilitários
│   ├── make_desktop_app.sh         — cria a .app clicável no Desktop
│   ├── make_feature_figures.py     — gera as figuras da tese
│   └── ble_time_sync_test.py
└── src/
    └── petbionic_analyser/         — o pacote
        ├── __init__.py
        ├── __main__.py             — `python -m petbionic_analyser`
        ├── analyzer.py             — GUI principal (tabs, 3D, calibração)
        └── cleaner.py              — limpeza de CSVs do firmware antigo
```

Estado de runtime (gerado, fora do controlo de versões): `imu_calibration_R.json`,
`model_orientation.json`, `calib_runs.json` na raiz; e a `.app` construída.

---

## Instalação / arranque

```bash
# Na 1.ª execução cria o venv (em ~/Library/Caches, fora do iCloud) e instala tudo:
bash run.sh

# Com um ficheiro pré-carregado:
bash run.sh /caminho/para/run.csv
```

App clicável no Desktop (com ícone):
```bash
bash scripts/make_desktop_app.sh        # cria "PetBionic Analyser.app" + atalho no Desktop
```

Execução directa do pacote (dev):
```bash
PYTHONPATH=src python -m petbionic_analyser
```

---

## 1. `csv_cleaner.py` — Limpeza de CSVs

Corrige dois bugs presentes nos CSVs gravados com o firmware anterior à versão de Maio 2026:

| Bug | Causa | Correcção |
|-----|-------|-----------|
| `load_cell_est_kg` congelado em blocos de 20 linhas | `rawToKg()` fazia uma 2.ª leitura bloqueante do HX711; na maioria das vezes estava em timeout → kg não actualizava | Recalcula kg por linha: `(raw − offset) / factor` |
| `load_cell_raw` ≈ 2500 intercalado com leituras reais | `readRaw()` sem `return` no caminho de timeout → UB do C++ devolve valor de registo ≈ 2500 | Remove linhas com raw no cluster de lixo detectado automaticamente |

**Calibração padrão** (derivada dos dados de aquecimento):

```
offset = −16 000 counts   (cluster sem carga ≈ −13 918; kg congelado = 0.112 kg → −13 918 − 0.112 × 18 570 ≈ −15 998)
factor = 18 570 counts/kg (kHx711CalibrationFactor no firmware)
```

### Uso

```bash
# Processa toda a pasta TestData (cria ficheiros *_cleaned ao lado dos originais):
PYTHONPATH=src python -m petbionic_analyser.cleaner

# Pasta ou ficheiro específico:
PYTHONPATH=src python -m petbionic_analyser.cleaner TestData/Round1dia28

# Ajustar calibração:
PYTHONPATH=src python -m petbionic_analyser.cleaner --offset -15500 --factor 18570

# Substituir ficheiros originais (tem backup!):
PYTHONPATH=src python -m petbionic_analyser.cleaner --in-place

# Saída numa pasta separada:
PYTHONPATH=src python -m petbionic_analyser.cleaner --output-dir TestData/cleaned

# Ajuda completa:
PYTHONPATH=src python -m petbionic_analyser.cleaner --help
```

### Formato dos ficheiros de saída

Idêntico ao original, com:
- Linhas de lixo removidas (≈ 21% do total nos dados de Maio 2026)
- `load_cell_est_kg` recalculado por linha

### Convenção de nomes dos CSVs

Os ficheiros são renomeados para `YYYYMMDD_HHhMM_runNN.csv` (sortável = ordem temporal).
Quando dois runs partilham o mesmo minuto, o segundo inclui os segundos: `HHhMMmSS`.

---

## 2. App — Visualizador Interactivo (`petbionic_analyser`)

Interface gráfica Qt com 6 tabs e browser de ficheiros lateral.

```bash
# Lança o visualizador (venv já criado):
bash run.sh

# Com ficheiro pré-carregado:
bash run.sh TestData/Round1dia28_cleaned/AndamentoTestes28/20260528_12h32_run01.csv
```

### Browser de ficheiros (painel esquerdo)

- Árvore expansível com **Downloads** e **Test Data** como raízes
- Duplo-clique num CSV → carrega em todas as tabs
- Botão **Actualizar** restaura o estado de expansão das pastas abertas
- Botão **Outro…** → diálogo de ficheiro padrão

### Tabs disponíveis

#### Tab "Main" (interface principal)

| Zona | Conteúdo |
|------|----------|
| Gráfico superior | Roll / Pitch / Yaw — ângulos **unwrapped** (sem descontinuidades de ±360°) |
| Gráfico intermédio | Força (kg) — sincronizado com o cursor |
| Modelo 3D | Prótese em forma de taco de golfe, actualizado pelo slider |
| Slider | Scrubber de tempo — arrastar = ver orientação em cada instante |
| Controlos | ▶ Play / ⏸ Pausa · velocidades 0.5× 1× 2× 5× |

**Interacção nos gráficos** (RPY e Kg): clica + arrasta → cursor move, modelo 3D actualiza em tempo real.

**Unwrapping dos ângulos**: `np.unwrap` remove os saltos artificiais quando roll/pitch/yaw atravessa ±180°, tornando os gráficos de linha contínuos.

**Modelo 3D sem "teleports"**: usa ângulos suavizados (média rolante) em vez dos raw, eliminando artefactos de ruído de amostragem.

#### Tab "Célula de Carga"

- Scatter de `load_cell_est_kg` (azul, eixo esquerdo) + `load_cell_raw` (laranja, eixo direito)
- Toggles on/off por traço
- **Botão Linha/Pontos**: alterna entre scatter e curva suavizada (média rolante)
- SpanSelector: arraste horizontal = zoom na janela temporal; duplo-clique = reset

#### Tabs "Acelerómetro", "Giroscópio", "Magnetómetro"

- Scatter dos 3 eixos (vermelho/verde/roxo) + **Módulo** `√(x²+y²+z²)` (laranja)
- Toggles individuais por eixo e para o módulo
- **Botão Linha/Pontos**: alterna entre scatter e curva suavizada
- SpanSelector para zoom; duplo-clique = reset

#### Tab "Orientação — Análise"

- Roll/Pitch/Yaw com **unwrap** aplicado
- SpanSelector zoom + duplo-clique reset
- Botão Linha/Pontos (curva suavizada)
- Toolbar matplotlib (pan, zoom rect, guardar imagem)

### Heurística de fase de marcha

| Condição | Fase |
|----------|------|
| `load_cell_est_kg > 0.15 kg` | Apoio (stance) — pé em contacto com o solo |
| `load_cell_est_kg ≤ 0.15 kg` | Balanço (swing) — prótese no ar |

Valores ligeiramente negativos são normais durante o swing (inércia da perna em tracção).

---

## Formato dos CSVs

```
sample_us, time_of_day, batt_v, load_cell_raw, load_cell_est_kg,
imu_ax, imu_ay, imu_az, imu_gx, imu_gy, imu_gz,
imu_mx, imu_my, imu_mz, roll_deg, pitch_deg, yaw_deg
```

| Coluna | Unidade | Notas |
|--------|---------|-------|
| `sample_us` | µs (micros ESP32) | Timestamp principal — usa-se `(sample_us − sample_us[0]) / 1e6` para tempo relativo |
| `load_cell_raw` | counts HX711 | 16-bit, ±2× full-scale. Nos dados limpos: sem cluster de lixo |
| `load_cell_est_kg` | kg | Nos dados limpos: recalculado por amostra |
| `imu_ax/ay/az` | LSB (±2g, 16384 LSB/g) | Sem calibração de montagem aplicada (ver `imu_mounting_cal`) |
| `roll_deg / pitch_deg / yaw_deg` | graus | Filtro complementar do firmware; não têm em conta a orientação de montagem |

---

## Dependências

```
pandas   ≥ 2.0
matplotlib ≥ 3.8
PyQt6    ≥ 6.6
numpy    ≥ 1.26
```
