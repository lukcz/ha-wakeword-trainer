# ha-wakeword-trainer

Repo do prostego uruchamiania treningu na Linuxie / WSL2 bez ręcznego składania środowiska i bez ręcznego kompletowania wszystkich datasetów na start.

Obecnie repo ma dwa tryby:

- `vad`: trening klasyfikatora mowa vs brak mowy
- `wakeword`: trening modelu wake word

Całość jest spięta przez:

- `setup_environment.sh` - instaluje systemowe zależności, klonuje repo jeśli trzeba i tworzy `.venv`
- `train.sh` - prosty launcher do uruchomienia treningu
- `train_vad_full.sh` - preset do pełnego VAD
- `train_wakeword_full.sh` - preset do pełnego wake word
- `train_wakeword.py` - pipeline krok po kroku

## Szybki start przez `wget`

Jeśli chcesz zacząć od zera na WSL2 / Linux:

```bash
wget -O setup_environment.sh https://raw.githubusercontent.com/lukcz/ha-wakeword-trainer/main/setup_environment.sh
chmod +x setup_environment.sh
./setup_environment.sh
```

Po setupie:

```bash
cd ~/ha-wakeword-trainer
./train.sh vad
```

Albo jeszcze krócej:

```bash
cd ~/ha-wakeword-trainer
./train_vad_full.sh
```

Albo dla wake word:

```bash
cd ~/ha-wakeword-trainer
./train.sh wakeword
```

Albo:

```bash
cd ~/ha-wakeword-trainer
./train_wakeword_full.sh
```

## Co repo robi samo

- instaluje `ffmpeg`, `python3`, `git`, `wget`, `venv`
- tworzy lokalne środowisko `.venv`
- instaluje paczki z `requirements.txt`
- pobiera dane tła i walidacyjne potrzebne do treningu
- dla VAD:
  - jeśli nie ma negatywnych danych, pipeline pobiera publiczne subsety tła / muzyki / noise
  - jeśli nie ma pozytywnych danych mowy, generuje awaryjny polski fallback speech przez `edge-tts`

To oznacza, że `./train.sh vad` powinno wystartować bez ręcznego wrzucania datasetów. Własne dane nadal są lepsze, ale repo nie blokuje się już na pustym `data/`.

## Najprostsze komendy

Pełny trening VAD:

```bash
./train.sh vad
```

Pełny trening wake word:

```bash
./train.sh wakeword
```

Wznowienie od wybranego kroku:

```bash
./train.sh vad --from augment
```

Lista kroków pipeline:

```bash
python train_wakeword.py --list-steps
```

## Własne dane do VAD

Domyślna konfiguracja to [configs/polish_vad.yaml](/D:/Github/ha-wakeword-trainer/configs/polish_vad.yaml).

Jeśli masz własne datasety, możesz je podpiąć bez edycji pliku YAML:

```bash
./train.sh vad \
  --positive-datasets mc_speech,bigos \
  --negative-datasets no_speech,dinner_party,musan,fma \
  --dataset-path mc_speech=/data/polish/mc_speech \
  --dataset-path bigos=/data/polish/bigos \
  --dataset-path no_speech=/data/no_speech
```

Pozytywne dane:

- ludzka mowa
- wielu mówców
- różne mikrofony i pokoje

Negatywne dane:

- cisza
- szum tła
- HVAC
- muzyka
- ambient

## Własny wake word

Przykładowa konfiguracja jest w [configs/wakeword_example.yaml](/D:/Github/ha-wakeword-trainer/configs/wakeword_example.yaml).

Najprościej:

```bash
cp configs/wakeword_example.yaml configs/moj_wakeword.yaml
```

Potem zmień:

- `model_name`
- `target_phrase`
- `custom_negative_phrases`

I uruchom:

```bash
./train.sh wakeword --config configs/moj_wakeword.yaml
```

## Gdzie są wyniki

Pipeline zapisuje artefakty do:

- `output/` - pliki pośrednie
- `export/` - gotowe modele do użycia

W repo są też przykładowe gotowe artefakty:

- `hej_zgredek.tflite`
- `hej_zgredek.json`

To są przykładowe pliki wynikowe, nie wejście do treningu.

## Najważniejsze pliki

- [train.sh](/D:/Github/ha-wakeword-trainer/train.sh) - najprostszy punkt wejścia
- [train_vad_full.sh](/D:/Github/ha-wakeword-trainer/train_vad_full.sh) - pełny preset VAD
- [train_wakeword_full.sh](/D:/Github/ha-wakeword-trainer/train_wakeword_full.sh) - pełny preset wake word
- [setup_environment.sh](/D:/Github/ha-wakeword-trainer/setup_environment.sh) - bootstrap na WSL2 / Linux
- [train_wakeword.py](/D:/Github/ha-wakeword-trainer/train_wakeword.py) - pipeline
- [configs/polish_vad.yaml](/D:/Github/ha-wakeword-trainer/configs/polish_vad.yaml) - VAD
- [configs/wakeword_example.yaml](/D:/Github/ha-wakeword-trainer/configs/wakeword_example.yaml) - wake word

## Typowy flow

1. Uruchamiasz `./setup_environment.sh` albo pobierasz go przez `wget`.
2. Wchodzisz do repo.
3. Startujesz `./train.sh vad` albo `./train.sh wakeword`.
4. Jeśli pipeline padnie, wznawiasz od kroku `--from ...`.
5. Gotowy model bierzesz z `export/`.

## Uwagi praktyczne

- Trening najlepiej odpalać w Linuxie / WSL2, nie natywnie z PowerShell.
- Pierwsze uruchomienie może pobrać kilka GB danych.
- Fallback speech dla VAD jest wygodny, ale jakość końcowego modelu będzie lepsza na prawdziwych nagraniach.
- Jeśli chcesz tylko sprawdzić stan pipeline, użyj `--verify-only`.
