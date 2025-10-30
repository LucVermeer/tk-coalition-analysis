# Verkiezingsprogramma Downloader

Script om alle beschikbare verkiezingsprogramma's voor de Tweede Kamerverkiezingen 2025 te downloaden vanaf [verkiezingsprogrammasdownloaden.nl](https://www.verkiezingsprogrammasdownloaden.nl/).

## Vereisten

- Python 3.10 of hoger
- Internetverbinding

## Gebruik

```bash
python3 download_verkiezingsprogrammas.py
```

Standaard worden de bestanden opgeslagen in de map `verkiezingsprogrammas/`, waarbij elke partij een eigen submap krijgt. Bestaande bestanden worden overgeslagen.

### Opties

- `--output-dir PAD` – aangepaste doelmap ingesteld.
- `--force` – downloadt bestanden opnieuw, ook als ze al bestaan.

### Voorbeeld

```bash
python3 download_verkiezingsprogrammas.py --output-dir data/programma's --force
```

Na afloop toont de tool hoeveel bestanden nieuw zijn gedownload, overgeslagen of niet konden worden opgehaald (bijvoorbeeld bij ontbrekende bronbestanden).

## OCR en Tekstextractie

Voor optimale tekstkwaliteit kun je de PDFs converteren naar kale tekst met een hybride pipeline:

```bash
python3 extract_program_texts.py
```

Standaard zoekt het script PDF-bestanden in `verkiezingsprogrammas/` en schrijft tekstbestanden naar `verkiezingsprogrammas_text/`, waarbij de mapstructuur behouden blijft. Het script combineert directe PDF-tekstextractie met OCR (Tesseract) voor pagina's zonder tekstlaag.

### Extra opties

- `--input-dir PAD` / `--output-dir PAD` – stel de bron- en doelmap in.
- `--force-ocr` – dwing OCR af voor alle pagina's (traag maar grondig).
- `--lang NLD+ENG` – geef gewenste Tesseract-taalcode(s) door. Standaard `nld+eng`.
- `--dpi 300` – renderresolutie voor OCR fallback.
- `--tesseract-config "--psm 6"` – extra configuratie rechtstreeks naar Tesseract.
- `--overwrite` – bestaande tekstbestanden overschrijven.

### Systeemvereisten

- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) inclusief Nederlandse taaldata (`tesseract-ocr-nld` of `tesseract-lang` afhankelijk van je pakketbeheerder).

Installeer daarna de Python dependencies:

```bash
pip install -r requirements.txt
```

## ManifestoBERTa Analyse

Gebruik `analyze_manifestos.py` om de opgeschoonde teksten te classificeren met het Manifesto Project model:

```bash
python3 analyze_manifestos.py \
  --input-dir verkiezingsprogrammas_text \
  --output-dir verkiezingsprogrammas_predictions \
  --context-window 3 \
  --batch-size 8 \
  --max-length 200
```

Belangrijk:

- De script verwacht per manifesto een `.txt`-bestand (zoals gegenereerd door de extractietool).
- Tekst wordt automatisch in volledige zinnen gesegmenteerd (op `.`); bij te lange context wordt teruggevallen op de laatste volledige zin die binnen `max_length` tokens past.
- `--context-window` bepaalt hoeveel voorgaande zinnen als context worden meegenomen.
- Bestanden waarvoor al een JSON-uitvoer bestaat in `--output-dir` worden automatisch overgeslagen (handig bij onderbroken runs).
- Zorg dat je een geschikte GPU beschikbaar hebt of laat `--device cpu` staan (langzamer).
- Output is JSON met per zin het voorspelde label en kansverdeling.

## Vectoranalyse & Gelijkenissen

Na het genereren van de zin-voorspellingen kun je de partijprofielen samenvatten en vergelijken:

```bash
python3 analyze_similarity.py \
  --predictions-dir verkiezingsprogrammas_predictions \
  --output-dir analysis \
  --top-k 5
```

Dit script:

- bouwt voor iedere partij het gemiddelde topic-profiel (gesaved als `analysis/party_topic_vectors.parquet` en `.csv`);
- maakt een cosine-similaritymatrix (`analysis/party_similarity_matrix.csv`);
- schrijft per partijpaar de belangrijkste gezamenlijke topics en grootste verschillen naar `analysis/pairwise_contributions.json`.
