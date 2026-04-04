# GLiNER AU PII — Australian Organisations & Locations

This the training repo for gliner-aus-pii avaliable on hugging face: https://huggingface.co/cutaa/gliner-au-pii-v1

This model is fine-tuned to detect Australian specific organizations and locations.

Fine-tuned extension of [knowledgator/gliner-pii-large-v1.0](https://huggingface.co/knowledgator/gliner-pii-large-v1.0) for detecting Australian organisations, government agencies, and locations in open-text survey responses.

The base model performs well on general PII but lacks coverage of Australian-specific entities — remote Indigenous place names, Australian-only companies, state government agencies, and AU-specific acronyms. This model addresses those gaps.

To train, a list of Australian only companies, government agencies and unique location names were created. See ```./data``` folder.
Synthetic PII texts containing the Australian entities were then generated using gpt-4o-mini. Training data is in ```./training_data```.


## Entity Labels

| Label | Description | Examples |
|---|---|---|
| `AU_ORGANISATION` | Private companies, banks, retailers, universities, NFPs | Guzman y Gomez, Bapcor, TAFE Queensland, Lifeline Australia |
| `AU_GOV_AGENCY` | Government departments, regulatory bodies, public services | TfNSW, SA Department for Education, Fair Work Commission, APRA |
| `AU_LOCATION` | Suburbs, cities, towns, states, territories | Nhulunbuy, Warakurna, Indooroopilly, Mount Kuring-gai |


```python
from gliner import GLiNER

model = GLiNER.from_pretrained("cutaa/gliner-au-pii-v1")

labels = ["AU_ORGANISATION", "AU_GOV_AGENCY", "AU_LOCATION"]

text = "I switched from HCF to Teachers Health and it made no difference in Yugambeh."
entities = model.predict_entities(text, labels, threshold=0.5)

for entity in entities:
print(f"[{entity['label']}] '{entity['text']}' ({entity['score']:.2f})")
```
