# Type L Label Lists

Each `.txt` file contains one complete benchmark label per line.

- `imagenet1k.txt` contains only the 1,000 leaf ImageNet classes, not WordNet
  parent labels added by the runtime adapter.
- `inaturalist2017.txt` contains the 4,895 binomial species categories from
  iNaturalist 2017. Genus-only, subspecies, varieties, and hybrids are omitted.
- `openimages_v4.txt` contains all 601 entries in the official V4 boxable
  class-description file rather than the much larger image-level ontology.

Run `python benchmark_choices/generate_type_l_labels.py` to regenerate and
validate all files.
