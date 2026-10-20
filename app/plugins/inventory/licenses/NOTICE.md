# Lexical data attribution

The reviewed entity vocabulary derives from Princeton WordNet 3.0 and WOLF
(Wordnet Libre du Français), via the pinned Open Multilingual Wordnet revision
recorded in `../entity_lexicon.sources.json`.

- Princeton WordNet: Copyright 2006 Princeton University. Preserve WORDNET-3.0.txt.
- WOLF: Benoît Sagot and Darja Fišer; CeCILL-C. Preserve WOLF-CeCILL-C.html.
- Citations are provided in wordnet.bib and wolf.bib.

Aria's modifications select ten household noun senses, normalize French/English
forms, exclude generic/inappropriate terms and omit cross-concept collisions.
The generated data, review exclusions, term/sense provenance and offline generation
script are supplied in source form. No upstream library or online lookup runs
within Aria. To reproduce, prepare the two pinned tables and run
`python -m scripts.build_entity_lexicon --sources-dir <prepared-files>` from the
repository root. Source hashes are verified before generation.
