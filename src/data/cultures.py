"""Canonical culture roster for the WVS track.

Nine of the ten cultures are read straight from the ``Finetune`` directories
CultureLLM ships.  The tenth, ``spanish-mx``, has no upstream file: CultureLLM's
``Spanish`` partition concatenates Argentine and Mexican respondents into a single
training set, so a Mexico-only partition has to be rebuilt from the per-country
aggregate CSV sitting beside it.  ``DERIVED_CULTURES`` describes that rebuild.

Every other module takes its culture list from here.  The YFCC track keeps its own
nine-name roster in ``src.data.yfcc_schema``: photo partitions come from geotags,
and Mexican geotags stay with ``spanish`` there.
"""

from __future__ import annotations

from dataclasses import dataclass

CULTURES: tuple[str, ...] = (
    "arabic",
    "bengali",
    "chinese",
    "english",
    "german",
    "korean",
    "portuguese",
    "spanish",
    "spanish-mx",
    "turkish",
)

INFERENCE_ONLY_CULTURE = "inference_only"

CULTURE_DIR_MAP: dict[str, str] = {
    "arabic": "Arabic",
    "bengali": "Bengali",
    "chinese": "China",
    "english": "English",
    "german": "Germany",
    "korean": "Korean",
    "portuguese": "Portuguese",
    "spanish": "Spanish",
    "spanish-mx": "Spanish",
    "turkish": "Turkey",
}


@dataclass(frozen=True)
class DerivedCultureSpec:
    """How to rebuild a culture CultureLLM never wrote a ``Finetune`` file for.

    Attributes:
        country_csv (str): Per-country aggregate CSV, relative to the CultureLLM
            data root.
        aggregate_row (str): ``B_COUNTRY`` value marking the row of mean answers.
            ``Avg`` covers every respondent, ``Avg_First`` only the first thousand.
        system_prompt_token (str): Country name substituted into the shipped system
            prompt, matching how upstream names each culture in its own files.
        question_files (tuple[str, ...]): Question banks relative to the data root,
            concatenated in the order given.
    """

    country_csv: str
    aggregate_row: str
    system_prompt_token: str
    question_files: tuple[str, ...]


DERIVED_CULTURES: dict[str, DerivedCultureSpec] = {
    "spanish-mx": DerivedCultureSpec(
        country_csv="Spanish/Mexico.csv",
        aggregate_row="Avg",
        system_prompt_token="Mexico",
        question_files=("WVQ.jsonl", "new_WVQ_1000.jsonl"),
    ),
}

EXTRA_CULTURE_CONTEXTS: dict[str, str] = {
    "spanish-mx": (
        "The culture of Mexico is the product of a long encounter between Mesoamerican "
        "civilisation and the Spanish colonial order, and of the mestizo identity that "
        "emerged from it. Before the conquest the territory held Olmec, Maya, Zapotec, "
        "Mixtec, Toltec and Mexica societies, whose calendars, foodways, languages and "
        "monumental architecture were never fully displaced; dozens of indigenous "
        "languages remain officially recognised alongside Spanish, and Nahuatl, Maya and "
        "Zapotec are still spoken by millions. Three centuries of New Spain layered "
        "Catholicism, the Spanish language and European legal and artistic forms over that "
        "base, producing the syncretism most visible in the cult of the Virgin of Guadalupe "
        "and in Día de Muertos, where Catholic All Saints observance is fused with "
        "pre-Hispanic rites for the dead. The independence war begun in 1810 and the "
        "Revolution of 1910 supplied the country's central political narratives; the latter "
        "also produced muralism, in Rivera, Orozco and Siqueiros, and a state project of "
        "national culture built around indigenismo. Mexican cuisine, founded on maize, "
        "beans, chilli, cacao and tomato, is inscribed on the UNESCO list of intangible "
        "cultural heritage. Extended family ties, compadrazgo, and community and religious "
        "festivity remain organising social institutions, while mariachi, son, ranchera and "
        "later norteño and cumbia carry regional identity. Proximity to the United States, "
        "migration in both directions, and a largely young and urban population continue to "
        "rework these traditions rather than replace them."
    ),
}
