"""Canonical culture roster for the WVS track."""

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
