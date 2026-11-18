"""Tests for the robust Hebrew bidi heuristic."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.pdf_parser import _fix_bidi

FORWARD = "הוועדה תכלול 2 מרכזי הוועדה, 9 נציגי ציבור ונציגי משרד השיכון. זהו תקנון השיכון ובו סעיפים רבים."
REVERSED = FORWARD[::-1]


def test_reversed_hebrew_is_corrected():
    assert _fix_bidi(REVERSED) == FORWARD


def test_forward_hebrew_is_untouched():
    assert _fix_bidi(FORWARD) == FORWARD


def test_english_text_is_untouched():
    text = "The housing committee shall include 2 coordinators and 9 representatives."
    assert _fix_bidi(text) == text


def test_empty_text():
    assert _fix_bidi("") == ""


def test_short_noise_is_untouched():
    # a single reversed word is not enough evidence to flip a page
    noise = "אבג"
    assert _fix_bidi(noise) == noise


def test_reversed_line_order_is_preserved_and_dates_intact():
    # visual-order page: lines are char-reversed, line order is correct,
    # digit runs (dates) are stored logically and must survive the fix
    raw = (
        "ךיראת רשאמ ףוג הסרג\n"
        "18.7.22 טרופסו תוברת לוכשא 1\n"
        ":הכירב להונ\n"
        ".דבלב ויבשותו ץוביקה ירבחל תדעוימ הייחשה תכירב .1\n"
    )
    expected = (
        "גרסה גוף מאשר תאריך\n"
        "1 אשכול תרבות וספורט 18.7.22\n"
        "נוהל בריכה:\n"
        ".1 בריכת השחייה מיועדת לחברי הקיבוץ ותושביו בלבד."
    )
    assert _fix_bidi(raw) == expected


def test_phone_numbers_survive_reversal():
    raw = "3780500 השנמ .נ.ד הליהקה ץוביק\n04-6394510 :סקפ ,04-6394500 .לט תוריכזמ"
    fixed = _fix_bidi(raw)
    assert "04-6394500" in fixed and "04-6394510" in fixed and "3780500" in fixed
    assert "קיבוץ הקהילה" in fixed and "מזכירות" in fixed


def test_mixed_page_majority_decides():
    # mostly logical page with one stray reversed-looking line: page stays forward
    text = "הוועדה תכלול מרכזי הוועדה ונציגי ציבור רבים בתקנון שיכון.\nרבוטקואל"
    assert _fix_bidi(text) == text
