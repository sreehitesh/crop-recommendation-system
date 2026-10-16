"""Short, factual reference notes for the 22 crop classes in the dataset.

These notes exist so the result card can add context to a predicted label.
They are deliberately brief and generic: they describe the crop at a
high level and are compiled from public general-agronomy descriptions
(English Wikipedia crop articles and standard agronomy handbooks).

They are NOT agronomic recommendations, and they are NOT used by the model in
any way. The model predicts a label; this text only explains what that label
is. See docs/DATASET.md for attribution.
"""

from __future__ import annotations

CROP_INFO: dict[str, dict[str, str]] = {
    "apple": {
        "display": "Apple",
        "category": "Fruit (temperate tree crop)",
        "note": "Deciduous fruit tree grown in cooler climates; needs a cold period for good fruiting.",
    },
    "banana": {
        "display": "Banana",
        "category": "Fruit (tropical herbaceous crop)",
        "note": "Tropical crop grown from suckers; requires warm temperatures and steady moisture.",
    },
    "blackgram": {
        "display": "Black gram (urad)",
        "category": "Pulse (legume)",
        "note": "Warm-season pulse; as a legume it can fix atmospheric nitrogen in the soil.",
    },
    "chickpea": {
        "display": "Chickpea (gram)",
        "category": "Pulse (legume)",
        "note": "Cool-season pulse commonly grown on residual soil moisture after the monsoon.",
    },
    "coconut": {
        "display": "Coconut",
        "category": "Plantation crop (palm)",
        "note": "Long-lived tropical palm; prefers humid coastal climates and well-drained soil.",
    },
    "coffee": {
        "display": "Coffee",
        "category": "Plantation crop (beverage)",
        "note": "Perennial shrub grown on shaded slopes in tropical highlands.",
    },
    "cotton": {
        "display": "Cotton",
        "category": "Fibre crop",
        "note": "Warm-season fibre crop; requires a long frost-free growing period.",
    },
    "grapes": {
        "display": "Grapes",
        "category": "Fruit (vine)",
        "note": "Perennial vine; yields depend strongly on pruning, training and climate.",
    },
    "jute": {
        "display": "Jute",
        "category": "Fibre crop",
        "note": "Warm, humid-season fibre crop grown mainly in river plains and delta soils.",
    },
    "kidneybeans": {
        "display": "Kidney beans (rajma)",
        "category": "Pulse (legume)",
        "note": "Cooler-season pulse; grown in well-drained soils with moderate rainfall.",
    },
    "lentil": {
        "display": "Lentil (masoor)",
        "category": "Pulse (legume)",
        "note": "Cool-season pulse, often rainfed on stored soil moisture.",
    },
    "maize": {
        "display": "Maize (corn)",
        "category": "Cereal",
        "note": "Warm-season cereal used for food, feed and industrial purposes.",
    },
    "mango": {
        "display": "Mango",
        "category": "Fruit (tropical tree)",
        "note": "Long-lived tropical tree; flowering and fruiting are sensitive to moisture timing.",
    },
    "mothbeans": {
        "display": "Moth beans",
        "category": "Pulse (legume)",
        "note": "Drought-tolerant warm-season pulse suited to arid and semi-arid areas.",
    },
    "mungbean": {
        "display": "Mung bean (green gram)",
        "category": "Pulse (legume)",
        "note": "Short-duration warm-season pulse; can fix nitrogen in the soil.",
    },
    "muskmelon": {
        "display": "Muskmelon",
        "category": "Fruit (cucurbit)",
        "note": "Warm-season vining crop that needs a dry, warm period for ripening.",
    },
    "orange": {
        "display": "Orange",
        "category": "Fruit (citrus tree)",
        "note": "Subtropical citrus tree; sensitive to frost and to poor soil drainage.",
    },
    "papaya": {
        "display": "Papaya",
        "category": "Fruit (tropical)",
        "note": "Fast-growing tropical crop; sensitive to waterlogging and strong wind.",
    },
    "pigeonpeas": {
        "display": "Pigeon peas (tur/arhar)",
        "category": "Pulse (legume)",
        "note": "Deep-rooted warm-season pulse, relatively tolerant of dry spells.",
    },
    "pomegranate": {
        "display": "Pomegranate",
        "category": "Fruit (shrub/tree)",
        "note": "Drought-hardy fruit shrub; fruit quality is affected by irrigation scheduling.",
    },
    "rice": {
        "display": "Rice (paddy)",
        "category": "Cereal",
        "note": "Warm, humid-season cereal usually grown under flooded or assured-irrigation conditions.",
    },
    "watermelon": {
        "display": "Watermelon",
        "category": "Fruit (cucurbit)",
        "note": "Warm-season trailing crop that requires a long frost-free period.",
    },
}

UNKNOWN_CROP_NOTE = (
    "No reference note is available for this label. The label itself comes "
    "directly from the trained model's class list."
)


def get_crop_info(label: str) -> dict[str, str]:
    """Return reference notes for a predicted crop label."""
    key = str(label).strip().lower()
    info = CROP_INFO.get(key)
    if info is None:
        return {"display": str(label).title(), "category": "Unknown", "note": UNKNOWN_CROP_NOTE}
    return dict(info)
