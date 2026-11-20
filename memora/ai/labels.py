"""Class definitions shared by training, inference and the interface."""

CLASSES = [
    {
        "key": "CN",
        "name": "Cognitively Normal",
        "short": "Normal",
        "tone": "calm",
        "summary": "No pattern of Alzheimer's-type atrophy was detected across the series.",
        "advice": "Routine follow-up. Repeat imaging only if symptoms appear or progress.",
    },
    {
        "key": "MCI",
        "name": "Early Stage (MCI)",
        "short": "Early stage",
        "tone": "watch",
        "summary": "The series shows a pattern the model associates with mild cognitive impairment or very mild dementia.",
        "advice": "Correlate with cognitive testing (MMSE or MoCA) and history. Consider specialist referral and a follow-up scan.",
    },
    {
        "key": "AD",
        "name": "Alzheimer's Disease",
        "short": "Alzheimer's",
        "tone": "alert",
        "summary": "The series shows a pattern the model associates with established Alzheimer's-type atrophy.",
        "advice": "Refer for full neurological assessment. Imaging alone does not confirm the diagnosis.",
    },
]

CLASS_KEYS = [c["key"] for c in CLASSES]
CLASS_NAMES = [c["name"] for c in CLASSES]
BY_NAME = {c["name"]: c for c in CLASSES}
BY_KEY = {c["key"]: c for c in CLASSES}
