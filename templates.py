"""Prompt templates and class-name constants.

Templates are the CoOp-style custom templates, matching what the AReS paper
uses to build the CLIP zero-shot head of the service model (one template per
dataset). GTSRB class names follow ILM-VP.
"""

CUSTOM_TEMPLATES = {
    "oxfordpets": "a photo of a {}, a type of pet.",
    "flowers102": "a photo of a {}, a type of flower.",
    "dtd": "{} texture.",
    "eurosat": "a centered satellite photo of {}.",
    "svhn": "This is a photo of a {}",
    "gtsrb": "This is a photo of a {}",
}

# index -> readable name, taken from ILM-VP's label map
GTSRB_CLASSES = [
    "20_speed", "30_speed", "50_speed", "60_speed", "70_speed", "80_speed",
    "80_lifted", "100_speed", "120_speed", "no_overtaking_general",
    "no_overtaking_trucks", "right_of_way_crossing", "right_of_way_general",
    "give_way", "stop", "no_way_general", "no_way_trucks", "no_way_one_way",
    "attention_general", "attention_left_turn", "attention_right_turn",
    "attention_curvy", "attention_bumpers", "attention_slippery",
    "attention_bottleneck", "attention_construction", "attention_traffic_light",
    "attention_pedestrian", "attention_children", "attention_bikes",
    "attention_snowflake", "attention_deer", "lifted_general", "turn_right",
    "turn_left", "turn_straight", "turn_straight_right", "turn_straight_left",
    "turn_right_down", "turn_left_down", "turn_circle",
    "lifted_no_overtaking_general", "lifted_no_overtaking_trucks",
]

IMAGENET_NORM = {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}

# normalization used by OpenAI CLIP's preprocess
CLIP_NORM = {
    "mean": [0.48145466, 0.4578275, 0.40821073],
    "std": [0.26862954, 0.26130258, 0.27577711],
}
