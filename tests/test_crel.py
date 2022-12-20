import os
import pytest
from REL.crel.conv_el import ConvEL
import yaml
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


@pytest.fixture()
def cel():
    return ConvEL(base_url=os.environ.get("REL_BASE_URL", "."))


@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS")=='true', reason="No way of testing this on Github actions."
)
def test_conv1(cel):
    example = [
        {
            "speaker": "USER",
            "utterance": "I think science fiction is an amazing genre for anything. Future science, technology, time travel, FTL travel, they're all such interesting concepts.",
        },
        {
            "speaker": "SYSTEM",
            "utterance": "Awesome! I really love how sci-fi storytellers focus on political/social/philosophical issues that would still be around even in the future. Makes them relatable.",
        },
        {
            "speaker": "USER",
            "utterance": "I agree. One of my favorite forms of science fiction is anything related to time travel! I find it fascinating.",
        },
    ]

    result = cel.annotate(example)
    assert isinstance(result, list)

    expected_annotations = [
        [
            [8, 15, "science fiction", "Science_fiction"],
            [38, 5, "genre", "Genre_fiction"],
            [74, 10, "technology", "Technology"],
            [86, 11, "time travel", "Time_travel"],
            [99, 10, "FTL travel", "Faster-than-light"],
        ],
        [
            [37, 15, "science fiction", "Science_fiction"],
            [76, 11, "time travel", "Time_travel"],
            [16, 36, "my favorite forms of science fiction", "Time_travel"],
        ],
    ]

    annotations = [res["annotations"] for res in result if res["speaker"] == "USER"]

    assert annotations == expected_annotations


@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS")=='true', reason="No way of testing this on Github actions."
)
def test_conv2(cel):
    example = [
        {
            "speaker": "USER",
            "utterance": "I am allergic to tomatoes but we have a lot of famous Italian restaurants here in London.",
        },
        {
            "speaker": "SYSTEM",
            "utterance": "Some people are allergic to histamine in tomatoes.",
        },
        {
            "speaker": "USER",
            "utterance": "Talking of food, can you recommend me a restaurant in my city for our anniversary?",
        },
    ]

    result = cel.annotate(example)
    assert isinstance(result, list)

    annotations = [res["annotations"] for res in result if res["speaker"] == "USER"]

    expected_annotations = [
        [
            [17, 8, "tomatoes", "Tomato"],
            [54, 19, "Italian restaurants", "Italian_cuisine"],
            [82, 6, "London", "London"],
        ],
        [
            [11, 4, "food", "Food"],
            [40, 10, "restaurant", "Restaurant"],
            [54, 7, "my city", "London"],
        ],
    ]

    assert annotations == expected_annotations
