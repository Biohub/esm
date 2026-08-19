from typing import Callable

import pygtrie
from ipywidgets import widgets

from esm.sdk.api import FunctionAnnotation
from esm.tokenization.function_tokenizer import (
    InterProQuantizedTokenizer,
)

TRIE: pygtrie.CharTrie | None = None


def get_trie() -> pygtrie.CharTrie:
    global TRIE
    if TRIE is None:
        # Init keyword trie
        TRIE = pygtrie.CharTrie(separator=" ")
        interpro_tokenizer = InterProQuantizedTokenizer()
        for keyword in interpro_tokenizer._tfidf.vocabulary:
            TRIE[keyword.lower()] = keyword

        for interpro_tag in interpro_tokenizer.interpro_labels:
            TRIE[interpro_tag.lower()] = interpro_tag
    return TRIE


def create_function_annotator(
    protein_length: int,
    add_annotation_callback: Callable[[FunctionAnnotation], None],
    delete_annotation_callback: Callable[[FunctionAnnotation], None],
) -> widgets.Widget:
    trie = get_trie()

    # Create a text box input and a list of children to select from
    # if a keyword is not in the trie, raise an error

    text_input = widgets.Text(
        description="Function", disabled=False, layout=widgets.Layout(width="400px")
    )
    suggestions = widgets.SelectMultiple(
        options=[],
        description="Suggestions",
        disabled=False,
        layout=widgets.Layout(width="400px"),
    )
    add_button = widgets.Button(
        description="Add",
        disabled=True,
        tooltip="Add the selected function to the target range",
        icon="plus",
    )
    target_range_slider_label = widgets.Label(
        value="Target Range in Prompt:", layout=widgets.Layout(width="150px")
    )
    target_range_slider = widgets.IntRangeSlider(
        value=[0, protein_length - 1],
        min=0,
        max=protein_length - 1,
        step=1,
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        readout_format="d",
        layout=widgets.Layout(width="600px"),
    )
    output = widgets.Output()

    entries = widgets.VBox([])

    def on_text_change(change):
        output.clear_output()
        text: str = change["new"]
        if not text:
            suggestions.options = []
            return
        try:
            options = list(trie.itervalues(text.lower()))
        except KeyError:
            options = []
            with output:
                print(f"Keyword {text} not found in the Function Annotation vocabulary")
        suggestions.options = options

        if is_keyword_valid(text):
            add_button.disabled = False
        else:
            add_button.disabled = True

    def on_suggestion_click(change):
        if not change["new"]:
            return
        value, *_ = change["new"]
        text_input.value = value

    def on_add_click(b):
        output.clear_output()
        try:
            function_label = text_input.value
            start, end = target_range_slider.value
            add_annotation_callback(
                FunctionAnnotation(function_label, start + 1, end + 1)
            )

            function_str = f"[{start}-{end}]: {function_label} "

            def on_delete_click(b):
                delete_annotation_callback(
                    FunctionAnnotation(function_label, start + 1, end + 1)
                )
                entries.children = tuple(
                    entry
                    for entry in entries.children
                    if entry.children[1].value != function_str
                )

            delete_button = widgets.Button(
                description="Delete", tooltip="Delete this annotation", icon="trash"
            )
            entry = widgets.HBox([delete_button, widgets.Label(value=function_str)])
            delete_button.on_click(on_delete_click)
            entries.children += (entry,)

        except Exception as e:
            with output:
                print(f"Error: {e}")

    def is_keyword_valid(keyword: str) -> bool:
        return keyword.lower() in trie

    text_input.observe(on_text_change, names="value")
    suggestions.observe(on_suggestion_click, names="value")
    add_button.on_click(on_add_click)

    function_annotation_ui = widgets.VBox(
        [
            widgets.HBox([text_input, add_button]),
            suggestions,
            widgets.HBox([target_range_slider_label, target_range_slider]),
            output,
            entries,
        ]
    )

    return function_annotation_ui
