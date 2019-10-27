from typing import List, Dict, Tuple, Union
from allennlp.data.dataset_readers.dataset_utils.ontonotes import TypedStringSpan

from allennlp.data.fields.field import Field

def get_syntactic_spans_from_SpanField(spans: List[Field],
                        span_labels: Union[List[str], List[int]]) -> Dict[Tuple[int, int], str]:

    typed_spans: Dict[Tuple[int, int], str] = {}

    for span, span_label in zip(spans, span_labels):

        if span_label != "NO-LABEL" and span_label != "-":
            typed_spans[(span.span_start, span.span_end)] = span_label

    return typed_spans

def get_syntactic_spans(spans: List[Tuple[int, int]],
                        span_labels: Union[List[str], List[int]]) -> Dict[Tuple[int, int], str]:

    typed_spans: Dict[Tuple[int, int], str] = {}

    for span, span_label in zip(spans, span_labels):

        if span_label != "NO-LABEL" and span_label != "-":
            typed_spans[span] = span_label

    return typed_spans

# Base code from allennlp.data.dataset_readers.dataset_utils.span_utils
def bio_tags_to_spans(tag_sequence: List[str],
                      classes_to_ignore: List[str] = None) -> List[TypedStringSpan]:
    """
    Given a sequence corresponding to BIO tags, extracts spans.
    Spans are inclusive and can be of zero length, representing a single word span.
    Ill-formed spans are also included (i.e those which do not start with a "B-LABEL"),
    as otherwise it is possible to get a perfect precision score whilst still predicting
    ill-formed spans in addition to the correct spans.

    Parameters
    ----------
    tag_sequence : List[str], required.
        The integer class labels for a sequence.
    classes_to_ignore : List[str], optional (default = None).
        A list of string class labels `excluding` the bio tag
        which should be ignored when extracting spans.

    Returns
    -------
    spans : List[TypedStringSpan]
        The typed, extracted spans from the sequence, in the format (label, (span_start, span_end)).
        Note that the label `does not` contain any BIO tag prefixes.
    """
    classes_to_ignore = classes_to_ignore or []
    spans = set()
    span_start = 0
    span_end = 0
    active_conll_tag = None

    # Keeping track of different type of BIO violations in case Ill-formed spans
    n_b_i = 0
    n_i_i = 0
    n_o_i = 0

    for index, string_tag in enumerate(tag_sequence):
        # Actual BIO tag.
        bio_tag = string_tag[0]
        conll_tag = string_tag[2:]
        if bio_tag == "O" or conll_tag in classes_to_ignore:
            # The span has ended.
            if active_conll_tag:
                spans.add((active_conll_tag, (span_start, span_end)))
            active_conll_tag = None
            # We don't care about tags we are
            # told to ignore, so we do nothing.
            continue
        elif bio_tag == "U":
            # The U tag is used to indicate a span of length 1,
            # so if there's an active tag we end it, and then
            # we add a "length 0" tag.
            if active_conll_tag:
                spans.add((active_conll_tag, (span_start, span_end)))
            spans.add((conll_tag, (index, index)))
            active_conll_tag = None
        elif bio_tag == "B":
            # We are entering a new span; reset indices
            # and active tag to new span.
            if active_conll_tag:
                spans.add((active_conll_tag, (span_start, span_end)))
            active_conll_tag = conll_tag
            span_start = index
            span_end = index
        elif bio_tag == "I" and conll_tag == active_conll_tag:
            # We're inside a span.
            span_end += 1
        else:
            # This is the case the bio label is an "I", but either:
            # 1) the span hasn't started - i.e. an ill formed span.
            # 2) The span is an I tag for a different conll annotation.
            # We'll process the previous span if it exists, but also
            # include this span. This is important, because otherwise,
            # a model may get a perfect F1 score whilst still including
            # false positive ill-formed spans.
            if active_conll_tag:
                spans.add((active_conll_tag, (span_start, span_end)))
            active_conll_tag = conll_tag
            span_start = index
            span_end = index

            if active_conll_tag and index >= 1:
                prev_bio_tag = tag_sequence[index-1][0]
                if prev_bio_tag == "B":
                    n_b_i += 1
                elif prev_bio_tag == "I":
                    n_i_i += 1
                else:
                    n_o_i += 1
            else:
                # If first tag itself is I-label..It is assumed as O-I violation
                n_o_i += 1
    # Last token might have been a part of a valid span.
    if active_conll_tag:
        spans.add((active_conll_tag, (span_start, span_end)))

    return list(spans), (n_b_i, n_i_i, n_o_i)

def srl_bio_tags_to_spans(tag_sequence: List[str]):

    spans, violations = bio_tags_to_spans(tag_sequence=tag_sequence)

    srl_spans: Dict[Tuple[int, int], str] = {}

    for span in spans:

        span_label = span[0]
        span_start = span[1][0]
        span_end = span[1][1]

        # We skip "V" tags/labels as we are interested only in argument tags/labels
        if span_label != "V":
            srl_spans[(span_start, span_end)] = span_label

    return srl_spans

def bio_tags_to_conll_format(labels: List[str]):

    conll_labels = []
    active_conll_tag = None

    for i, label in enumerate(labels):

        # Actual BIO tag.
        bio_tag = label[0]
        conll_tag = label[2:]

        if bio_tag == "O":
            # The span has ended.
            if active_conll_tag:
                conll_labels[-1] = conll_labels[-1] + ")"
            active_conll_tag = None
            conll_labels.append("*")
            # We don't care about tags we are
            # told to ignore, so we do nothing.
            continue
        elif bio_tag == "B":
            # We are entering a new span; reset indices
            # and active tag to new span.
            if active_conll_tag:
                conll_labels[-1] = conll_labels[-1] + ")"
            active_conll_tag = conll_tag
            new_label = "(" + active_conll_tag + "*"
            conll_labels.append(new_label)

        elif bio_tag == "I" and conll_tag == active_conll_tag:
            # We're inside a span.
            conll_labels.append("*")
        else:
            # This is the case the bio label is an "I", but either:
            # 1) the span hasn't started - i.e. an ill formed span.
            # 2) The span is an I tag for a different conll annotation.
            # We'll process the previous span if it exists, but also
            # include this span. This is important, because otherwise,
            # a model may get a perfect F1 score whilst still including
            # false positive ill-formed spans.
            if active_conll_tag:
                conll_labels[-1] = conll_labels[-1] + ")"
            active_conll_tag = conll_tag
            new_label = "(" + active_conll_tag + "*"
            conll_labels.append(new_label)

    if active_conll_tag:
        conll_labels[-1] = conll_labels[-1] + ")"

    return conll_labels