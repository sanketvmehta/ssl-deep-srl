import os
import logging

from typing import Iterable, TextIO, List

from allennlp.data.tokenizers import Token
from allennlp.data.instance import Instance
from allennlp.data.fields import SpanField

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def write_parse_tree_to_file(parse_file: TextIO, tokens: List[Token], tags: List[str],
                             pos_tags: List[str], spans: SpanField, span_labels: List[str]):

    for token, tag, pos_tag in zip(tokens, tags, pos_tags):
        parse_file.write(token.__str__().ljust(15))
        parse_file.write(tag.rjust(15))
        parse_file.write(pos_tag.rjust(15) + "\n")

    parse_file.write("\n")

    for span, span_label in zip(spans, span_labels):
        # parse_file.write(' '.join([token.__str__() for token in span.sequence_field.tokens]) + "\n")
        if span_label != "NO-LABEL":
            parse_file.write(str(span.span_start) + "-" + str(span.span_end) + ":" + span_label + "\n")

    parse_file.write("---------------------\n")

def write_parse_trees_to_file(serialization_dir: str, instances: Iterable[Instance], split: str):

    parse_file_path = os.path.join(serialization_dir, "parse-trees-" + split + ".txt")
    logger.info("Writing parse trees to file: %s", parse_file_path)
    parse_file = open(parse_file_path, "w+")

    for instance in instances:

        fields = instance.fields

        tokens = fields["tokens"].tokens
        tags = fields["tags"].labels
        pos_tags = fields["pos_tags"]
        spans = fields["spans"].field_list
        span_labels = fields["span_labels"]

        metadata = instance.fields["metadata"].metadata

        if metadata['pos_tags'] is False:
            pos_tags = ['X'] * len(tokens)
        else:
            pos_tags = pos_tags.labels

        if metadata['span_labels'] is False:
            span_labels = ['-'] * len(spans)
        else:
            span_labels = span_labels.labels

        write_parse_tree_to_file(parse_file, tokens, tags, pos_tags, spans, span_labels)

    parse_file.close()
