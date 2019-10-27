import logging

from typing import Dict, List, Tuple, Iterable, Any
from overrides import overrides
from nltk.tree import Tree

#sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(os.pardir)),"allennlp"))

from allennlp.common import Params
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.dataset_utils import Ontonotes, OntonotesSentence
from allennlp.data.fields import TextField, SpanField, SequenceLabelField, ListField, Field, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.dataset_readers.dataset_utils.span_utils import enumerate_spans


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@DatasetReader.register("srl_with_parse")
class SrlwithConstituencySpanOntonotesReader(DatasetReader):
    """
    Reads constituency parses from the English OntoNotes v5.0 data from the LDC.
    This ``DatasetReader`` enumerates all possible spans in the sentence and returns them,
    along with gold labels for the relevant spans present in a gold tree, if provided.

    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
        Note that the `output` tags will always correspond to single token IDs based on how they
        are pre-tokenised in the data file.
    lazy : ``bool``, optional, (default = ``False``)
        Whether or not instances can be consumed lazily.
    """
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 domain_identifier: str = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._domain_identifier = domain_identifier

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        ontonotes_reader = Ontonotes()
        logger.info("Reading SRL instances along with constituent parse from data files at: %s", file_path)

        if self._domain_identifier is not None:
            logger.info("Filtering to only include file paths containing the %s domain", self._domain_identifier)

        for sentence in self._ontonotes_subset(ontonotes_reader, file_path, self._domain_identifier):
            tokens = [Token(t) for t in sentence.words]

            parse = sentence.parse_tree
            if parse:
                pos_tags = [x[1] for x in parse.pos()]
                # yield self.text_to_instance(parse.leaves(), [x[1] for x in parse.pos()], parse)
            else:
            # parse information is missing for this sentence
                parse = None
                pos_tags = None

            if not sentence.srl_frames:
                # Sentence contains no predicates.
                tags = ["O" for _ in tokens]
                verb_label = [0 for _ in tokens]
                yield self.text_to_instance(tokens, verb_label, tags, pos_tags, parse)
            else:
                for (_, tags) in sentence.srl_frames:
                    verb_indicator = [1 if label[-2:] == "-V" else 0 for label in tags]
                    yield self.text_to_instance(tokens, verb_indicator, tags, pos_tags, parse)

    @staticmethod
    def _ontonotes_subset(ontonotes_reader: Ontonotes,
                          file_path: str,
                          domain_identifier: str) -> Iterable[OntonotesSentence]:
        """
        Iterates over the Ontonotes 5.0 dataset using an optional domain identifier.
        If the domain identifier is present, only examples which contain the domain
        identifier in the file path are yielded.
        """
        for conll_file in ontonotes_reader.dataset_path_iterator(file_path):
            if domain_identifier is None or f"/{domain_identifier}/" in conll_file:
                yield from ontonotes_reader.sentence_iterator(conll_file)

    @overrides
    def text_to_instance(self, # type: ignore
                         tokens: List[str],
                         verb_label: List[int],
                         tags: List[str] = None,
                         pos_tags: List[str] = None,
                         gold_tree: Tree = None) -> Instance:
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.

        Parameters
        ----------
        tokens : ``List[str]``, required.
            The tokens in a given sentence.
        verb_label: ``List[int]``, required
            The verb label should be a one-hot binary vector,
            the same length as the tokens, indicating the position of the verb to find arguments for.
        tags: ``List[str]``, , optional (default = None).
            SRL tags
        pos_tags ``List[str]``, optional (default = None).
            The pos tags for the words in the sentence.
        gold_tree : ``Tree``, optional (default = None).
            The gold parse tree to create span labels from.

        Returns
        -------
        An ``Instance`` containing the following fields:
            tokens : ``TextField``
                The tokens in the sentence.
            pos_tags : ``SequenceLabelField``
                The pos tags of the words in the sentence.
            spans : ``ListField[SpanField]``
                A ListField containing all possible subspans of the
                sentence.
            span_labels : ``SequenceLabelField``, optional.
                The constituency tags for each of the possible spans, with
                respect to a gold parse tree. If a span is not contained
                within the tree, a span will have a ``NO-LABEL`` label.
        """
        # pylint: disable=arguments-differ

        fields: Dict[str, Field] = {}
        text_field = TextField(tokens, token_indexers=self._token_indexers)
        fields['tokens'] = text_field
        fields['verb_indicator'] = SequenceLabelField(verb_label, text_field)

        metadata: Dict [str, Any] = {}

        if tags:
            fields['tags'] = SequenceLabelField(tags, text_field)

        if pos_tags:
            pos_tag_field = SequenceLabelField(pos_tags, text_field, "pos_tags")
            fields['pos_tags'] = pos_tag_field
            metadata['pos_tags'] = True
        else:
            pos_tags = ['X' for _ in tokens]
            fields['pos_tags'] = SequenceLabelField(pos_tags, text_field, "pos_tags")
            metadata['pos_tags'] = False

        spans: List[Field] = []
        gold_labels = []

        if gold_tree is not None:
            gold_spans_with_pos_tags: Dict[Tuple[int, int], str] = {}
            self._get_gold_spans(gold_tree, 0, gold_spans_with_pos_tags)
            gold_spans = {span: label for (span, label)
                          in gold_spans_with_pos_tags.items() if "-POS" not in label}
        else:
            gold_spans = None

        for start, end in enumerate_spans(tokens):
            spans.append(SpanField(start, end, text_field))

            if gold_spans is not None:
                if (start, end) in gold_spans.keys():
                    gold_labels.append(gold_spans[(start, end)])
                else:
                    gold_labels.append("NO-LABEL")
            else:
                gold_labels.append("NO-LABEL")

        span_list_field: ListField = ListField(spans)
        fields['spans'] = span_list_field

        if gold_tree is not None:
            fields['span_labels'] = SequenceLabelField(gold_labels, span_list_field, "constituent_labels")
            metadata['span_labels'] = True
        else:
            fields['span_labels'] = SequenceLabelField(gold_labels, span_list_field, "constituent_labels")
            metadata['span_labels'] = False

        metadata_field = MetadataField(metadata)
        fields['metadata'] = metadata_field

        return Instance(fields)

    def _get_gold_spans(self, # pylint: disable=arguments-differ
                        tree: Tree,
                        index: int,
                        typed_spans: Dict[Tuple[int, int], str]) -> int:
        """
        Recursively construct the gold spans from an nltk ``Tree``.
        Spans are inclusive.

        Parameters
        ----------
        tree : ``Tree``, required.
            An NLTK parse tree to extract spans from.
        index : ``int``, required.
            The index of the current span in the sentence being considered.
        typed_spans : ``Dict[Tuple[int, int], str]``, required.
            A dictionary mapping spans to span labels.

        Returns
        -------
        typed_spans : ``Dict[Tuple[int, int], str]``.
            A dictionary mapping all subtree spans in the parse tree
            to their constituency labels. Leaf nodes have POS tag spans, which
            are denoted by a label of "LABEL-POS".
        """
        # NLTK leaves are strings.
        if isinstance(tree[0], str):
            # The "length" of a tree is defined by
            # NLTK as the number of children.
            # We don't actually want the spans for leaves, because
            # their labels are pos tags. However, it makes the
            # indexing more straightforward, so we'll collect them
            # and filter them out below. We subtract 1 from the end
            # index so the spans are inclusive.
            end = index + len(tree)
            typed_spans[(index, end - 1)] = tree.label() + "-POS"
        else:
            # otherwise, the tree has children.
            child_start = index
            for child in tree:
                # typed_spans is being updated inplace.
                end = self._get_gold_spans(child, child_start, typed_spans)
                child_start = end
            # Set the end index of the current span to
            # the last appended index - 1, as the span is inclusive.
            typed_spans[(index, end - 1)] = tree.label()
        return end

    @classmethod
    def from_params(cls, params: Params) -> 'SrlwithConstituencySpanOntonotesReader':
        token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
        domain_identifier = params.pop("domain_identifier", None)
        lazy = params.pop('lazy', False)
        params.assert_empty(cls.__name__)
        return SrlwithConstituencySpanOntonotesReader(token_indexers=token_indexers,
                                                         lazy=lazy)
