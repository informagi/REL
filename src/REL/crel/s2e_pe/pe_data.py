from tokenizers.pre_tokenizers import Whitespace

pre_tokenizer = Whitespace()

TMP_DOC_ID = "tmp"  # temporary doc id


class PreProcess:
    """Create input for PE Linking module"""

    def _error_check(self, conv):
        assert type(conv) == list
        assert len(conv) > 0, f"conv should be a list of dicts, but got {conv}"
        for turn in conv:
            assert type(turn) == dict, f"conv should be a list of dicts, but got {turn}"
            assert set(turn.keys()) == {
                "speaker",
                "utterance",
                "mentions",
                "pems",
            } or set(turn.keys()) == {
                "speaker",
                "utterance",
            }, f"Each turn should have either [speaker, utterance, mentions, pems] keys for USER or [speaker, utterance] keys for SYSTEM. If there is no pems or mentions, then set them to empty list."
            assert turn["speaker"] in [
                "USER",
                "SYSTEM",
            ], f'The speaker should be either USER or SYSTEM, but got {turn["speaker"]}'
            assert (
                type(turn["utterance"]) == str
            ), f'The utterance should be a string, but got {turn["utterance"]}'
            if turn["speaker"] == "USER":
                assert (
                    type(turn["mentions"]) == list
                ), f'The mentions should be a list, but got {turn["mentions"]}'
                assert (
                    type(turn["pems"]) == list
                ), f'The pems should be a list, but got {turn["pems"]}'

        # Check there are only one pem per conv
        pems = [pem for turn in conv if "pems" in turn for pem in turn["pems"]]
        assert (
            len(pems) == 1
        ), f"Current implementation only supports one pem per input conv. If there are multiple PEM, then split them into multiple conv."  # This is also a TODO for the future

    def get_span(self, ment, text, flag_start_end_span_representation=True):
        """Get (start, end) span of a mention (inc. PEM) in a text

        Args:
            ment (str): E.g., 'Paris'
            text (str): E.g., 'Paris. The football club Paris Saint-Germain and the rugby union club Stade Français are based in Paris.'

        Returns: mention spans
            if flag_start_end_span_representation==True:
                E.g.,  [(0, 5), (25, 30), (98, 103)]
            if flag_start_end_span_representation==False:
                E.g.,  [(0, 5), (25, 5), (98, 5)]

        Note:
            - re.finditer is NOT used since it takes regex pattern (not string) as input and fails for the patterns such as:
                text = 'You you dance? I love cuban Salsa but also like other types as well. dance-dance.'
                ment = 'dance? '
        """
        assert ment in text, f"The mention {ment} is not in the text {text}"
        spans = []  # [(start_pos, length), ...]
        offset = 0
        while True:
            try:
                start_pos = text.index(ment, offset)
                spans.append((start_pos, len(ment)))
                offset = start_pos + len(ment)
            except:
                break

        if (
            flag_start_end_span_representation
        ):  # (start_pos, length) --> (start_pos, end_pos)
            spans = [(s, l + s) for s, l in spans]  # pos_end = pos_start + length

        return spans

    def _token_belongs_to_mention(
        self, m_spans: list, t_span: tuple, utt: str, print_warning=False
    ) -> bool:
        """Check whether token span is in ment span(s) or not

        Args:
            m_spans: e.g., [(0, 4), (10, 14), (2,4)]
            t_span: e.g., (1, 3)
        """

        def _error_check(m_spans, t_span):
            assert len(t_span) == 2
            assert (
                t_span[1] > t_span[0]
            )  # Note that span must be (start_ind, end_ind), NOT the REL style output of (start_ind, length)
            assert any(
                [True if m_span[1] > m_span[0] else False for m_span in m_spans]
            )  # The same as above

        _error_check(m_spans, t_span)

        # Main
        for m_span in m_spans:

            # if token span is out of mention span (i.e., does not have any overlaps), then go to next
            t_out_m = (t_span[1] <= m_span[0]) or (m_span[1] <= t_span[0])
            if t_out_m:
                continue

            # Check whether (1) token is in mention, (2) mention is in token, or (3) token partially overlaps with mention
            t_in_m = (m_span[0] <= t_span[0]) and (t_span[1] <= m_span[1])
            m_in_t = (t_span[0] <= m_span[0]) and (
                m_span[1] <= t_span[1]
            )  # To deal with the case of ment:"physicians" ent:"Physician"
            t_ol_m = not (t_in_m or t_out_m)  # token overlaps with mention

            if t_in_m or m_in_t:  # Case 1 or 2
                return True
            elif t_ol_m:  # Case 3
                if print_warning:
                    print(
                        f"WARNING: There is overlaps between mention and word spans. t_span:{t_span} ({utt[t_span[0]:t_span[1]]}), m_span:{m_span} ({utt[m_span[0]:m_span[1]]})"
                    )
                # NOTE: Treat this token as not belonging to the mention

        return False

    def _tokens_info(
        self,
        text: str,
        speaker: str,
        ments: list,
        pems: list,
    ):
        """Append information for each token
        The information includes:
            - span: (start_pos, end_pos) of the token, acquired from the pre_tokenizers
            - mention: what mention the token belongs to (or if not in any mention, None)
            - pem: same as mention, but for pems
            - speaker: the speaker of the utterance, either "USER" or "SYSTEM"

        Args:
            text: the text of the utterance
            speaker: the speaker of the utterance, either "USER" or "SYSTEM"
            ments: list of mentions
            pems: list of pems

        Returns: list of tokens with information
            E.g., [{'token': 'Blue',
                    'span': (0, 4),
                    'mention': 'Blue',
                    'pem': None,
                    'speaker': 'USER'}, ...]
        """
        ments = list(sorted(ments, key=len))  # to perform longest match
        tokens_conv = []

        try:  # if tokenizer version is 0.10.3 etc where pre_tokenize_str is available
            tokens_per_utt = pre_tokenizer.pre_tokenize(text)
        except:  # if 0.8.1.rc2 etc where pre_tokenizer_str is NOT available
            tokens_per_utt = pre_tokenizer.pre_tokenize_str(text)

        ment2span = {ment: self.get_span(ment, text) for ment in ments}  # mention spans
        pem2span = {pem: self.get_span(pem, text) for pem in pems}  # pem spans
        # NOTE: get_span() cannot consider the case of the same surface mention occurring multiple times in the same utterance, only one of which is detected by the MD module.
        # This is because the function detects ALL of the spans for the given surface mention.
        # However, this case is extremely rare thus here the code ignores this case.

        for tkn_span in tokens_per_utt:
            tkn = tkn_span[0]
            span = tkn_span[1]
            ment_out, pem_out = None, None  # Initialize

            # First check if token is in any PEMs
            for pem in pems:
                if self._token_belongs_to_mention(pem2span[pem], span, text):
                    pem_out = pem
                    break

            # If token is not in any pem, then check if it is in any mention
            if pem_out is None:
                for ment in ments:
                    if self._token_belongs_to_mention(ment2span[ment], span, text):
                        ment_out = ment
                        break

            tokens_conv.append(
                {
                    "token": tkn,
                    "span": span,
                    "mention": ment_out,
                    "pem": pem_out,
                    "speaker": speaker,
                }
            )
        return tokens_conv

    def get_tokens_with_info(self, conv):
        """Get tokens with information of:
            - span: (start_pos, end_pos) of the token, acquired from the pre_tokenizers
            - mention: what mention the token belongs to (or if not in any mention, None)
            - pem: same as mention, but for pems
            - speaker: the speaker of the utterance, either "USER" or "SYSTEM"

        Args:
            conv: a conversation
                E.g.,
                    {"speaker": "USER",
                    "utterance": "I agree. One of my favorite forms of science fiction is anything related to time travel! I find it fascinating.",
                    "mentions": ["science fiction", "time travel", ],
                    "pems": ["my favorite forms of science fiction", ],},
        """
        self._error_check(conv)
        ret = []
        for turn_num, turn in enumerate(conv):
            speaker = turn["speaker"]
            if speaker == "USER":
                ments = turn["mentions"] if "mentions" in turn else []
                pems = turn["pems"] if "pems" in turn else []
            elif speaker == "SYSTEM":
                ments = []
                pems = []
            else:
                raise ValueError(
                    f'Unknown speaker: {speaker}. Speaker must be either "USER" or "SYSTEM".'
                )
            tkn_info = self._tokens_info(turn["utterance"], speaker, ments, pems)
            for elem in tkn_info:
                elem["turn_number"] = turn_num
            ret += tkn_info
        return ret

    def _get_token_level_span(self, token_info: list, key_for_ment: str):
        """Get token-level spans for mentions and pems
        Token-level span is the input for s2e

        Args:
            token_info (list): E.g., [{'token': 'Blue',  'span': (0, 4),  'corresponding_ment': 'Blue',  'speaker': 'USER',  'turn_number': 0}, ...]
            key_ment_or_pem (str): 'mention' or 'pem'

        Returns:
            E.g., [[2, 5], [8, 9], [0, 3]]
        """
        # Error check
        assert key_for_ment in [
            "mention",
            "pem",
        ]  # key_for_ment should be mention or pem

        pem_and_eem = []
        start_pos, end_pos = None, None
        for i in range(len(token_info)):
            prev_ment = token_info[i - 1][key_for_ment] if i > 0 else None
            curr_ment = token_info[i][key_for_ment]
            futr_ment = (
                token_info[i + 1][key_for_ment] if (i + 1) < len(token_info) else None
            )

            if (prev_ment != curr_ment) and (curr_ment != None):  # mention start
                if start_pos == None:  # Error check
                    start_pos = i
                else:
                    raise ValueError("pos should be None to assign the value")
            if (futr_ment != curr_ment) and (curr_ment != None):
                # print(curr_ment,start_pos,end_pos)
                if end_pos == None:  # Error check
                    end_pos = i
                else:
                    raise ValueError("pos should be None to assign the value")

            # print(prev_ment,curr_ment,futr_ment,'\tSTART_END_POS:',start_pos,end_pos)

            if (start_pos != None) and (end_pos != None):
                # print('curr_ment:',curr_ment)
                pem_and_eem.append([start_pos, end_pos])
                start_pos, end_pos = None, None  # Initialize

        return pem_and_eem

    def get_input_of_pe_linking(self, token_info):
        """Get the input of PE Linking module

        Args:
            token_info (list): list of dict where tokens and their information is stored. Output of get_tokens_with_info()
                E.g.,
                    [{'token': 'I', # token
                    'span': (0, 1), # (start_pos, end_pos) of the token
                    'mention': None, # what mention the token belongs to (or if not in any mention, None)
                    'pem': None, # same as mention, but for pems
                    'speaker': 'USER', # the speaker of the utterance, either "USER" or "SYSTEM"
                    'turn_number': 0},  # turn number of the utterance which the token belongs to (0-based)
                    ...]

        Returns:
            Input of PE Linking module (the same input as s2e-coref)
        """
        ret = []
        pem_spans = self._get_token_level_span(token_info, "pem")
        # TODO: This is redundant and not efficient.
        # Instead `pem_spans` can be acquired by pos_to_token mapping, which maps char_position --> token_position
        # But I am too lazy to implement this code now. (Especially since it affects all other functions and data flows)

        for (
            pem_span
        ) in (
            pem_spans
        ):  # [[142, 143], [256, 258], [309, 310]]. Note that this is token-level, not char-level
            (start, end) = pem_span
            turn_num = token_info[start]["turn_number"]
            assert (
                turn_num == token_info[end]["turn_number"]
            ), f"Start token and end token should have the same turn_number. start: {start}, end: {end}"
            tokens_until_current_turn = [
                e for e in token_info if e["turn_number"] <= turn_num
            ]  # Extract tokens until the current turn

            ret.append(
                {
                    "clusters": [],  # Not used for inference
                    "doc_key": "tmp",  # Not used for inference
                    "mentions": self._get_token_level_span(
                        tokens_until_current_turn, "mention"
                    ),  # Detected mention spans, where format is (start_token_ind, end_token_ind) # E.g., [[28, 43], [67, 78]]. TODO: The same as above todo
                    "pems": [
                        pem_span
                    ],  # Detected personal entity mention span. The format is the same as mention spans # E.g., [[7, 43]]. NOTE: Currently our tool support only one mention at a time.
                    "sentences": [
                        [e["token"] for e in tokens_until_current_turn]
                    ],  # Tokenized sentences. E.g., ['I', 'think', 'science', 'fiction', 'is', ...]
                    "speakers": [
                        [e["speaker"] for e in tokens_until_current_turn]
                    ],  # Speaker information. E.g., ['SYSTEM', 'SYSTEM', ..., 'USER', 'USER', ...]
                    "text": None,
                }
            )
        return ret


class PostProcess:
    """Handle output of PE Linking module"""

    def _get_ment2score(
        self, doc_key: str, mentions: list, pems: list, scores: list, flag_print=False
    ) -> dict:
        """Get mention to score map

        Args:
            doc_key (str): E.g., 'dialind:1_turn:0_pem:My-favourite-type-of-cake'
            mentions: E.g., [[6, 7], [12, 12], [14, 15]]
            pems: E.g., [[0, 4]]
            scores: The scores for all mention (inc. PE) pair
                E.g., [{'doc_key': 'tmp', 'span_token_anaphora': [8, 8], 'span_token_antecedent': [0, 0], 'coref_logits': -66.80387115478516}, ...]

        Returns:
            {(6, 7): -42.52804183959961,
            (12, 12): -83.429443359375,
            (14, 15): -47.706520080566406}

        """
        assert all(
            [isinstance(m, list) for m in mentions]
        )  # Check both mentions and pems are 2d lists
        assert all([isinstance(m, list) for m in pems])  # The same for pems
        assert len(pems) == 1  # Check we only have one PEM
        if doc_key not in [sj["doc_key"] for sj in scores]:  # 220403
            if flag_print:
                print(
                    f"{doc_key} not in scores. It might be that EL tool could not detect any candidate EEMs for this PEM. Return empty dict."
                )
            return {}  # ment2score = {}

        # Change all ments and pems to tuple to be able to compare
        ment_tpl_list = [
            tuple(m) for m in mentions
        ]  # E.g., [(6, 7), (12, 12), (14, 15)]
        pem_tpl = tuple(pems[0])  # E.g., (0, 4)

        ment2score = {}
        span_hist = set()  # This is used to error check
        for sj in scores:
            if sj["doc_key"] == doc_key:
                # print(sj)
                span_ano = tuple(sj["span_token_anaphora"])  # E.g., (6, 7)
                span_ant = tuple(sj["span_token_antecedent"])  # E.g., (0, 4)
                span_hist.add(span_ano)
                span_hist.add(span_ant)

                if (
                    span_ano == pem_tpl and span_ant in ment_tpl_list
                ):  # anaphora is the PEM case
                    ment2score[span_ant] = sj["coref_logits"]
                elif (
                    span_ant == pem_tpl and span_ano in ment_tpl_list
                ):  # antecedent is the PEM case
                    ment2score[span_ano] = sj["coref_logits"]

        # Check all ment_tpl_list and pem_tpl are in span_hist
        assert sorted(ment_tpl_list + [pem_tpl]) == sorted(
            list(span_hist)
        ), f"mentions in score.json and pred.jsonl should be the same. {sorted(ment_tpl_list + [pem_tpl])} != {sorted(list(span_hist))}. doc_key: {doc_key}"
        return ment2score

    def _convert_to_mention_from_token(self, mention, comb_text):
        """
        Args:
            mention (list): [start, end] (this is token-level (NOT subtoken-level))
        """
        start = mention[0]  # output['subtoken_map'][mention[0]]
        end = mention[1] + 1  # output['subtoken_map'][mention[1]] + 1
        mtext = "".join(" ".join(comb_text[start:end]).split(" ##"))
        return mtext

    def get_results(self, pel_input, conv, threshold, scores):
        """Get PE Linking post-processed results

        Args:
            pel_input (dict): input for PE Linking module
                E.g., {'clusters': [], # Not used for inference
                       'doc_key': 'tmp', # Not used for inference
                       'mentions': [[2, 3], [77, 78]], # Detected concept and NE mentions
                       'pems': [[67, 72]], # Detected personal entity mention. Only one PEM is allowed now.
                       'sentences': [['I', 'think', 'science', 'fiction', 'is', ...]], # tokenized sentences using tokenizers.pre_tokenizers
                       'speakers': [['USER', 'USER', 'USER', 'USER', 'USER', ...]], # speaker information
                       'text': None
                       }
            threshold: default 0
            conv: The conversation input to preprocessing module (conversation before preprocessing)
            scores: The scores for all mention (inc. PE) pair
                E.g.,
                    [{'doc_key': 'tmp',
                    'span_token_anaphora': (67, 72), # This could be either a mention or a PEM
                    'span_token_antecedent': (2, 3), # The same as above
                    'coref_logits': -4.528693675994873}, # Output score from PE Linking module
        Returns:
            E.g.,
                [{'personal_entity_mention': 'my favorite forms of science fiction',
                'mention': 'time travel',
                'score': 4.445976734161377}]

        """
        assert type(pel_input) == dict, f"pel_input should be a dict. {type(pel_input)}"
        ments_span_tokenlevel = [m for m in pel_input["mentions"]]
        pems_span_tokenlevel = [p for p in pel_input["pems"]]  # len(pems) == 1
        assert (
            len(pems_span_tokenlevel) == 1
        ), f"len(pems_span_tokenlevel) should be 1. {len(pems_span_tokenlevel)}"

        mspan2score = self._get_ment2score(
            TMP_DOC_ID, ments_span_tokenlevel, pems_span_tokenlevel, scores
        )  # Mention span to score
        comb_text = pel_input["sentences"][
            0
        ]  # pel_input['sentences'] should have only one sentence

        pred_peas = []

        pem = [m for turn in conv if turn["speaker"] == "USER" for m in turn["pems"]][
            0
        ]  # Each conv has only one pem for current implementation
        for ment_span_tokenlevel in pel_input["mentions"]:
            score = mspan2score[tuple(ment_span_tokenlevel)]
            ment = self._convert_to_mention_from_token(ment_span_tokenlevel, comb_text)
            if score > threshold:
                pred_peas.append(
                    {"personal_entity_mention": pem, "mention": ment, "score": score}
                )
        return pred_peas
