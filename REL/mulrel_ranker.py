import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

"""
PreRank class is used for preranking entities for a given mention by multiplying entity vectors with
word vectors
"""


class PreRank(torch.nn.Module):
    def __init__(self, config, embeddings=None):
        super(PreRank, self).__init__()
        self.config = config

    def forward(self, token_ids, token_offsets, entity_ids, embeddings, emb):
        """
        Multiplies local context words with entity vectors for a given mention.

        :return: entity scores.
        """

        sent_vecs = embeddings["word_embeddings_bag"](
            token_ids, token_offsets
        )  # (batch_size, emb_size=300)

        # entity_vecs = emb.emb(entity_names)

        entity_vecs = embeddings["entity_embeddings"](
            entity_ids
        )  # (batch_size, n_cands, emb_size)

        # compute scores
        batchsize, dims = sent_vecs.size()
        n_entities = entity_vecs.size(1)
        scores = torch.bmm(entity_vecs, sent_vecs.view(batchsize, dims, 1))
        scores = scores.view(batchsize, n_entities)

        log_probs = F.log_softmax(scores, dim=1)
        return log_probs


"""
Multi-relational global model with context token attention, using loopy belief propagation. 
With local model context token attention (from G&H's EMNLP paper).

Function descriptions will refer to paper.

Author: Phong Le 
Paper: Improving Entity Linking by Modeling Latent Relations between Mentions
"""


class MulRelRanker(torch.nn.Module):
    def __init__(self, config, device):
        super(MulRelRanker, self).__init__()
        self.config = config
        # self.embeddings = embeddings
        self.device = device
        self.max_dist = 1000
        self.ent_top_n = 1000
        self.ent_ent_comp = "bilinear"  # config.get('ent_ent_comp', 'bilinear')  # bilinear, trans_e, fbilinear

        self.att_mat_diag = torch.nn.Parameter(torch.ones(self.config["emb_dims"]))
        self.tok_score_mat_diag = torch.nn.Parameter(
            torch.ones(self.config["emb_dims"])
        )

        self.score_combine_linear_1 = torch.nn.Linear(2, self.config["hid_dims"])
        self.score_combine_act_1 = torch.nn.ReLU()
        self.score_combine_linear_2 = torch.nn.Linear(self.config["hid_dims"], 1)

        if self.config["use_local"]:
            self.ent_localctx_comp = torch.nn.Parameter(
                torch.ones(self.config["emb_dims"])
            )

        if self.config["use_pad_ent"]:
            self.pad_ent_emb = torch.nn.Parameter(
                torch.randn(1, self.config["emb_dims"]) * 0.1
            )
            self.pad_ctx_vec = torch.nn.Parameter(
                torch.randn(1, self.config["emb_dims"]) * 0.1
            )

        self.ctx_layer = torch.nn.Sequential(
            torch.nn.Linear(self.config["emb_dims"] * 3, self.config["emb_dims"]),
            torch.nn.Tanh(),
            torch.nn.Dropout(p=self.config["dropout_rate"]),
        )

        self.rel_embs = (
            torch.randn(self.config["n_rels"], self.config["emb_dims"]) * 0.01
        )
        self.rel_embs[0] = 1 + torch.randn(self.config["emb_dims"]) * 0.01
        self.rel_embs = torch.nn.Parameter(self.rel_embs)

        self.ew_embs = torch.nn.Parameter(
            torch.randn(self.config["n_rels"], self.config["emb_dims"]) * 0.01
        )
        self._coh_ctx_vecs = None

        self.score_combine = torch.nn.Sequential(
            torch.nn.Linear(2, self.config["hid_dims"]),
            torch.nn.ReLU(),
            torch.nn.Linear(self.config["hid_dims"], 1),
        )

    def __local_ent_scores(
        self, token_ids, tok_mask, entity_ids, entity_mask, embeddings, p_e_m=None
    ):
        """
        Local entity scores

        :return: Entity scores.
        """

        batchsize, n_words = token_ids.size()
        n_entities = entity_ids.size(1)
        tok_mask = tok_mask.view(batchsize, 1, -1)

        tok_vecs = embeddings["word_embeddings"](token_ids)
        entity_vecs = embeddings["entity_embeddings"](entity_ids)

        ent_tok_att_scores = torch.bmm(
            entity_vecs * self.att_mat_diag, tok_vecs.permute(0, 2, 1)
        )
        ent_tok_att_scores = (ent_tok_att_scores * tok_mask).add_(
            (tok_mask - 1).mul_(1e10)
        )
        tok_att_scores, _ = torch.max(ent_tok_att_scores, dim=1)
        top_tok_att_scores, top_tok_att_ids = torch.topk(
            tok_att_scores, dim=1, k=min(self.config["tok_top_n"], n_words)
        )
        att_probs = F.softmax(top_tok_att_scores, dim=1).view(batchsize, -1, 1)
        att_probs = att_probs / torch.sum(att_probs, dim=1, keepdim=True)

        selected_tok_vecs = torch.gather(
            tok_vecs,
            dim=1,
            index=top_tok_att_ids.view(batchsize, -1, 1).repeat(1, 1, tok_vecs.size(2)),
        )
        ctx_vecs = torch.sum(
            (selected_tok_vecs * self.tok_score_mat_diag) * att_probs,
            dim=1,
            keepdim=True,
        )
        ent_ctx_scores = torch.bmm(entity_vecs, ctx_vecs.permute(0, 2, 1)).view(
            batchsize, n_entities
        )

        # combine with p(e|m) if p_e_m is not None
        if p_e_m is not None:
            inputs = torch.cat(
                [
                    ent_ctx_scores.view(batchsize * n_entities, -1),
                    torch.log(p_e_m + 1e-20).view(batchsize * n_entities, -1),
                ],
                dim=1,
            )
            hidden = self.score_combine_linear_1(inputs)
            hidden = self.score_combine_act_1(hidden)
            scores = self.score_combine_linear_2(hidden).view(batchsize, n_entities)
        else:
            scores = ent_ctx_scores

        scores = (scores * entity_mask).add_((entity_mask - 1).mul_(1e10))

        self._entity_vecs = entity_vecs
        self._local_ctx_vecs = ctx_vecs

        return scores

    def forward(
        self, token_ids, tok_mask, entity_ids, entity_mask, p_e_m, embeddings, gold=None
    ):
        """
        Responsible for forward pass of ED model and produces a ranking of candidates for a given set of mentions.

        - ctx_layer refers to function f. See Figure 3 in respective paper.
        - ent_scores refers to function q.
        - score_combine refers to function g.

        :return: Ranking of entities per mention.
        """

        n_ments, n_cands = entity_ids.size()
        n_rels = self.config["n_rels"]

        if self.config["use_local"]:
            local_ent_scores = self.__local_ent_scores(
                token_ids, tok_mask, entity_ids, entity_mask, embeddings, p_e_m=None
            )
            ent_vecs = self._entity_vecs
        else:
            ent_vecs = embeddings["entity_embeddings"](entity_ids)
            local_ent_scores = Variable(
                torch.zeros(n_ments, n_cands), requires_grad=False
            ).to(self.device)

        # compute context vectors
        ltok_vecs = embeddings["snd_embeddings"](
            self.s_ltoken_ids
        ) * self.s_ltoken_mask.view(n_ments, -1, 1)
        local_lctx_vecs = torch.sum(ltok_vecs, dim=1) / torch.sum(
            self.s_ltoken_mask, dim=1, keepdim=True
        ).add_(1e-5)
        rtok_vecs = embeddings["snd_embeddings"](
            self.s_rtoken_ids
        ) * self.s_rtoken_mask.view(n_ments, -1, 1)
        local_rctx_vecs = torch.sum(rtok_vecs, dim=1) / torch.sum(
            self.s_rtoken_mask, dim=1, keepdim=True
        ).add_(1e-5)
        mtok_vecs = embeddings["snd_embeddings"](
            self.s_mtoken_ids
        ) * self.s_mtoken_mask.view(n_ments, -1, 1)
        ment_vecs = torch.sum(mtok_vecs, dim=1) / torch.sum(
            self.s_mtoken_mask, dim=1, keepdim=True
        ).add_(1e-5)
        bow_ctx_vecs = torch.cat([local_lctx_vecs, ment_vecs, local_rctx_vecs], dim=1)

        if self.config["use_pad_ent"]:
            ent_vecs = torch.cat(
                [ent_vecs, self.pad_ent_emb.view(1, 1, -1).repeat(1, n_cands, 1)], dim=0
            )
            tmp = torch.zeros(1, n_cands)
            tmp[0, 0] = 1
            tmp = Variable(tmp).to(self.device)
            entity_mask = torch.cat([entity_mask, tmp], dim=0)
            p_e_m = torch.cat([p_e_m, tmp], dim=0)
            local_ent_scores = torch.cat(
                [
                    local_ent_scores,
                    Variable(torch.zeros(1, n_cands), requires_grad=False).to(
                        self.device
                    ),
                ],
                dim=0,
            )
            n_ments += 1

        if self.config["use_local_only"]:
            inputs = torch.cat(
                [
                    Variable(torch.zeros(n_ments * n_cands, 1)).to(self.device),
                    local_ent_scores.view(n_ments * n_cands, -1),
                    torch.log(p_e_m + 1e-20).view(n_ments * n_cands, -1),
                ],
                dim=1,
            )
            scores = self.score_combine(inputs).view(n_ments, n_cands)
            return scores

        if n_ments == 1:
            ent_scores = local_ent_scores

        else:
            # distance - to consider only neighbor mentions
            ment_pos = torch.arange(0, n_ments).long()
            dist = (ment_pos.view(n_ments, 1) - ment_pos.view(1, n_ments)).abs()
            dist.masked_fill_(dist == 1, -1)
            dist.masked_fill_((dist > 1) & (dist <= self.max_dist), -1)
            dist.masked_fill_(dist > self.max_dist, 0)
            dist.mul_(-1)

            ctx_vecs = self.ctx_layer(bow_ctx_vecs)
            if self.config["use_pad_ent"]:
                ctx_vecs = torch.cat([ctx_vecs, self.pad_ctx_vec], dim=0)

            m1_ctx_vecs, m2_ctx_vecs = ctx_vecs, ctx_vecs
            rel_ctx_vecs = m1_ctx_vecs.view(1, n_ments, -1) * self.ew_embs.view(
                n_rels, 1, -1
            )
            rel_ctx_ctx_scores = torch.matmul(
                rel_ctx_vecs, m2_ctx_vecs.view(1, n_ments, -1).permute(0, 2, 1)
            )  # n_rels x n_ments x n_ments

            rel_ctx_ctx_scores = rel_ctx_ctx_scores.add_(
                (1 - Variable(dist.float()).to(self.device)).mul_(-1e10)
            )
            eye = Variable(torch.eye(n_ments)).view(1, n_ments, n_ments).to(self.device)
            rel_ctx_ctx_scores.add_(eye.mul_(-1e10))
            rel_ctx_ctx_scores.mul_(
                1 / np.sqrt(self.config["emb_dims"])
            )  # scaling proposed by "attention is all you need"

            # get top_n neighbour
            if self.ent_top_n < n_ments:
                topk_values, _ = torch.topk(
                    rel_ctx_ctx_scores, k=min(self.ent_top_n, n_ments), dim=2
                )
                threshold = topk_values[:, :, -1:]
                mask = 1 - (rel_ctx_ctx_scores >= threshold).float()
                rel_ctx_ctx_scores.add_(mask.mul_(-1e10))

            rel_ctx_ctx_probs = F.softmax(rel_ctx_ctx_scores, dim=2)
            rel_ctx_ctx_weights = rel_ctx_ctx_probs + rel_ctx_ctx_probs.permute(0, 2, 1)
            self._rel_ctx_ctx_weights = rel_ctx_ctx_probs

            # compute phi(ei, ej)
            rel_ent_vecs = ent_vecs.view(1, n_ments, n_cands, -1) * self.rel_embs.view(
                n_rels, 1, 1, -1
            )
            rel_ent_ent_scores = torch.matmul(
                rel_ent_vecs.view(n_rels, n_ments, 1, n_cands, -1),
                ent_vecs.view(1, 1, n_ments, n_cands, -1).permute(0, 1, 2, 4, 3),
            )

            rel_ent_ent_scores = rel_ent_ent_scores.permute(
                0, 1, 3, 2, 4
            )  # n_rel x n_ments x n_cands x n_ments x n_cands
            rel_ent_ent_scores = (rel_ent_ent_scores * entity_mask).add_(
                (entity_mask - 1).mul_(1e10)
            )
            ent_ent_scores = torch.sum(
                rel_ent_ent_scores
                * rel_ctx_ctx_weights.view(n_rels, n_ments, 1, n_ments, 1),
                dim=0,
            ).mul(
                1.0 / n_rels
            )  # n_ments x n_cands x n_ments x n_cands

            # LBP
            prev_msgs = Variable(torch.zeros(n_ments, n_cands, n_ments)).to(self.device)

            for _ in range(self.config["n_loops"]):
                mask = 1 - Variable(torch.eye(n_ments)).to(self.device)
                ent_ent_votes = (
                    ent_ent_scores
                    + local_ent_scores * 1
                    + torch.sum(
                        prev_msgs.view(1, n_ments, n_cands, n_ments)
                        * mask.view(n_ments, 1, 1, n_ments),
                        dim=3,
                    ).view(n_ments, 1, n_ments, n_cands)
                )
                msgs, _ = torch.max(ent_ent_votes, dim=3)
                msgs = (
                    F.softmax(msgs, dim=1).mul(self.config["dropout_rate"])
                    + prev_msgs.exp().mul(1 - self.config["dropout_rate"])
                ).log()
                prev_msgs = msgs

            # compute marginal belief
            mask = 1 - Variable(torch.eye(n_ments)).to(self.device)
            ent_scores = local_ent_scores * 1 + torch.sum(
                msgs * mask.view(n_ments, 1, n_ments), dim=2
            )
            ent_scores = F.softmax(ent_scores, dim=1)

        # combine with p_e_m
        inputs = torch.cat(
            [
                ent_scores.view(n_ments * n_cands, -1),
                torch.log(p_e_m + 1e-20).view(n_ments * n_cands, -1),
            ],
            dim=1,
        )
        scores = self.score_combine(inputs).view(n_ments, n_cands)
        # scores = F.softmax(scores, dim=1)

        if self.config["use_pad_ent"]:
            scores = scores[:-1]
        return scores, ent_scores

    def regularize(self, max_norm=1):
        """
        Regularises model parameters.

        :return: -
        """

        l1_w_norm = self.score_combine_linear_1.weight.norm()
        l1_b_norm = self.score_combine_linear_1.bias.norm()
        l2_w_norm = self.score_combine_linear_2.weight.norm()
        l2_b_norm = self.score_combine_linear_2.bias.norm()

        if (l1_w_norm > max_norm).data.all():
            self.score_combine_linear_1.weight.data = (
                self.score_combine_linear_1.weight.data * max_norm / l1_w_norm.data
            )
        if (l1_b_norm > max_norm).data.all():
            self.score_combine_linear_1.bias.data = (
                self.score_combine_linear_1.bias.data * max_norm / l1_b_norm.data
            )
        if (l2_w_norm > max_norm).data.all():
            self.score_combine_linear_2.weight.data = (
                self.score_combine_linear_2.weight.data * max_norm / l2_w_norm.data
            )
        if (l2_b_norm > max_norm).data.all():
            self.score_combine_linear_2.bias.data = (
                self.score_combine_linear_2.bias.data * max_norm / l2_b_norm.data
            )

    def loss(self, scores, true_pos, lamb=1e-7):
        """
        Computes given ranking loss (Equation 7) and adds a regularization term.

        :return: loss of given batch
        """
        loss = F.multi_margin_loss(scores, true_pos, margin=self.config["margin"])
        if self.config["use_local_only"]:
            return loss

        # regularization
        X = F.normalize(self.rel_embs)
        diff = (
            (
                X.view(self.config["n_rels"], 1, -1)
                - X.view(1, self.config["n_rels"], -1)
            )
            .pow(2)
            .sum(dim=2)
            .add_(1e-5)
            .sqrt()
        )
        diff = diff * (diff < 1).float()
        loss -= torch.sum(diff).mul(lamb)

        X = F.normalize(self.ew_embs)
        diff = (
            (
                X.view(self.config["n_rels"], 1, -1)
                - X.view(1, self.config["n_rels"], -1)
            )
            .pow(2)
            .sum(dim=2)
            .add_(1e-5)
            .sqrt()
        )
        diff = diff * (diff < 1).float()
        loss -= torch.sum(diff).mul(lamb)
        return loss
