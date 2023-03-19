from transformers import AutoModelForSeq2SeqLM, T5EncoderModel, AutoTokenizer
from torch import nn
import torch


class Model(nn.Module):
    def __init__(self, model_path, tokenizer_path=None, max_ranks_per_batch=2):
        super().__init__()
        model = T5EncoderModel.from_pretrained(model_path)
        self.config = model.config
        self.config.n_embd = self.config.hidden_size if hasattr(self.config, "hidden_size") else self.config.n_embd
        self.transformer = model
        self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)
        self.tokenizer_path = tokenizer_path if tokenizer_path is not None else model_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.PAD_ID = self.tokenizer(self.tokenizer.pad_token)["input_ids"][0]
        self.max_ranks_per_batch = max_ranks_per_batch

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            head_mask=None,
            inputs_embeds=None,
            return_dict=False,
            output_attentions=False,
            output_hidden_states=False,
    ):
        loss = None

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = transformer_outputs[0]

        rewards = self.v_head(hidden_states).squeeze(-1)

        bs = input_ids.shape[0] // self.max_ranks_per_batch

        ranked = [input_ids[i:i + bs] for i in range(0, len(input_ids), bs)]
        ranked_rewards = [rewards[i:i + bs] for i in range(0, len(rewards), bs)]

        end_scores = [list() for _ in range(self.max_ranks_per_batch)]
        for i in range(bs):
            for j in range(self.max_ranks_per_batch):
                end_scores[j].append(ranked_rewards[j][i][self.get_start_of_padding(ranked[j][i])])

        loss = torch.tensor(0.0, requires_grad=True).to("cuda")
        for i in range(bs):
            unpadded = [rank[i] for rank in ranked if
                        not torch.equal(rank[i], torch.tensor(self.PAD_ID).repeat(rank[i].shape).to("cuda"))]
            # If we don't have enough for a proper comparison, how should it be factored into the loss?
            if len(unpadded) > 1:
                pairwise_rewards = torch.stack(
                    [self.get_pairwise_reward(ranked[j][i], ranked[k][i], ranked_rewards[j][i], ranked_rewards[k][i])
                     for (j, k) in all_pairs(len(unpadded))])
                mean = torch.mean(pairwise_rewards[~torch.any(pairwise_rewards.isnan())])
                # Ideally, we'd just filter, not set the nan to zero
                loss += torch.mean(torch.where(torch.isnan(mean).to("cuda"), torch.tensor(0.0).to("cuda"), mean))
        loss = loss / bs

        return {
            "end_scores": end_scores,
            "loss": loss
        }

    def get_pairwise_reward(self, input1, input2, reward1, reward2):
        end_of_prompt = self.get_end_of_prompt(input1, input2)
        start_of_padding = max(self.get_start_of_padding(input1), self.get_start_of_padding(input2))
        return -torch.log(
            torch.sigmoid(reward1[end_of_prompt:start_of_padding] - reward2[end_of_prompt:start_of_padding])).mean()

    def get_end_of_prompt(self, input1, input2):
        inds = (input1 != input2).nonzero()
        return inds[0] if len(inds) > 0 else len(input1) - 1

    def get_start_of_padding(self, input1):
        inds = (input1 == self.PAD_ID).nonzero()
        return inds[0].item() if len(inds) > 0 else len(input1) - 1

    def loss(self, outputs, batch):
        return outputs["loss"]


def all_pairs(n):
    return [(i, j) for i in range(n) for j in range(i+1, n)]
